import multiprocessing
import threading

import pytest

from adversariallm.io_utils.filedb import FileDatabase, db_root_from_env

MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
SHARD = "gcg/meta-llama__Meta-Llama-3.1-8B-Instruct.jsonl"


def cfg(i):
    return {"attack": "gcg", "model": MODEL, "dataset": "adv_behaviors", "dataset_params": {"idx": i}}


def test_replace_find_one_roundtrip(tmp_path):
    runs = FileDatabase(tmp_path).runs
    assert runs.find_one({"config": cfg(0)}) is None
    runs.replace_one({"config": cfg(0)}, {"config": cfg(0), "log_file": "/a.json", "scored_by": []}, upsert=True)
    doc = runs.find_one({"config": cfg(0)})
    assert doc["log_file"] == "/a.json" and doc["_id"].startswith("sha256:")
    assert (tmp_path / SHARD).exists()
    runs.replace_one({"config": cfg(0)}, {"config": cfg(0), "log_file": "/b.json", "scored_by": []}, upsert=True)
    assert runs.find_one({"config": cfg(0)})["log_file"] == "/b.json"
    assert len(runs.find()) == 1


def test_key_order_does_not_matter(tmp_path):
    runs = FileDatabase(tmp_path).runs
    runs.replace_one({"config": cfg(0)}, {"config": cfg(0), "log_file": "/a"}, upsert=True)
    reordered = dict(reversed(list(cfg(0).items())))
    assert runs.find_one({"config": reordered}) is not None


def test_replace_without_upsert_is_noop(tmp_path):
    runs = FileDatabase(tmp_path).runs
    runs.replace_one({"config": cfg(0)}, {"config": cfg(0), "log_file": "/a"}, upsert=False)
    assert runs.find() == []


def test_other_process_sees_writes(tmp_path):
    FileDatabase(tmp_path).runs.replace_one({"config": cfg(0)}, {"config": cfg(0), "log_file": "/a"}, upsert=True)
    reader = FileDatabase(tmp_path).runs
    assert reader.find_one({"config": cfg(0)}) is not None
    FileDatabase(tmp_path).runs.replace_one({"config": cfg(1)}, {"config": cfg(1), "log_file": "/b"}, upsert=True)
    assert reader.find_one({"config": cfg(1)}) is not None


def test_find_by_log_file_projection_and_add_to_set(tmp_path):
    runs = FileDatabase(tmp_path).runs
    runs.insert_many([
        {"config": cfg(0), "log_file": "/a", "scored_by": []},
        {"config": cfg(1), "log_file": "/b", "scored_by": []},
    ])
    runs.update_many({"log_file": "/a"}, {"$addToSet": {"scored_by": "harmbench"}})
    runs.update_many({"log_file": "/a"}, {"$addToSet": {"scored_by": "harmbench"}})
    assert runs.find({"log_file": "/a"})[0]["scored_by"] == ["harmbench"]
    projected = runs.find({}, {"log_file": 1})
    assert sorted(d["log_file"] for d in projected) == ["/a", "/b"]
    assert set(projected[0]) == {"_id", "log_file"}


def test_delete_one_and_misc_shard(tmp_path):
    runs = FileDatabase(tmp_path).runs
    runs.insert_many([{"config": {"no": "attack"}, "log_file": "/m"}])
    assert (tmp_path / "_misc" / "_misc.jsonl").exists()
    doc = runs.find_one({"log_file": "/m"})
    runs.delete_one({"_id": doc["_id"]})
    assert runs.find() == [] and FileDatabase(tmp_path).runs.find() == []


def test_torn_trailing_line_is_ignored_and_repaired(tmp_path):
    runs = FileDatabase(tmp_path).runs
    runs.replace_one({"config": cfg(0)}, {"config": cfg(0), "log_file": "/a"}, upsert=True)
    with open(tmp_path / SHARD, "ab") as f:
        f.write(b'{"op":"put","key":"sha256:x","do')
    fresh = FileDatabase(tmp_path).runs
    assert len(fresh.find()) == 1
    fresh.replace_one({"config": cfg(1)}, {"config": cfg(1), "log_file": "/b"}, upsert=True)
    assert len(FileDatabase(tmp_path).runs.find()) == 2


def test_unsupported_operations_raise(tmp_path):
    db = FileDatabase(tmp_path)
    with pytest.raises(NotImplementedError):
        db.runs.find({"attack": "gcg"})
    with pytest.raises(NotImplementedError):
        db.runs.find({"log_file": {"$in": ["/a"]}})
    with pytest.raises(NotImplementedError):
        db.runs.update_many({"log_file": "/a"}, {"$set": {"x": 1}})
    with pytest.raises(NotImplementedError):
        db.runs.find({}, {"config": 0})
    with pytest.raises(NotImplementedError):
        db["other"]
    assert db["runs"] is db.runs


def test_stats_and_compact(tmp_path):
    runs = FileDatabase(tmp_path).runs
    for _ in range(3):
        runs.replace_one({"config": cfg(0)}, {"config": cfg(0), "log_file": "/a"}, upsert=True)
    assert runs.stats()[SHARD]["lines"] == 3 and runs.stats()[SHARD]["live"] == 1
    assert runs.compact() == [SHARD]
    assert runs.stats()[SHARD]["lines"] == 1
    assert runs.find_one({"config": cfg(0)})["log_file"] == "/a"


def _writer(args):
    root, worker = args
    runs = FileDatabase(root).runs
    for i in range(25):
        c = cfg(worker * 1000 + i)
        runs.replace_one({"config": c}, {"config": c, "log_file": f"/{worker}/{i}"}, upsert=True)


def test_concurrent_writers(tmp_path):
    with multiprocessing.get_context("spawn").Pool(8) as pool:
        pool.map(_writer, [(tmp_path, w) for w in range(8)])
    runs = FileDatabase(tmp_path).runs
    assert len(runs.find()) == 200
    assert runs.stats()[SHARD]["lines"] == 200


def test_concurrent_threads_in_one_process(tmp_path):
    runs = FileDatabase(tmp_path).runs

    def writer(worker):
        for i in range(25):
            c = cfg(worker * 1000 + i)
            runs.replace_one({"config": c}, {"config": c, "log_file": f"/{worker}/{i}"}, upsert=True)

    threads = [threading.Thread(target=writer, args=(w,)) for w in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(runs.find()) == 200
    assert runs.stats()[SHARD]["lines"] == 200


def test_db_root_from_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ADVLLM_DB", f"file:{tmp_path}")
    assert db_root_from_env() == tmp_path
    monkeypatch.setenv("ADVLLM_DB", "mongodb://x")
    with pytest.raises(RuntimeError):
        db_root_from_env()
