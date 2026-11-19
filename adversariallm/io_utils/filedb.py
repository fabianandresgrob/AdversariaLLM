"""File-backed stand-in for the pymongo `runs` collection, taken from jsc-jobs (commit 3b6b9d5).

Selected with ADVLLM_DB=file:<dir>. Supports exactly the operations this repo uses. Storage: one
append-only JSONL shard per (attack, model); the latest line per key wins; writers hold a per-shard
file lock; readers take no lock and ignore a torn trailing line. The lock is fcntl.lockf, so the
directory must support POSIX locks: on the MPI cluster /home (NFS) does, /fast (Lustre, noflock) does not.
"""

import copy
import fcntl
import hashlib
import json
import os
import re
import threading
import time
from contextlib import contextmanager
from pathlib import Path

MISC_SHARD = "_misc/_misc.jsonl"
_SLUG_BAD = re.compile(r"[^A-Za-z0-9._,=+-]")
_FILTER_KEYS = {"config", "log_file", "_id"}


def db_root_from_env() -> Path:
    value = os.environ.get("ADVLLM_DB", "")
    if not value.startswith("file:"):
        raise RuntimeError("ADVLLM_DB is not a file database (expected file:<dir>)")
    return Path(os.path.expandvars(value[len("file:"):]))


def _canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_key(config) -> str:
    return "sha256:" + hashlib.sha256(_canonical(config).encode()).hexdigest()


def _slug(text) -> str:
    return _SLUG_BAD.sub("_", str(text).replace("/", "__"))


def shard_for(config) -> str:
    config = config if isinstance(config, dict) else {}
    attack, model = config.get("attack"), config.get("model")
    if isinstance(attack, str) and isinstance(model, str) and attack and model:
        return f"{_slug(attack)}/{_slug(model)}.jsonl"
    return MISC_SHARD


def _check_filter(flt) -> None:
    if not isinstance(flt, dict):
        raise TypeError(f"filter must be a dict, got {type(flt).__name__}")
    unsupported = set(flt) - _FILTER_KEYS
    if unsupported:
        raise NotImplementedError(f"file DB does not support filter keys {sorted(unsupported)}")
    for key in ("log_file", "_id"):
        if isinstance(flt.get(key), dict):
            raise NotImplementedError(f"file DB does not support operators in filter on {key!r}")


class _PosixFileLock:
    """Cross-node lock on GPFS: fcntl.lockf takes byte-range locks coordinated cluster-wide by the
    filesystem (flock is not). lockf locks are per-process, so a per-shard threading.Lock is held
    around it to also exclude threads within one process."""

    def __init__(self, path: str):
        self.path = path
        self._thread_lock = threading.Lock()
        self._fd = None

    def __enter__(self):
        self._thread_lock.acquire()
        try:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
            self._fd = os.open(self.path, os.O_CREAT | os.O_RDWR, 0o644)
            fcntl.lockf(self._fd, fcntl.LOCK_EX)
        except BaseException:
            self._thread_lock.release()
            raise
        return self

    def __exit__(self, *exc_info) -> None:
        try:
            fcntl.lockf(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
        finally:
            self._fd = None
            self._thread_lock.release()


class _Shard:
    def __init__(self, root: Path, rel: str):
        self.rel = rel
        self.path = root / rel
        self.lock = _PosixFileLock(str(self.path) + ".lock")
        self.lines = 0
        self._signature = None
        self._index: dict[str, dict] = {}

    def load(self, force: bool = False) -> dict:
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self._signature, self._index, self.lines = None, {}, 0
            return self._index
        signature = (st.st_size, st.st_mtime_ns, st.st_ino)
        if not force and signature == self._signature:
            return self._index
        index, lines = {}, 0
        for chunk in self.path.read_bytes().split(b"\n")[:-1]:  # the part after the last newline is torn or empty
            if not chunk.strip():
                continue
            try:
                record = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            lines += 1
            if record.get("op") == "put":
                index[record["key"]] = record["doc"]
            elif record.get("op") == "del":
                index.pop(record["key"], None)
        self._signature, self._index, self.lines = signature, index, lines
        return index

    def append(self, records: list[dict]) -> None:
        """Caller must hold self.lock."""
        payload = "".join(_canonical(r) + "\n" for r in records).encode()
        with open(self.path, "ab+") as f:
            f.seek(0, os.SEEK_END)
            if f.tell() > 0:
                f.seek(-1, os.SEEK_END)
                if f.read(1) != b"\n":
                    payload = b"\n" + payload  # isolate a torn line left by a crashed writer
                f.seek(0, os.SEEK_END)
            f.write(payload)
            f.flush()
            os.fsync(f.fileno())


@contextmanager
def _locked(shard: _Shard):
    shard.path.parent.mkdir(parents=True, exist_ok=True)
    with shard.lock:
        yield


def _put(key: str, doc: dict) -> dict:
    return {"op": "put", "key": key, "doc": doc, "ts": time.time()}


class FileCollection:
    def __init__(self, root):
        self.root = Path(root)
        self._shards: dict[str, _Shard] = {}
        self._guard = threading.Lock()

    def _shard(self, rel: str) -> _Shard:
        with self._guard:
            if rel not in self._shards:
                self._shards[rel] = _Shard(self.root, rel)
            return self._shards[rel]

    def _all_shards(self) -> list[_Shard]:
        if not self.root.is_dir():
            return []
        return [self._shard(p.relative_to(self.root).as_posix()) for p in sorted(self.root.glob("**/*.jsonl"))]

    def _shards_for(self, flt: dict) -> list[_Shard]:
        _check_filter(flt)
        if "config" in flt:
            return [self._shard(shard_for(flt["config"]))]
        return self._all_shards()

    @staticmethod
    def _matches(doc: dict, flt: dict) -> bool:
        if "config" in flt and config_key(doc.get("config")) != config_key(flt["config"]):
            return False
        if "log_file" in flt and doc.get("log_file") != flt["log_file"]:
            return False
        if "_id" in flt and doc.get("_id") != flt["_id"]:
            return False
        return True

    def find_one(self, filter=None):
        flt = filter or {}
        if "config" in flt:
            doc = self._shards_for(flt)[0].load().get(config_key(flt["config"]))
            return copy.deepcopy(doc) if doc is not None and self._matches(doc, flt) else None
        for shard in self._shards_for(flt):
            for doc in shard.load().values():
                if self._matches(doc, flt):
                    return copy.deepcopy(doc)
        return None

    def find(self, filter=None, projection=None) -> list[dict]:
        flt = filter or {}
        docs = [copy.deepcopy(d) for s in self._shards_for(flt) for d in s.load().values() if self._matches(d, flt)]
        if projection:
            if not all(projection.values()):
                raise NotImplementedError("file DB supports only inclusion projections")
            keep = set(projection) | {"_id"}
            docs = [{k: v for k, v in d.items() if k in keep} for d in docs]
        return docs

    def replace_one(self, filter, replacement, upsert=False) -> None:
        _check_filter(filter)
        if set(filter) != {"config"}:
            raise NotImplementedError("replace_one supports only {'config': ...} filters")
        key = config_key(filter["config"])
        shard = self._shard(shard_for(filter["config"]))
        with _locked(shard):
            if not upsert and key not in shard.load(force=True):
                return
            shard.append([_put(key, {**replacement, "_id": key})])

    def insert_many(self, documents) -> None:
        by_shard: dict[str, list[dict]] = {}
        for doc in documents:
            if "config" not in doc:
                raise NotImplementedError("insert_many needs documents with a 'config' field")
            key = config_key(doc["config"])
            by_shard.setdefault(shard_for(doc["config"]), []).append(_put(key, {**doc, "_id": key}))
        for rel, records in by_shard.items():
            shard = self._shard(rel)
            with _locked(shard):
                shard.append(records)

    def update_many(self, filter, update) -> None:
        if set(update) != {"$addToSet"}:
            raise NotImplementedError(f"update_many supports only $addToSet, got {sorted(update)}")
        for shard in self._shards_for(filter):
            with _locked(shard):
                records = []
                for key, doc in shard.load(force=True).items():
                    if not self._matches(doc, filter):
                        continue
                    new = copy.deepcopy(doc)
                    changed = False
                    for field, value in update["$addToSet"].items():
                        values = new.setdefault(field, [])
                        if value not in values:
                            values.append(value)
                            changed = True
                    if changed:
                        records.append(_put(key, new))
                if records:
                    shard.append(records)

    def delete_one(self, filter) -> None:
        for shard in self._shards_for(filter):
            with _locked(shard):
                for key, doc in shard.load(force=True).items():
                    if self._matches(doc, filter):
                        shard.append([{"op": "del", "key": key, "ts": time.time()}])
                        return

    def stats(self) -> dict[str, dict]:
        out = {}
        for shard in self._all_shards():
            live = len(shard.load(force=True))
            out[shard.rel] = {"live": live, "lines": shard.lines, "bytes": shard.path.stat().st_size}
        return out

    def compact(self, rel: str | None = None) -> list[str]:
        shards = [self._shard(rel)] if rel else self._all_shards()
        for shard in shards:
            with _locked(shard):
                index = shard.load(force=True)
                tmp = shard.path.with_name(shard.path.name + ".compact.tmp")
                with open(tmp, "wb") as f:
                    for key, doc in index.items():
                        f.write((_canonical(_put(key, doc)) + "\n").encode())
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, shard.path)
                shard.load(force=True)
        return [s.rel for s in shards]


class FileDatabase:
    def __init__(self, root):
        self.root = Path(root)
        self.runs = FileCollection(self.root)

    def __getitem__(self, name: str) -> FileCollection:
        if name != "runs":
            raise NotImplementedError("file DB only provides the 'runs' collection")
        return self.runs
