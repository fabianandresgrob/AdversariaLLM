import pandas as pd

from pretrain_prompt_probe import split_groups
from score_prompt_probe import esg_val_prompts, sha1


def test_esg_val_prompts_are_the_rows_pretraining_held_out(tmp_path):
    df = pd.DataFrame({
        "row_id": [f"p{i:03d}" for i in range(40)],
        "text": [f"prompt {i}" if i != 7 else " " for i in range(40)],
        "label": [str(i % 2) for i in range(40)],
        "source": ["plain" if i % 4 == 0 else "jf" for i in range(40)],
        "input_id": [f"g{i // 2}" for i in range(40)],
        "split": ["train"] * 40,
    })
    path = tmp_path / "probe.parquet"
    df.to_parquet(path)
    rows = esg_val_prompts(str(path), 0.25, 0)

    kept = df[df.text.str.strip() != ""].sort_values("row_id").reset_index(drop=True)
    _, val_idx = split_groups(kept.input_id.tolist(), 0.25, 0)
    assert {r["row_id"] for r in rows.values()} == set(kept.row_id[val_idx])
    assert "p007" not in {r["row_id"] for r in rows.values()}
    one = rows[sha1(kept.text[val_idx[0]])]
    assert one["label"] in (0, 1) and one["esg_split"] == "train" and one["esg_source"] in ("plain", "jf")
