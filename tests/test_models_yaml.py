from pathlib import Path

import yaml

MODELS = Path(__file__).resolve().parents[1] / "conf" / "models" / "models.yaml"


def test_model_entry_names_and_paths_have_no_whitespace():
    """A run list pasted as one string once became a single entry named 'A B C ...' with a matching
    adapter path; every attack on those models then failed at startup with 'Key ... is not in struct'."""
    entries = yaml.safe_load(MODELS.read_text())
    bad = [name for name, e in entries.items()
           if any(c.isspace() for c in name)
           or any(c.isspace() for c in str((e or {}).get("adapter_path") or ""))]
    assert not bad, bad
