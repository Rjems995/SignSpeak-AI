import json
from collections import Counter
from pathlib import Path
import pytest
from ml.data_quality import curate, coverage
from ml.balance_dataset import balance
from ml.prepare import prepare
from ml.train import split_indices
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
REFS = json.loads((ROOT / "signspeak/static/models/references.json").read_text())


def record(**updates):
    return {"letter": "L", "points": REFS["L"], "participant": "one", "session": "visit1", "hand": "left", "lighting": "soft", "angle": "front", "label_review": "collector_confirmed", **updates}


def test_prepare_preserves_collection_metadata_and_conflicts_are_not_first_label_wins(tmp_path):
    path = tmp_path / "samples.json"
    path.write_text(json.dumps({"samples": [record()]}))
    prepared = prepare([path])
    assert prepared[0]["hand"] == "left"
    assert prepared[0]["session"] == "visit1"
    assert prepared[0]["lighting"] == "soft"
    path.write_text(json.dumps({"samples": [record(), record(letter="Q")]}))
    with pytest.raises(ValueError, match="conflicting labels"):
        prepare([path])
    assert prepare([path], drop_conflicts=True) == []


def test_duplicates_and_unknown_metadata_do_not_invent_diversity():
    clean, report = curate([record(participant=None, hand="unknown", lighting="unknown", angle="unknown"), record(participant=None, hand="unknown", lighting="unknown", angle="unknown")])
    assert len(clean) == 1
    assert report["duplicate_rows"] == 1
    summary = coverage(clean, kind="static")
    assert summary["participant_count"] == 0
    assert summary["letters"]["L"]["hand"] == {"unknown": 1}
    assert not summary["collection_targets_met"]
    assert "missing hand: left, right" in summary["letters"]["L"]["needs"]


def test_cross_participant_duplicates_are_quarantined_and_malformed_metadata_is_invalid():
    clean, report = curate([record(participant="one"), record(participant="two"), record(hand=[])])
    assert clean == []
    assert len(report["cross_participant_duplicates"]) == 1
    assert len(report["invalid"]) == 1


def test_public_collection_is_balanced_real_data_with_disjoint_signers():
    rows = json.loads((ROOT / "datasets/aslyset-balanced.json").read_text())
    assert len(rows) == 3840
    assert len({r["fingerprint"] for r in rows}) == 3840
    assert len({r["source_sha256"] for r in rows}) == 3840
    cells = Counter((r["participant"], r["hand"], r["letter"]) for r in rows)
    assert len(cells) == 4*24
    assert set(cells.values()) == {40}
    assert all(r["hand"] == "right" and r["lighting"] == "unknown" and r["angle"] == "unknown" for r in rows)
    y = np.array([r["letter"] for r in rows])
    train, val, test, name = split_indices(rows, y)
    assert name == "participant-disjoint"
    people = [{rows[i]["participant"] for i in indices} for indices in [train, val, test]]
    assert not people[0] & people[1] and not people[0] & people[2] and not people[1] & people[2]


def test_balance_refuses_to_repeat_examples_to_fill_missing_hands():
    with pytest.raises(ValueError, match="fewer"):
        balance([record()], 2, ["left", "right"], ["L"])
