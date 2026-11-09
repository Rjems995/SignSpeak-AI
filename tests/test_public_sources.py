import numpy as np
import pytest
from ml.import_public import source_metadata, select_members
from ml.train import split_indices
from ml.data_quality import coverage


def test_source_metadata_does_not_turn_light_direction_or_detector_estimates_into_hand_or_angle():
    row = source_metadata("massey", "ASL-Finger-Spelling-Recognition-"+"a"*40+"/asl_dataset/q/hand2_q_left_seg_1_cropped.jpeg")
    assert row["participant"] == "massey_hand2"
    assert row["lighting_direction"] == "left"
    assert row["hand"] == row["angle"] == row["lighting"] == "unknown"
    assert row["training_eligible"] is False
    row = source_metadata("signalphaset", "SignAlphaSet/Q/Q_750.jpg")
    assert row["participant"] is None and row["hand"] == "unknown"
    assert row["training_eligible"] is False
    row = source_metadata("asl_hg", "asl_dataset/Q/P10_Q_999.jpg")
    assert row["participant"] == "asl_hg_p10" and row["training_eligible"]
    assert source_metadata("asl_hg", "asl_dataset/J/P1_J_1.jpg") is None
    assert source_metadata("asl_hg", "../asl_dataset/Q/P1_Q_1.jpg") is None
    assert source_metadata("asl_hg", "asl_dataset/Q/P1_P_1.jpg") is None
    summary = coverage([dict(row, observed_hand="left", observed_hand_score=.99)], kind="static")
    assert summary["letters"]["Q"]["hand"] == {"unknown": 1}


def test_public_selection_is_order_independent_and_never_creates_examples():
    names = [f"asl_dataset/Q/P1_Q_{i}.jpg" for i in range(50)]
    selected = select_members("asl_hg", names, per_cell=5, priority_extra=3)
    assert selected == select_members("asl_hg", reversed(names), per_cell=5, priority_extra=3)
    assert len(selected) == len(set(selected)) == 8
    assert set(selected) <= set(names)


def test_source_stratification_preserves_every_cohort_and_disjoint_signers():
    rows = [{"participant": f"{source}_{i}", "source": source} for source, n in [("a", 4), ("b", 10)] for i in range(n) for _ in range(4)]
    y = np.array(["Q"]*len(rows))
    train, val, test, method = split_indices(rows, y)
    assert method == "participant-disjoint"
    groups = [{rows[i]["participant"] for i in part} for part in [train, val, test]]
    assert not (groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
    for part in [train, val, test]:
        assert {rows[i]["source"] for i in part} == {"a", "b"}
    assert set(np.concatenate([train, val, test])) == set(range(len(rows)))


def test_source_balance_refuses_missing_letters_instead_of_filling_a_quota():
    import json
    from pathlib import Path
    from ml.build_public_candidate import select_balanced
    points = json.loads((Path(__file__).resolve().parents[1] / "signspeak/static/models/references.json").read_text())["L"]
    with pytest.raises(ValueError, match="Every signer"):
        select_balanced([{"letter": "L", "points": points, "participant": "one"}])


def test_bundled_candidate_is_balanced_original_data_and_unknown_hands_stay_unknown():
    import json
    from collections import Counter
    from ml.data_quality import ROOT
    rows = json.loads((ROOT / "datasets/public-balanced.json").read_text())
    assert len({r["fingerprint"] for r in rows}) == len(rows)
    assert len({r["source_sha256"] for r in rows}) == len(rows)
    assert {r["source"] for r in rows} == {"ASLYset", "ASL-HG"}
    cells = Counter((r["participant"], r["letter"]) for r in rows)
    assert len(cells) == 14*24 and set(cells.values()) == {17}
    assert all(r["participant"] == source_metadata("asl_hg", r["source_file"])["participant"] and r["hand"] == "unknown" for r in rows if r["source"] == "ASL-HG")
    assert all(r["hand"] == "right" for r in rows if r["source"] == "ASLYset")
    assert all(r["lighting"] == r["angle"] == "unknown" for r in rows)
    summary = coverage(rows, kind="static")
    assert not summary["collection_targets_met"]
    assert summary["letters"]["Q"]["hand"] == {"right": 68, "unknown": 170}
