import json
from copy import deepcopy
from pathlib import Path
import numpy as np
import pytest
from ml.data_quality import curate
from ml.motion_audit import motion_observations
from ml.motion_features import motion_features
from ml.train_motion import split_records

ROOT = Path(__file__).resolve().parents[1]
BASE = json.loads((ROOT / "datasets/motion.json").read_text())["samples"]


def test_real_non_jz_sequences_can_be_controls_but_conflicting_labels_are_quarantined():
    clip = deepcopy(BASE[0])
    clip["letter"] = "_"  # Unit fixture only, never added to a training file.
    clean, audit = curate([clip])
    assert len(clean) == 1 and not audit["invalid"]
    assert "frames" in clean[0]
    clean, audit = curate([BASE[0], clip])
    assert clean == [] and len(audit["conflicts"]) == 1


def test_frozen_baseline_clips_keep_their_original_partitions_when_data_is_added():
    frozen = json.loads((ROOT / "datasets/baselines/motion-v1/motion-evaluation.json").read_text())
    records = [{"legacy_clip": clip, "clip": "signn/"+clip, "letter": clip[0], "source": "SigNN"} for clips in frozen["split_clips"].values() for clip in clips]
    records.extend({"clip": f"new/{label}/{i}", "letter": label, "source": "new", "participant": None} for label in ["J", "Z", "_"] for i in range(10))
    y = np.array([r["letter"] for r in records])
    train, val, test, description = split_records(records, y, frozen)
    assert "frozen" in description
    assert not set(train) & set(test) and not set(train) & set(val) and not set(val) & set(test)
    for key, indices in zip(["train", "validation", "test"], [train, val, test]):
        assert {records[i]["legacy_clip"] for i in indices if "legacy_clip" in records[i]} == set(frozen["split_clips"][key])
    with pytest.raises(ValueError, match="incomplete"):
        split_records(records[1:], y[1:], frozen)


def test_duration_and_speed_are_measured_separately_from_path_and_do_not_create_samples():
    original = BASE[0]["frames"]
    slower = [{**f, "t": f["t"]*2} for f in original]
    a, b = motion_observations(original), motion_observations(slower)
    assert b["active_duration_ms"] == a["active_duration_ms"]*2
    assert b["trajectory_extent_palm_lengths"] == a["trajectory_extent_palm_lengths"]
    assert b["travel_speed_palm_lengths_per_second"] == pytest.approx(a["travel_speed_palm_lengths_per_second"]/2)
    np.testing.assert_allclose(motion_features(original), motion_features(slower), atol=1e-10)
    assert "derived" in a["measurement_source"]


def test_complete_fast_motion_is_allowed_but_extremely_short_captures_are_rejected():
    original = BASE[0]["frames"]
    def timed(duration):
        return [{**frame, "t": (frame["t"]-original[0]["t"])/(original[-1]["t"]-original[0]["t"])*duration} for frame in original]
    np.testing.assert_allclose(motion_features(original), motion_features(timed(300)), atol=1e-10)
    with pytest.raises(ValueError, match="throughout"):
        motion_features(timed(150))
