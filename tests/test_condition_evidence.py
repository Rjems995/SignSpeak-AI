import json
from pathlib import Path
import pytest
from ml.condition_evidence import summarize_conditions

ROOT = Path(__file__).resolve().parents[1]


def annotated(**updates):
    row = {"hand": "right", "lighting_direction": "diffuse", "camera_view": "frontal", "performance_id": "fixture_01", "metadata_evidence": {}}
    row.update(updates)
    row["metadata_evidence"] = {field: {"value": row[field], "basis": "publisher_per_sample", "source": "https://example.org/fixture-documentation", "locator": "test fixture annotation row"} for field in ["hand", "lighting_direction", "camera_view"]}
    row["metadata_evidence"]["hand"]["assertion"] = "physical_hand_used"
    return row


def test_detector_handedness_and_normalized_brightness_do_not_fill_capture_annotations():
    summary = summarize_conditions([{"observed_hand": "left", "observed_hand_score": .999, "lighting_processing": "brightness_normalized"}])
    assert summary["documented_physical_hands"] == {"left": 0, "right": 0, "unknown": 1}
    assert summary["hand_lighting_view_together"] == 0
    assert summary["documented_lighting_directions"] == {}
    assert summary["independent_performances"] == 0
    row = annotated(hand="left")
    row["metadata_evidence"]["hand"]["basis"] = "mediapipe_estimate"
    assert summarize_conditions([row])["documented_physical_hands"]["left"] == 0


def test_camera_protocol_is_not_a_per_image_view_and_light_side_is_not_physical_hand():
    row = {"hand": "unknown", "lighting_direction": "left", "camera_setup": "fixed_tripod_perpendicular_to_subject", "metadata_evidence": {}}
    for field in ["lighting_direction", "camera_setup"]:
        row["metadata_evidence"][field] = {"value": row[field], "basis": "publisher_protocol", "source": "https://example.org/fixture", "locator": "capture protocol"}
    summary = summarize_conditions([row])
    assert summary["documented_lighting_directions"] == {"left": 1}
    assert summary["documented_physical_hands"]["left"] == 0
    assert summary["documented_camera_views"] == {}
    assert summary["hand_lighting_view_together"] == 0


def test_evidence_must_match_value_and_have_source_and_locator():
    row = annotated()
    row["hand"] = "left"
    assert summarize_conditions([row])["documented_physical_hands"]["left"] == 0
    row = annotated()
    del row["metadata_evidence"]["hand"]["locator"]
    assert summarize_conditions([row])["documented_physical_hands"]["right"] == 0


def test_repeated_views_and_augmentation_do_not_create_new_performances_or_both_hand_coverage():
    front = annotated()
    profile = annotated(camera_view="profile_right")
    flipped = annotated(hand="left", augmentation_of="fixture_01")
    summary = summarize_conditions([front, profile, flipped])
    assert summary["sample_views"] == 2
    assert summary["independent_performances"] == 1
    assert summary["documented_physical_hands"]["left"] == 0
    assert not summary["both_hands_with_joint_annotations"]


def test_bundled_lighting_benchmark_retains_real_metadata_and_missing_hands():
    rows = json.loads((ROOT / "datasets/massey-landmarks.json").read_text())["samples"]
    summary = summarize_conditions(rows)
    assert len(rows) == len({r["source_sha256"] for r in rows}) == 895
    assert len(summary["documented_lighting_directions"]) == 5
    assert sum(summary["documented_lighting_directions"].values()) == 895
    assert summary["documented_camera_setups"] == {"fixed_tripod_perpendicular_to_subject": 895}
    assert summary["documented_camera_views"] == {}
    assert summary["documented_physical_hands"]["unknown"] == 895
    assert all(r["training_eligible"] is False for r in rows)


def test_training_refuses_benchmark_records_before_fitting(tmp_path, monkeypatch):
    from ml.train import main
    path = tmp_path / "benchmark.json"
    path.write_text(json.dumps([{"training_eligible": False}]))
    monkeypatch.setattr("sys.argv", ["ml.train", "--dataset", str(path), "--output", str(tmp_path / "model")])
    with pytest.raises(ValueError, match="Benchmark/review-only"):
        main()
    assert not (tmp_path / "model").exists()


def test_saved_view_audit_counts_correlated_views_and_does_not_bundle_restricted_coordinates():
    report = json.loads((ROOT / "datasets/condition-benchmark-report.json").read_text())
    views = report["camera_annotations"]
    assert views["sample_views"] == 10 and views["independent_performances"] == 5
    assert views["documented_camera_views"] == {"frontal": 5, "profile_right": 5}
    assert views["documented_physical_hands"] == {"left": 0, "right": 0, "unknown": 10}
    assert views["documented_signer_dominance"] == {"right": 10}
    assert views["documented_lighting_directions"] == {}
    assert report["verified_left_hand_examples_found"] == 0
    assert report["joint_hand_lighting_view_examples"] == 0
    def inspect(value):
        if isinstance(value, dict):
            assert not set(value) & {"frames", "points", "source_consultant", "source_file"}
            for child in value.values(): inspect(child)
        elif isinstance(value, list):
            for child in value: inspect(child)
    inspect(report)


def test_signer_dominance_is_not_a_per_token_physical_hand_annotation():
    row = annotated()
    del row["metadata_evidence"]["hand"]["assertion"]
    row["metadata_evidence"]["hand"]["locator"] = "subject right-hand dominant"
    assert summarize_conditions([row])["documented_physical_hands"]["right"] == 0


def test_explicit_source_annotations_for_both_physical_hands_are_recognized():
    right, left = annotated(), annotated(hand="left", performance_id="fixture_02")
    summary = summarize_conditions([right, left])
    assert summary["hand_lighting_view_together"] == 2
    assert summary["both_hands_with_joint_annotations"]
    assert summary["documented_physical_hands"] == {"left": 1, "right": 1, "unknown": 0}
