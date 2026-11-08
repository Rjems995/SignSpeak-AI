import json
from pathlib import Path
import numpy as np
import pytest
from ml.features import normalize

ROOT = Path(__file__).resolve().parents[1] / "signspeak" / "static" / "models"


def test_feature_normalization_invariant_to_position_scale_mirror():
    refs = json.loads((ROOT / "references.json").read_text())
    points = np.array(refs["L"]).reshape(21, 2)
    expected = normalize(points)
    np.testing.assert_allclose(normalize(points * 2 + [0.2, -0.7]), expected, atol=1e-10)
    np.testing.assert_allclose(normalize(points * [-1, 1]), expected, atol=1e-10)


def test_no_duplicate_samples_cross_splits_and_export_matches():
    report = json.loads((ROOT / "evaluation.json").read_text())
    splits = report["split_fingerprints"]
    assert not set(splits["train"]) & set(splits["test"])
    assert not set(splits["train"]) & set(splits["validation"])
    assert not set(splits["validation"]) & set(splits["test"])
    from ml.evaluate import main
    main()


def test_all_static_classes_and_orientation_are_preserved():
    from ml.features import LABELS
    assert set(LABELS) == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ") - {"J", "Z"}
    model = json.loads((ROOT / "classifier.json").read_text())
    assert set(model["labels"]) == set(LABELS)
    refs = json.loads((ROOT / "references.json").read_text())
    points = np.array(refs["K"]).reshape(21, 2)
    assert not np.allclose(normalize(points), normalize(-points))


def test_motion_features_and_exported_probabilities():
    from ml.motion_features import motion_features
    model = json.loads((ROOT / "motion-classifier.json").read_text())
    report = json.loads((ROOT / "motion-evaluation.json").read_text())
    clips = report["split_clips"]
    assert not set(clips["train"]) & set(clips["test"])
    assert not set(clips["validation"]) & set(clips["test"])
    assert not set(clips["train"]) & set(clips["validation"])
    for example in report["test_examples"]:
        features = motion_features(example["frames"])
        np.testing.assert_allclose(features, example["features"], atol=1e-10)
        x = (features-np.array(model["mean"]))/np.array(model["scale"])
        for i, layer in enumerate(model["layers"]):
            x = x @ layer["kernel"] + layer["bias"]
            if i < len(model["layers"])-1:
                x = np.maximum(x, 0)
        e = np.exp(x-x.max())
        np.testing.assert_allclose(e/e.sum(), example["probabilities"], atol=1e-6)
    still = [{"t": i*100, "points": report["test_examples"][0]["frames"][0]["points"]} for i in range(20)]
    with pytest.raises(ValueError, match="stationary"):
        motion_features(still)
