import json
from pathlib import Path
import numpy as np
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
