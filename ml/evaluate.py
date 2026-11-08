"""Independently reproduce exported neural-network predictions on held-out examples."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report

ROOT = Path(__file__).resolve().parents[1] / "signspeak" / "static" / "models"


def evaluate(prefix=""):
    model = json.loads((ROOT / f"{prefix}classifier.json").read_text())
    report = json.loads((ROOT / f"{prefix}evaluation.json").read_text())
    examples = report["test_examples"]
    x = (np.array([r["features"] for r in examples]) - model["mean"]) / model["scale"]
    for i, layer in enumerate(model["layers"]):
        x = x @ layer["kernel"] + layer["bias"]
        if i < len(model["layers"]) - 1:
            x = np.maximum(0, x)
    exp = np.exp(x - x.max(axis=1, keepdims=True))
    probability = exp / exp.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probability, [r["probabilities"] for r in examples], atol=1e-6)
    prediction = np.array(model["labels"])[probability.argmax(axis=1)]
    print(classification_report([r["expected"] for r in examples], prediction, zero_division=0))
    print("Exported weights match the trained classifier on every held-out sample.")


def main():
    evaluate()
    if (ROOT / "motion-classifier.json").exists():
        evaluate("motion-")


if __name__ == "__main__":
    main()
