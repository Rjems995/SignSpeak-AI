"""Compare exported models on exactly the candidate's frozen unseen-signer test."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .data_quality import ROOT


def predict(model, features):
    x = (features - np.array(model["mean"])) / np.array(model["scale"])
    for i, layer in enumerate(model["layers"]):
        x = x @ np.array(layer["kernel"]) + np.array(layer["bias"])
        if i < len(model["layers"])-1:
            x = np.maximum(0, x)
    exp = np.exp(x-x.max(axis=1, keepdims=True))
    probabilities = exp/exp.sum(axis=1, keepdims=True)
    prediction = np.array(model["labels"])[probabilities.argmax(axis=1)]
    gated = prediction.copy()
    for label in set(prediction)-{"_"}:
        indices = np.where(prediction == label)[0]
        support = np.array(model["support"][label])
        distances = np.sqrt(((features[indices, None, :]-support[None, :, :])**2).mean(axis=2)).min(axis=1)
        rejected = (probabilities[indices].max(axis=1) < .85) | (distances > model["distance_thresholds"][label])
        gated[indices[rejected]] = "_"
    return prediction, gated


def metrics(rows, raw, gated):
    truth = np.array([r["letter"] for r in rows])
    def score(indices):
        ids = np.array(indices, dtype=int)
        accepted = gated[ids] != "_"
        return {"samples": len(ids), "raw_accuracy": float((raw[ids] == truth[ids]).mean()),
                "gated_accuracy": float((gated[ids] == truth[ids]).mean()),
                "acceptance_rate": float(accepted.mean()),
                "accuracy_when_accepted": float((gated[ids][accepted] == truth[ids][accepted]).mean()) if accepted.any() else None}
    result = score(range(len(rows)))
    for key in ["source", "participant", "letter", "hand"]:
        result["by_"+key] = {value: score([i for i, row in enumerate(rows) if row.get(key, "unknown") == value]) for value in sorted({r.get(key, "unknown") for r in rows})}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, default=ROOT / "artifacts/public-candidate")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/public-balanced.json")
    args = parser.parse_args()
    dataset_bytes = args.dataset.read_bytes()
    rows = json.loads(dataset_bytes)
    evaluation = json.loads((args.candidate / "evaluation.json").read_text())
    if hashlib.sha256(dataset_bytes).hexdigest() != evaluation["dataset_sha256"]:
        raise ValueError("Candidate was trained on a different dataset")
    frozen = evaluation["split_fingerprints"]
    trainval = set(frozen["train"]) | set(frozen["validation"])
    if trainval & set(frozen["test"]):
        raise ValueError("Candidate test overlaps training/validation")
    by_fingerprint = {r["fingerprint"]: r for r in rows}
    test = [by_fingerprint[value] for value in frozen["test"]]
    training_people = {r["participant"] for r in rows if r["fingerprint"] in trainval}
    if training_people & {r["participant"] for r in test}:
        raise ValueError("Signer leakage in test partition")
    x = np.array([r["features"] for r in test])
    result = {"dataset_sha256": evaluation["dataset_sha256"], "test_fingerprints_sha256": hashlib.sha256("\n".join(frozen["test"]).encode()).hexdigest(),
              "test_samples": len(test), "test_participants": sorted({r["participant"] for r in test}),
              "method": "Same frozen participant-disjoint test for both exported models. No test-based parameter or threshold tuning.",
              "limitations": ["No verified left-hand test cohort", "Per-image lighting and camera angles unknown", "No unsupported-sign false-acceptance benchmark", "Public still images do not establish live webcam performance", "J/Z motion model not changed"]}
    for name, directory in [("published_baseline", ROOT / "signspeak/static/models"), ("public_candidate", args.candidate)]:
        model_bytes = (directory / "classifier.json").read_bytes()
        model = json.loads(model_bytes)
        prediction, gated = predict(model, x)
        result[name] = {"model_sha256": hashlib.sha256(model_bytes).hexdigest(), **metrics(test, prediction, gated)}
        if name == "public_candidate":
            np.testing.assert_allclose(result[name]["raw_accuracy"], evaluation["raw_classifier_accuracy"])
            np.testing.assert_allclose(result[name]["gated_accuracy"], evaluation["gated_accuracy_at_threshold_085"])
    output = ROOT / "datasets/public-candidate-evaluation.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({name: {k: v for k, v in result[name].items() if not k.startswith("by_")} for name in ["published_baseline", "public_candidate"]}, indent=2))


if __name__ == "__main__":
    main()
