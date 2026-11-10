"""Train a genuine MLP, export weights for TF.js, and save an untouched-test report."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .features import FEATURE_VERSION, LABELS
from .data_quality import coverage, UNKNOWN

ROOT = Path(__file__).resolve().parents[1]


def split_indices(samples, y):
    ids = np.arange(len(y))
    groups = [r.get("participant") for r in samples]
    if all(value not in UNKNOWN for value in groups) and len(set(groups)) >= 3:
        # Stratify source cohorts when each has enough identified participants.
        # This stops a large source from accidentally monopolizing the holdout.
        sources = np.array([r.get("source", "unrecorded provenance") for r in samples])
        participant_sources = {person: set(sources[np.array(groups) == person]) for person in set(groups)}
        if any(len(values) != 1 for values in participant_sources.values()):
            raise ValueError("A participant ID occurs in multiple source namespaces; reconcile IDs before splitting")
        if len(set(sources)) > 1 and all(len(set(np.array(groups)[sources == source])) >= 3 for source in set(sources)):
            partitions = [[], [], []]
            for source in sorted(set(sources)):
                source_ids = ids[sources == source]
                rows = [samples[i] for i in source_ids]
                train, val, test, _ = split_indices(rows, y[source_ids])
                for target, selection in zip(partitions, [train, val, test]):
                    target.extend(source_ids[selection])
            return *(np.array(values) for values in partitions), "participant-disjoint"
        trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(ids, y, groups))
        train, val = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=43).split(trainval, y[trainval], np.array(groups)[trainval]))
        return trainval[train], trainval[val], test, "participant-disjoint"
    # Allocate per class: sklearn's global rounding can omit a class with only
    # three examples, even though one example per split is possible.
    rng = np.random.default_rng(42)
    train, val, test = [], [], []
    for label in sorted(set(y)):
        indices = rng.permutation(ids[y == label])
        if len(indices) < 2:
            raise ValueError(f"{label} needs at least two unique samples for separate training/test data")
        ntest = max(1, round(len(indices)*.2))
        nval = 0 if len(indices) == 2 else max(1, round(len(indices)*.2))
        test.extend(indices[:ntest])
        val.extend(indices[ntest:ntest+nval])
        train.extend(indices[ntest+nval:])
    return np.array(train), np.array(val), np.array(test), "deduplicated per-class sample split; participant IDs unavailable or incomplete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets" / "prepared.json")
    parser.add_argument("--output", type=Path, default=ROOT / "signspeak" / "static" / "models")
    parser.add_argument("--require-participant-split", action="store_true")
    args = parser.parse_args()
    raw = args.dataset.read_bytes()
    samples = json.loads(raw)
    if any(row.get("training_eligible") is False for row in samples):
        raise ValueError("Benchmark/review-only records cannot enter training; preserve the independent evaluation source")
    x = np.array([r["features"] for r in samples])
    y = np.array([r["letter"] if r["letter"] in LABELS else "_" for r in samples])
    required = set(LABELS)
    if not required.issubset(set(y)):
        raise ValueError("Collect training data for all 24 static ASL letters before training.")
    train, val, test, split_name = split_indices(samples, y)
    if args.require_participant_split and split_name != "participant-disjoint":
        raise ValueError("A participant-disjoint split was required. Supply identified samples from at least three signers; do not mix unidentified legacy rows into this evaluation.")
    for indices in [train, test]:
        if set(y[indices]) != set(y):
            raise ValueError("Each split must contain every training class. Collect more participants/samples.")
    scaler = StandardScaler().fit(x[train])
    # Augmentation only touches training samples, after the split.
    rng = np.random.default_rng(17)
    train_x = np.concatenate([x[train]] + [x[train] + rng.normal(0, 0.015, x[train].shape) for _ in range(5)])
    train_y = np.tile(y[train], 6)
    classifier = MLPClassifier(hidden_layer_sizes=(96, 64), solver="lbfgs", alpha=0.1, max_iter=2000, random_state=17)
    classifier.fit(scaler.transform(train_x), train_y)
    labels = classifier.classes_.tolist()
    # Class-specific support gate. Tune only with validation distances, never test data.
    support = {label: x[train][y[train] == label].tolist() for label in labels if label != "_"}
    def distance(features, label):
        return np.sqrt(((features[:, None, :] - np.array(support[label])[None, :, :]) ** 2).mean(axis=2)).min(axis=1)
    gates = {label: float(max(0.12, np.quantile(distance(x[val][y[val] == label], label), 0.95))) if (y[val] == label).any() else 0.12 for label in support}
    probability = classifier.predict_proba(scaler.transform(x[test]))
    raw_prediction = np.array(labels)[probability.argmax(axis=1)]
    accepted = raw_prediction.copy()
    for i, label in enumerate(raw_prediction):
        if label != "_" and (probability[i].max() < 0.85 or distance(x[test][i:i+1], label)[0] > gates[label]):
            accepted[i] = "_"
    report = {
        "dataset_sha256": hashlib.sha256(raw).hexdigest(), "seed": {"test": 42, "validation": 43, "model": 17},
        "split": split_name, "feature_version": FEATURE_VERSION,
        "counts": {"unique_total": len(samples), "train": len(train), "validation": len(val), "test": len(test)},
        "dataset_coverage": coverage(samples, kind="static"),
        "sources": sorted({r.get("source", "unrecorded provenance") for r in samples}),
        "participant_counts": {key: len({samples[i].get("participant") for i in indices if samples[i].get("participant")}) for key, indices in [("train", train), ("validation", val), ("test", test)]},
        "participant_ids": {key: sorted({samples[i]["participant"] for i in indices if samples[i].get("participant")}) for key, indices in [("train", train), ("validation", val), ("test", test)]},
        "source_counts": {key: {source: sum(samples[i].get("source", "unrecorded provenance") == source for i in indices) for source in sorted({r.get("source", "unrecorded provenance") for r in samples})} for key, indices in [("train", train), ("validation", val), ("test", test)]},
        "classes": labels, "class_counts": {label: int((y == label).sum()) for label in labels},
        "raw_classifier_accuracy": float(accuracy_score(y[test], raw_prediction)),
        "supported_letter_accuracy": float(accuracy_score(y[test][y[test] != "_"], raw_prediction[y[test] != "_"])),
        "supported_letter_test_samples": int((y[test] != "_").sum()),
        "rejection_evaluated": "_" in y[test],
        "validation_missing_classes": [label for label in labels if label not in y[val]],
        "low_support_warning": ("No validation examples for " + ", ".join(label for label in labels if label not in y[val]) + ". These classes use a fixed conservative distance gate.") if any(label not in y[val] for label in labels) else "",
        "gated_accuracy_at_threshold_085": float(accuracy_score(y[test], accepted)),
        "classification_report": classification_report(y[test], raw_prediction, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y[test], raw_prediction, labels=labels).tolist(),
        "gated_labels": labels if "_" in labels else labels+["_"],
        "gated_confusion_matrix": confusion_matrix(y[test], accepted, labels=labels if "_" in labels else labels+["_"]).tolist(),
        "warning": ("Experimental participant-disjoint evaluation on this dataset only. Coverage of both hands, lighting and camera conditions must be inspected separately. Unknown signs can still be misclassified." if split_name == "participant-disjoint" else "Small experimental dataset. Sample holdout does not establish accuracy on unseen signers. Unknown signs can still be misclassified. Validate with consented, participant-disjoint webcam data before real-world reliance."),
        "test_examples": [{"features": x[i].tolist(), "expected": str(y[i]), "probabilities": probability[j].tolist()} for j, i in enumerate(test)],
        "split_fingerprints": {key: [samples[i]["fingerprint"] for i in indices] for key, indices in [("train", train), ("validation", val), ("test", test)]},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        (args.output / name).write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    save("classifier.json", {"feature_version": FEATURE_VERSION, "labels": labels, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
                             "layers": [{"kernel": kernel.tolist(), "bias": bias.tolist()} for kernel, bias in zip(classifier.coefs_, classifier.intercepts_)], "support": support, "distance_thresholds": gates})
    save("evaluation.json", report)
    save("metadata.json", {"ready": True, "labels": [x for x in labels if x != "_"], "feature_version": FEATURE_VERSION,
                           "training_samples": len(train), "test_samples": len(test), "test_accuracy": report["raw_classifier_accuracy"],
                           "supported_accuracy": report["supported_letter_accuracy"], "supported_test_samples": report["supported_letter_test_samples"],
                           "low_support_warning": report["low_support_warning"],
                           "split": split_name, "experimental": True, "source": ", ".join(report["sources"]), "warning": report["warning"]})
    references = {}
    for label in support:
        # The central training sample is a recorded reference, never a fake prediction.
        candidates = [samples[i] for i in train if y[i] == label]
        center = np.median([r["features"] for r in candidates], axis=0)
        references[label] = min(candidates, key=lambda r: np.linalg.norm(np.array(r["features"]) - center))["points"]
    save("references.json", references)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(12, 11))
    ConfusionMatrixDisplay(np.array(report["confusion_matrix"]), display_labels=labels).plot(ax=ax, colorbar=False, cmap="Purples")
    ax.set_title("Participant-disjoint holdout" if split_name == "participant-disjoint" else "Deduplicated sample holdout; signer IDs incomplete")
    fig.tight_layout()
    fig.savefig(args.output / "confusion-matrix.png", dpi=140)
    plt.close(fig)
    print(json.dumps({k: report[k] for k in ["counts", "split", "raw_classifier_accuracy", "gated_accuracy_at_threshold_085"]}, indent=2))


if __name__ == "__main__":
    main()

