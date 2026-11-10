"""Train J/Z (+ optional real non-J/Z controls); split originals before augmentation."""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from .data_quality import ROOT, UNKNOWN, curate
from .motion_features import MOTION_FEATURE_VERSION
from .motion_audit import ranges


def split_records(records, y, frozen=None):
    ids = np.arange(len(records))
    people = [r.get("participant") for r in records]
    if frozen is not None:
        partitions = {key: [] for key in ["train", "validation", "test"]}
        old = {clip: key for key, values in frozen["split_clips"].items() for clip in values}
        seen, extras = set(), defaultdict(list)
        for i, row in enumerate(records):
            legacy = row.get("legacy_clip")
            if legacy in old:
                if legacy in seen:
                    raise ValueError("Repeated frozen baseline clip")
                seen.add(legacy)
                partitions[old[legacy]].append(i)
            else:
                extras[(row.get("source", "unknown"), row["letter"])].append(i)
        if seen != set(old):
            raise ValueError("The frozen original partition is incomplete")
        rng = np.random.default_rng(42)
        for group in sorted(extras):
            rows = sorted(extras[group], key=lambda i: records[i]["clip"])
            ordered = rng.permutation(rows)
            if len(rows) < 3:
                raise ValueError("Each added source/class needs at least three clips for separate splits")
            ntest = max(1, round(len(rows)*.2))
            nval = max(1, round(len(rows)*.2))
            partitions["test"].extend(ordered[:ntest])
            partitions["validation"].extend(ordered[ntest:ntest+nval])
            partitions["train"].extend(ordered[ntest+nval:])
        train, val, test = (np.array(partitions[key]) for key in ["train", "validation", "test"])
        return train, val, test, "source-clip-disjoint; original baseline partitions frozen; participant IDs unavailable"
    if all(person not in UNKNOWN for person in people) and len(set(people)) >= 3:
        trainval, test = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42).split(ids, y, people))
        ti, vi = next(GroupShuffleSplit(n_splits=1, test_size=.25, random_state=43).split(trainval, y[trainval], np.array(people)[trainval]))
        return trainval[ti], trainval[vi], test, "participant-disjoint"
    trainval, test = train_test_split(ids, test_size=.2, random_state=42, stratify=y)
    train, val = train_test_split(trainval, test_size=.25, random_state=43, stratify=y[trainval])
    return train, val, test, "source-clip-disjoint; participant IDs unavailable or incomplete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/motion.json")
    parser.add_argument("--output", type=Path, default=ROOT / "signspeak/static/models")
    parser.add_argument("--require-participant-split", action="store_true")
    parser.add_argument("--freeze-baseline-splits", type=Path)
    args = parser.parse_args()
    dataset_bytes = args.dataset.read_bytes()
    records, audit = curate(json.loads(dataset_bytes)["samples"])
    if audit["invalid"] or audit["conflicts"]:
        raise ValueError("Audit invalid or conflicting movements before training; labels are not first-record-wins")
    if any(r.get("training_eligible") is False for r in records):
        raise ValueError("Research-only or otherwise ineligible records cannot enter training")
    if any("frames" not in r for r in records):
        raise ValueError("Motion training requires original sequences")
    x = np.array([r["features"] for r in records])
    y = np.array([r["letter"] for r in records])
    if not {"J", "Z"}.issubset(set(y)) or not set(y).issubset({"J", "Z", "_"}) or min((y == label).sum() for label in set(y)) < 5:
        raise ValueError("At least five unique complete clips per training class are required (J/Z, optionally _ controls)")
    frozen = json.loads(args.freeze_baseline_splits.read_text()) if args.freeze_baseline_splits else None
    train, val, test, split = split_records(records, y, frozen)
    if args.require_participant_split and split != "participant-disjoint":
        raise ValueError("A participant-disjoint split was required. Supply complete participant IDs from at least three signers.")
    if any(set(y[indices]) != set(y) for indices in [train, val, test]):
        raise ValueError("Every split needs every training class. Collect more real clips.")
    video_splits = [{records[i]["source_sha256"] for i in indices if records[i].get("source_sha256")} for indices in [train, val, test]]
    if video_splits[0] & video_splits[1] or video_splits[0] & video_splits[2] or video_splits[1] & video_splits[2]:
        raise ValueError("The same source video appears across splits")
    scaler = StandardScaler().fit(x[train])
    rng = np.random.default_rng(17)
    augmented = np.concatenate([x[train]]+[x[train]+rng.normal(0, .01, x[train].shape) for _ in range(4)])
    classifier = MLPClassifier(hidden_layer_sizes=(48, 24), solver="lbfgs", alpha=.2, random_state=17, max_iter=1500)
    classifier.fit(scaler.transform(augmented), np.tile(y[train], 5))
    probability = classifier.predict_proba(scaler.transform(x[test]))
    labels = classifier.classes_.tolist()
    prediction = np.array(labels)[probability.argmax(axis=1)]
    support = {label: x[train][y[train] == label].tolist() for label in labels if label != "_"}
    def distance(features, label):
        return np.sqrt(((features[:, None, :]-np.array(support[label])[None, :, :])**2).mean(axis=2)).min(axis=1)
    gates = {label: float(max(.25, np.quantile(distance(x[val][y[val] == label], label), .95))) for label in support}
    accepted = [str(label) if label != "_" and probability[i].max() >= .85 and distance(x[test][i:i+1], label)[0] <= gates[label] else "_" for i, label in enumerate(prediction)]
    kernels = [layer.copy() for layer in classifier.coefs_]
    biases = [layer.copy() for layer in classifier.intercepts_]
    if len(labels) == 2:
        kernels[-1] = np.column_stack([np.zeros(kernels[-1].shape[0]), kernels[-1][:, 0]])
        biases[-1] = np.array([0, biases[-1][0]])
    sources = sorted({r.get("source", "unrecorded provenance") for r in records})
    warning = "Experimental public-video motion model. Per-clip signer IDs and capture conditions are incomplete; clip-disjoint scores do not establish unseen-signer or live-webcam accuracy."
    if "_" not in labels:
        warning += " No held-out unsupported movements; false acceptance remains unmeasured."
    partition_items = [("train", train), ("validation", val), ("test", test)]
    report = {"dataset_sha256": hashlib.sha256(dataset_bytes).hexdigest(), "split": split, "feature_version": MOTION_FEATURE_VERSION,
              "seed": {"split": 42, "validation": 43, "model": 17}, "classes": labels,
              "counts": {"unique_total": len(records), "train": len(train), "validation": len(val), "test": len(test)},
              "class_counts": dict(Counter(y)), "sources": sources, "curation": audit,
              "raw_classifier_accuracy": float(accuracy_score(y[test], prediction)),
              "supported_letter_accuracy": float(accuracy_score(y[test][y[test] != "_"], prediction[y[test] != "_"])),
              "supported_letter_test_samples": int((y[test] != "_").sum()),
              "gated_accuracy_at_threshold_085": float(accuracy_score(y[test], accepted)),
              "rejection_evaluated": "_" in y[test],
              "classification_report": classification_report(y[test], prediction, output_dict=True, zero_division=0),
              "confusion_matrix": confusion_matrix(y[test], prediction, labels=labels).tolist(),
              "gated_labels": labels if "_" in labels else labels+["_"],
              "gated_confusion_matrix": confusion_matrix(y[test], accepted, labels=labels if "_" in labels else labels+["_"]).tolist(),
              "test_examples": [{"frames": records[i]["frames"], "features": x[i].tolist(), "expected": str(y[i]), "probabilities": probability[j].tolist()} for j, i in enumerate(test)],
              "split_clips": {key: [records[i].get("clip", records[i]["fingerprint"]) for i in indices] for key, indices in partition_items},
              "split_fingerprints": {key: [records[i]["fingerprint"] for i in indices] for key, indices in partition_items},
              "split_source_hashes": {key: sorted(video_splits[index]) for index, (key, _) in enumerate(partition_items)},
              "source_counts": {key: dict(Counter(records[i].get("source", "unrecorded provenance") for i in indices)) for key, indices in partition_items},
              "observed_ranges": {key: ranges([records[i] for i in indices if "observations" in records[i]]) for key, indices in partition_items}, "warning": warning}
    args.output.mkdir(parents=True, exist_ok=True)
    def save(name, data):
        (args.output/name).write_text(json.dumps(data, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    save("motion-classifier.json", {"feature_version": MOTION_FEATURE_VERSION, "labels": labels, "mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist(),
         "layers": [{"kernel": k.tolist(), "bias": b.tolist()} for k, b in zip(kernels, biases)], "support": support, "distance_thresholds": gates})
    save("motion-evaluation.json", report)
    save("motion-metadata.json", {"ready": True, "labels": [label for label in labels if label != "_"], "feature_version": MOTION_FEATURE_VERSION,
         "training_samples": len(train), "test_samples": report["supported_letter_test_samples"], "test_accuracy": report["supported_letter_accuracy"], "all_test_samples": len(test), "split": split,
         "experimental": True, "source": ", ".join(sources), "recording_ms": 4500, "negative_training_samples": int((y[train] == "_").sum()), "negative_test_samples": int((y[test] == "_").sum()), "warning": warning})
    references = {}
    for label in support:
        candidates = [i for i in train if y[i] == label]
        center = np.median(x[candidates], axis=0)
        best = min(candidates, key=lambda i: np.linalg.norm(x[i]-center))
        references[label] = records[best]["frames"]
    save("motion-references.json", references)
    print(json.dumps({k: report[k] for k in ["counts", "class_counts", "split", "raw_classifier_accuracy", "gated_accuracy_at_threshold_085"]}, indent=2))


if __name__ == "__main__":
    main()
