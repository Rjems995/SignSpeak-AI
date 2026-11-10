"""Compare motion models on fixed clips and a local-only independent research corpus."""
import hashlib
import json
from collections import Counter
import numpy as np
from .data_quality import ROOT
from .motion_features import motion_features
from .compare_public_models import predict


def predictions(model, records):
    raw, gated = [], []
    for row in records:
        try:
            x = motion_features(row["frames"])[None, :]
            a, b = predict(model, x)
            raw.append(str(a[0]))
            gated.append(str(b[0]))
        except ValueError:
            raw.append("_")
            gated.append("_")
    return np.array(raw), np.array(gated)


def summarize(records, raw, gated):
    truth = np.array([r["letter"] for r in records])
    positive, negative = truth != "_", truth == "_"
    return {"clips": len(records), "class_counts": dict(Counter(truth)),
            "raw_accuracy": float((raw == truth).mean()) if len(truth) else None,
            "gated_accuracy": float((gated == truth).mean()) if len(truth) else None,
            "jz_raw_accuracy": float((raw[positive] == truth[positive]).mean()) if positive.any() else None,
            "jz_gated_accuracy": float((gated[positive] == truth[positive]).mean()) if positive.any() else None,
            "non_jz_false_acceptance_rate": float((gated[negative] != "_").mean()) if negative.any() else None,
            "by_letter": {label: {"clips": int((truth == label).sum()), "raw_recall": float((raw[truth == label] == label).mean()),
                                   "gated_recall": float((gated[truth == label] == label).mean())} for label in sorted(set(truth))}}


def main():
    dataset = json.loads((ROOT / "datasets/motion-expanded.json").read_text())["samples"]
    candidate = ROOT / "artifacts/motion-candidate"
    baseline = ROOT / "datasets/baselines/motion-v1"
    report = json.loads((candidate / "motion-evaluation.json").read_text())
    if hashlib.sha256((ROOT / "datasets/motion-expanded.json").read_bytes()).hexdigest() != report["dataset_sha256"]:
        raise ValueError("Candidate dataset changed")
    by_clip = {r["clip"]: r for r in dataset}
    partitions = report["split_clips"]
    if set(partitions["test"]) & (set(partitions["train"]) | set(partitions["validation"])):
        raise ValueError("Test clip leakage")
    baseline_report = json.loads((baseline / "motion-evaluation.json").read_text())
    baseline_test = set(baseline_report["split_clips"]["test"])
    test = [by_clip[clip] for clip in partitions["test"]]
    train = [by_clip[clip] for clip in partitions["train"]]
    groups = {"expanded_frozen_test": test, "original_ten_test_clips": [r for r in test if r.get("legacy_clip") in baseline_test]}
    if len(groups["original_ten_test_clips"]) != len(baseline_test):
        raise ValueError("The original holdout was not preserved")
    # Quantile boundaries come only from training originals, never test scores.
    edges = {key: np.quantile([r["observations"][key] for r in train if r["letter"] != "_"], [1/3, 2/3]).tolist() for key in ["active_duration_ms", "trajectory_extent_palm_lengths", "travel_speed_palm_lengths_per_second"]}
    for key, bounds in edges.items():
        for index, name in enumerate(["lower_third", "middle_third", "upper_third"]):
            group = [r for r in test if r["letter"] != "_" and int(np.searchsorted(bounds, r["observations"][key], side="right")) == index]
            if group:
                groups[key+"_"+name] = group
    native = json.loads((ROOT / "artifacts/motion-research/asllvd.json").read_text())
    native_records = native["samples"]
    native_hashes = {r["source_sha256"] for r in native_records}
    if native_hashes & {r["source_sha256"] for r in dataset}:
        raise ValueError("Research benchmark overlaps training source videos")
    groups["asllvd_independent_research_only"] = native_records
    result = {"method": "Frozen original split; same expanded test for both models; no test-based fitting or gate tuning. Independent annotated corpus is research evaluation only, excluded from every training/validation input.",
              "dataset_sha256": report["dataset_sha256"], "measurement_bin_edges_from_training": edges,
              "research_benchmark": {"source": "https://www.bu.edu/asllrp/av/dai-asllvd.html", "terms": "Research/education only; no redistribution or commercial use without permission", "clips": len(native_records),
                  "signer_ids": len({r["participant"] for r in native_records}), "jz_signer_ids": len({r["participant"] for r in native_records if r["letter"] != "_"}),
                  "label_counts": dict(Counter(r["source_label"] for r in native_records)), "extraction_failures": len(native["failures"]),
                  "limitation": "Tiny native-signer benchmark. Aggregate metrics only; restricted clips, source annotation subset and coordinates are not bundled."},
              "limitations": ["Per-clip signer IDs unavailable for SigNN and SignAlphaSet, so expanded split is not signer-independent", "Measured speed/extent are detector-derived proxies, not source capture annotations", "Only two held-out training-source non-J/Z controls", "Independent research corpus is small and visually different", "No consented live-webcam evaluation"]}
    for name, directory in [("published_baseline", baseline), ("expanded_candidate", candidate)]:
        model_bytes = (directory / "motion-classifier.json").read_bytes()
        model = json.loads(model_bytes)
        scores = {}
        for group, rows in groups.items():
            a, b = predictions(model, rows)
            scores[group] = summarize(rows, a, b)
        result[name] = {"model_sha256": hashlib.sha256(model_bytes).hexdigest(), "groups": scores}
        if name == "expanded_candidate":
            np.testing.assert_allclose(scores["expanded_frozen_test"]["raw_accuracy"], report["raw_classifier_accuracy"])
            np.testing.assert_allclose(scores["expanded_frozen_test"]["gated_accuracy"], report["gated_accuracy_at_threshold_085"])
    (ROOT / "datasets/motion-candidate-evaluation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    for name in ["published_baseline", "expanded_candidate"]:
        print(json.dumps({name: {group: result[name]["groups"][group] for group in ["expanded_frozen_test", "original_ten_test_clips", "asllvd_independent_research_only"]}}, indent=2))


if __name__ == "__main__":
    main()
