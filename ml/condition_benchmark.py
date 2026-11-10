"""Evaluate actual source illumination labels and paired camera views without training."""
import argparse
import hashlib
import json
from collections import Counter
import numpy as np
from .data_quality import ROOT, curate, load_records
from .import_public import source_metadata
from .import_motion_expansion import extract_video
from .condition_evidence import summarize_conditions
from .features import normalize
from .compare_public_models import predict
from .compare_motion_models import predictions as motion_predictions


def extract_views():
    manifest = json.loads((ROOT / "datasets/public/asllvd/selection.json").read_text())
    records = []
    source = manifest["source"]["url"]
    for item in manifest["files"]:
        if item["letter"] not in {"J", "Z"}:
            continue
        path = ROOT / "datasets/public/asllvd" / item["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]:
            raise ValueError("Research source clip hash mismatch")
        for view in ["frontal", "profile_right"]:
            metadata = {"letter": item["letter"], "participant": item["participant"], "session": item["session"],
                        "performance_id": "asllvd/"+item["file"], "clip": "asllvd/"+item["file"]+"/"+view,
                        "source_sha256": item["sha256"], "hand": "unknown", "signer_dominance": "right", "camera_view": view,
                        "lighting": "unknown", "lighting_direction": "unknown", "angle_degrees": None,
                        "lighting_processing": "brightness_normalized", "training_eligible": False,
                        "source": "ASLLVD", "source_url": source, "license": manifest["source"]["license"], "source_row": item["source_row"],
                        "metadata_evidence": {
                            "signer_dominance": {"value": "right", "basis": "publisher_protocol", "source": source,
                                                "locator": "Video recording and processing: consultants right-hand dominant; actual hand not annotated per token"},
                            "camera_view": {"value": view, "basis": "publisher_camera", "source": source,
                                            "locator": "Processed video contains synchronized frontal and signer-right profile views; frontal panel above profile panel"}}}
            bounds = (50, 50+item["source_end_frame"]-item["source_start_frame"])
            try:
                record = extract_video(path, metadata, fps_override=60, bounds=bounds, panel=view)
            except ValueError as error:
                # Capture failures remain in the end-to-end denominator. Source
                # condition annotations do not depend on detector success.
                record = {**metadata, "frames": [], "extraction_failure": str(error)}
            records.append(record)
    target = ROOT / "artifacts/conditions"
    target.mkdir(parents=True, exist_ok=True)
    (target / "asllvd-views.json").write_text(json.dumps({"samples": records}, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"views": len(records), "performances": len({r["performance_id"] for r in records}), "tracking_failures": sum(bool(r.get("extraction_failure")) for r in records)}, indent=2))


def accuracy(rows, raw, gated):
    truth = np.array([r["letter"] for r in rows])
    letters = sorted(set(truth))
    return {"source_samples": len(rows), "raw_correct": int((raw == truth).sum()), "gated_correct": int((gated == truth).sum()),
            "raw_accuracy": float((raw == truth).mean()), "gated_accuracy": float((gated == truth).mean()),
            "macro_gated_recall": float(np.mean([(gated[truth == label] == label).mean() for label in letters])),
            "class_support": dict(Counter(truth)), "capture_failures": sum(bool(r.get("extraction_failure")) for r in rows)}


def static_predictions(model, rows):
    raw, gated = np.full(len(rows), "_", dtype="U1"), np.full(len(rows), "_", dtype="U1")
    valid = [i for i, row in enumerate(rows) if row.get("points")]
    if valid:
        x = np.array([normalize(rows[i]["points"]) for i in valid])
        a, b = predict(model, x)
        raw[valid], gated[valid] = a, b
    return raw, gated


def report():
    massey, audit = curate(load_records([ROOT / "datasets/massey-landmarks.json"]))
    if audit["invalid"] or audit["conflicts"]:
        raise ValueError("Lighting source has invalid/conflicting geometry")
    import_report = json.loads((ROOT / "datasets/massey-import-report.json").read_text())
    failed = [{**source_metadata("massey", row["file"]), "extraction_failure": row["reason"]} for row in import_report["rejections"]]
    lighting_rows = massey+failed
    views = json.loads((ROOT / "artifacts/conditions/asllvd-views.json").read_text())["samples"]
    combined = summarize_conditions(massey+views)
    public_training = load_records([ROOT / "datasets/public-balanced.json"])
    if {r["fingerprint"] for r in massey} & {r["fingerprint"] for r in public_training}:
        raise ValueError("Lighting benchmark shares geometry with the frozen public training candidate")
    result = {"scope": "Diagnostic source-condition benchmark; no model fitting or parameter changes.",
              "light_annotations": summarize_conditions(massey), "camera_annotations": summarize_conditions(views),
              "verified_left_hand_examples_found": combined["documented_physical_hands"]["left"], "joint_hand_lighting_view_examples": combined["hand_lighting_view_together"],
              "remaining_gaps": ["Verified physical left-hand ASL alphabet images", "Both physical hands jointly annotated with lighting and camera view", "Per-image calibrated camera angles", "Bright/soft/dim intensity and lux annotations", "Balanced condition coverage per signer and letter"],
              "lighting_source": {"url": import_report["source"]["paper_url"], "selected": len(lighting_rows), "usable_coordinates": len(massey), "tracking_failures": len(failed),
                                  "limits": "Segmented JPEG mirror; source labels are dataset-provided, not newly expert-certified. Unequal class/signer distributions and detection failures confound lighting comparisons. Capture setup is documented; actual per-image camera angle and physical hand are unknown."},
              "camera_source": {"url": "https://www.bu.edu/asllrp/av/dai-asllvd.html", "terms": "Research/education only; raw videos, signer mappings and coordinates are not redistributed", "views": len(views), "performances": len({r['performance_id'] for r in views}),
                                "limits": "Two views of each of five existing performances are correlated, not ten independent recordings. Source signer dominance does not annotate the actual hand per token. Physical illumination is unknown; brightness normalization is postprocessing. Frontal footage was previously used for diagnostic model review, so this is not a fresh untouched corpus."},
              "static_models": {}, "motion_models": {}}
    static_dirs = {"published_static": ROOT / "signspeak/static/models"}
    if (ROOT / "artifacts/public-candidate/classifier.json").exists():
        static_dirs["frozen_public_static_candidate"] = ROOT / "artifacts/public-candidate"
    for name, directory in static_dirs.items():
        data = (directory / "classifier.json").read_bytes()
        model = json.loads(data)
        raw, gated = static_predictions(model, lighting_rows)
        scores = {}
        for direction in sorted({r["lighting_direction"] for r in lighting_rows}):
            indices = [i for i, row in enumerate(lighting_rows) if row["lighting_direction"] == direction]
            rows = [lighting_rows[i] for i in indices]
            scores[direction] = accuracy(rows, raw[indices], gated[indices])
        result["static_models"][name] = {"model_sha256": hashlib.sha256(data).hexdigest(), "all_selected_images": accuracy(lighting_rows, raw, gated), "by_light_direction": scores}
    for name, directory in [("original_motion", ROOT / "datasets/baselines/motion-v1"), ("installed_motion", ROOT / "signspeak/static/models")]:
        data = (directory / "motion-classifier.json").read_bytes()
        model = json.loads(data)
        raw, gated = motion_predictions(model, views)
        scores = {}
        for view in ["frontal", "profile_right"]:
            ids = [i for i, row in enumerate(views) if row["camera_view"] == view]
            scores[view] = accuracy([views[i] for i in ids], raw[ids], gated[ids])
        result["motion_models"][name] = {"model_sha256": hashlib.sha256(data).hexdigest(), "by_camera_view": scores}
    path = ROOT / "datasets/condition-benchmark-report.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    summary = {"status": "partial_coverage", "verified_left": result["verified_left_hand_examples_found"], "joint_complete": result["joint_hand_lighting_view_examples"],
               "lighting_samples": len(massey), "camera_views": len(views), "camera_performances": result["camera_source"]["performances"],
               "notice": "Public annotation gaps remain: verified physical left-hand alphabet examples and samples jointly labeled with hand, illumination and camera view are still missing."}
    (ROOT / "signspeak/static/data/condition-coverage.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({**summary, "light_directions": result["light_annotations"]["documented_lighting_directions"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["extract-views", "report"])
    args = parser.parse_args()
    extract_views() if args.action == "extract-views" else report()
