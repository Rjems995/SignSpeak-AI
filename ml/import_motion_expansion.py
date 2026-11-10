"""Extract verified original J/Z videos; retain source IDs and audit natural variation."""
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from .data_quality import ROOT, curate
from .motion_features import motion_features
from .motion_audit import motion_observations, ranges


def extract_video(path, metadata, fps_override=None, bounds=None, frontal_panel=False, keep_stationary_control=False):
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    model = ROOT / "datasets/videos/hand_landmarker.task"
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO, num_hands=2)
    cap = cv2.VideoCapture(str(path))
    fps = fps_override or cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, round(fps/30))  # Fast signs need more than the old 12 fps sampler.
    frames, total, index = [], 0, 0
    try:
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, pixels = cap.read()
                if not ok:
                    break
                include = index % stride == 0 and (bounds is None or bounds[0] <= index <= bounds[1])
                if include:
                    if frontal_panel:
                        # ASLLVD's processed MOV stacks frontal then profile
                        # views. Never count the synchronized second view as a
                        # different performance or signer.
                        pixels = pixels[:pixels.shape[0]//2].copy()
                    timestamp = round(index/fps*1000)
                    total += 1
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB))
                    result = detector.detect_for_video(image, timestamp)
                    if len(result.hand_landmarks) == 1 or (frontal_panel and result.hand_landmarks):
                        # In this corpus one hand signs while the other rests
                        # at the waist. Select the raised hand in the frontal
                        # panel; this is a documented extraction rule, not a
                        # detector-generated physical-handedness annotation.
                        hand = min(result.hand_landmarks, key=lambda points: points[0].y) if frontal_panel else result.hand_landmarks[0]
                        frames.append({"t": timestamp, "points": [value for point in hand for value in (point.x, point.y)]})
                index += 1
    finally:
        cap.release()
    if not total or len(frames)/total < .6:
        raise ValueError("Too few sampled frames with exactly one detected hand")
    rejection = None
    try:
        motion_features(frames)
    except ValueError as error:
        if keep_stationary_control and metadata["letter"] == "_" and "stationary" in str(error):
            rejection = str(error)
        else:
            raise
    row = {**metadata, "frames": frames, "observations": motion_observations(frames),
           "tracking": {"sampled_frames": total, "accepted_frames": len(frames), "effective_fps": fps, "stride": stride}}
    if rejection:
        row["feature_rejection"] = rejection
    return row


def main():
    sources = json.loads((ROOT / "datasets/motion-sources.json").read_text())
    signalpha = sources["signalpha_motion"]
    directory = ROOT / "datasets/public/signalpha_motion"
    archive = directory / signalpha["archive"]
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != signalpha["sha256"]:
            raise ValueError("SignAlpha archive SHA-256 mismatch")
    provenance = json.loads((ROOT / "datasets/motion-expanded-provenance.json").read_text())
    inputs = []
    # Frozen baseline sequences keep their original extraction and fingerprint,
    # allowing a strict comparison on the original held-out examples.
    original = json.loads((ROOT / "datasets/motion.json").read_text())["samples"]
    original_by_clip = {r["clip"]: r for r in original}
    old_hashes = {r["file"]: r["sha256"] for r in json.loads((ROOT / "datasets/motion-provenance.json").read_text())["files"]}
    records, failures = [], []
    for item in provenance["files"]:
        path = ROOT / "datasets/videos/expanded/signn" / item["file"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError("SigNN clip SHA-256 mismatch")
        metadata = {"letter": Path(item["file"]).parts[0], "clip": "signn/"+item["file"], "legacy_clip": item["file"],
                    "participant": None, "session": None, "hand": "unknown", "lighting": "unknown", "angle": "unknown",
                    "source": "SigNN J/Z V1", "source_url": provenance["source"], "license": provenance["license"],
                    "source_file": item["upstream"], "source_sha256": actual, "label_review": "dataset_provided"}
        if item["file"] in original_by_clip:
            if actual != old_hashes[item["file"]]:
                raise ValueError("Baseline clip changed")
            row = original_by_clip[item["file"]]
            records.append({**metadata, "frames": row["frames"], "observations": motion_observations(row["frames"]), "extraction": "frozen_baseline"})
        else:
            inputs.append((path, metadata))
    pattern = re.compile(r"ASL_dynamic/([A-Z]+)/\1_clip([0-9]+)\.avi")
    with zipfile.ZipFile(archive) as zipped:
        if len(zipped.namelist()) != len(set(zipped.namelist())):
            raise ValueError("Duplicate ZIP paths")
        for member in sorted(zipped.namelist()):
            match = pattern.fullmatch(member)
            # Moving I/D and static-letter transitions provide real non-J/Z
            # examples. Keep source labels; never relabel them as J or Z.
            if not match or match[1] not in {"J", "Z", "I", "D", "A", "S", "L", "R"}:
                continue
            if match[1] not in {"J", "Z"} and int(match[2]) > 4:
                continue
            data = zipped.read(member)
            path = directory / "clips" / Path(member).parent.name / Path(member).name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            inputs.append((path, {"letter": match[1] if match[1] in {"J", "Z"} else "_", "source_label": match[1],
                                  "clip": "signalpha/"+member, "participant": None, "session": None,
                                  "hand": "unknown", "lighting": "unknown", "angle": "unknown",
                                  "source": signalpha["name"], "source_url": signalpha["url"], "license": signalpha["license"],
                                  "source_file": member, "source_sha256": hashlib.sha256(data).hexdigest(), "label_review": "dataset_provided"}))
    seen = {r["source_sha256"]: r["letter"] for r in records}
    duplicate_videos = []
    for index, (path, metadata) in enumerate(inputs):
        if metadata["source_sha256"] in seen:
            if seen[metadata["source_sha256"]] != metadata["letter"]:
                raise ValueError("The same video has conflicting labels; quarantine/review before import")
            duplicate_videos.append(metadata["clip"])
            continue
        seen[metadata["source_sha256"]] = metadata["letter"]
        try:
            records.append(extract_video(path, metadata))
        except ValueError as error:
            failures.append({"clip": metadata["clip"], "letter": metadata["letter"], "reason": str(error)})
        if (index+1) % 20 == 0:
            print(f"Processed {index+1}/{len(inputs)} new/retried clips; {len(records)} accepted total", flush=True)
    clean, audit = curate(records)
    if audit["invalid"] or audit["conflicts"]:
        raise ValueError("Resolve invalid or conflicting extracted movement records")
    # Features are recomputed during training, keeping source data compact.
    for row in clean:
        row.pop("features", None)
    output = ROOT / "datasets/motion-expanded.json"
    output.write_text(json.dumps({"samples": clean, "failures": failures}, separators=(",", ":")), encoding="utf-8")
    report = {"sources": [provenance["source"], signalpha["url"]], "accepted": len(clean), "class_counts": dict(Counter(r["letter"] for r in clean)),
              "curation": audit, "duplicate_video_files": duplicate_videos, "failures": failures,
              "participant_metadata": "Unavailable per clip in both sources; no new signer count or signer-disjoint accuracy is claimed.",
              "observed_ranges": {label: ranges([r for r in clean if r["letter"] == label]) for label in ["J", "Z", "_"]},
              "warning": "Speed and size values are measured landmark proxies from original recordings, not annotated signing speed, artificial video retiming, or additional real clips."}
    (ROOT / "datasets/motion-expansion-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": len(clean), "classes": report["class_counts"], "failed": len(failures)}, indent=2))


if __name__ == "__main__":
    main()
