"""Extract a deterministic subset from verified public ZIPs without running source code.

Only explicit source metadata becomes ground truth. Detector handedness remains
an observation, never a physical-hand annotation. Unclear-rights/augmentation
sources are retained under ignored artifacts for review, outside training.
"""
import argparse
import hashlib
import json
import re
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .features import LABELS, normalize

ROOT = Path(__file__).resolve().parents[1]
SOURCES = json.loads((ROOT / "datasets/public-sources.json").read_text(encoding="utf-8"))
PRIORITY = set("QESMNTUVRP")
LIGHTS = {"bot": "bottom", "top": "top", "left": "left", "right": "right", "dif": "diffuse"}
PATTERNS = {
    "asl_hg": re.compile(r"asl_dataset/([A-Z0-9])/P(10|[1-9])_\1_([0-9]+)\.jpg"),
    "signalphaset": re.compile(r"SignAlphaSet/([A-Z])/\1_([0-9]+)\.jpg"),
    "massey": re.compile(r"ASL-Finger-Spelling-Recognition-[0-9a-f]{40}/asl_dataset/([a-z0-9])/hand([1-5])_\1_(bot|top|left|right|dif)_seg_([0-9]+)_cropped\.jpeg"),
}


def source_metadata(source, name):
    match = PATTERNS[source].fullmatch(name)
    if not match or match[1].upper() not in LABELS:
        return None  # Also excludes J/Z stills: motion needs actual sequences.
    row = {"letter": match[1].upper(), "participant": None, "session": None,
           "hand": "unknown", "lighting": "unknown", "angle": "unknown",
           "lighting_direction": "unknown", "label_review": "dataset_provided",
           "source": SOURCES[source]["name"], "source_url": SOURCES[source]["url"],
           "license": SOURCES[source]["license"], "source_file": name,
           "training_eligible": SOURCES[source]["training_eligible"]}
    if source == "asl_hg":
        row["participant"] = "asl_hg_p" + match[2]
    elif source == "massey":
        row["participant"] = "massey_hand" + match[2]
        row["lighting_direction"] = LIGHTS[match[3]]
        row["lighting_direction_source"] = "source_filename"
    return row


def select_members(source, names, per_cell=20, priority_extra=10):
    if per_cell < 1 or priority_extra < 0:
        raise ValueError("Selection counts must be positive, with nonnegative priority extras")
    groups = defaultdict(list)
    for name in names:
        row = source_metadata(source, name)
        if row:
            # Distinct source illumination directions remain represented in the
            # small Massey review subset. No claim about illumination intensity.
            cell = (row["letter"], row["participant"], row["lighting_direction"])
            groups[cell].append(name)
    result = []
    for (letter, _, _), members in sorted(groups.items(), key=lambda item: str(item[0])):
        ordered = sorted(members, key=lambda name: hashlib.sha256(name.encode()).hexdigest())
        limit = per_cell + (priority_extra if letter in PRIORITY else 0)
        result.extend(ordered[:limit])
    return result


def main():
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument("--per-cell", type=int, default=20)
    parser.add_argument("--priority-extra", type=int, default=10)
    args = parser.parse_args()
    source = SOURCES[args.source]
    archive = ROOT / "datasets/public" / args.source / source["archive"]
    with archive.open("rb") as stream:
        actual = hashlib.file_digest(stream, "sha256").hexdigest()
    if archive.stat().st_size != source["bytes"] or actual != source["sha256"]:
        raise ValueError("Archive verification failed; run scripts/download-public-source.mjs")
    output_directory = ROOT / ("datasets" if source["training_eligible"] else "artifacts/public-review")
    output_directory.mkdir(parents=True, exist_ok=True)
    output = output_directory / f"{args.source}-landmarks.json"
    model_path = ROOT / "datasets/videos/hand_landmarker.task"
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=vision.RunningMode.IMAGE, num_hands=2,
        min_hand_detection_confidence=.6, min_hand_presence_confidence=.6)
    records, rejected = [], []
    with zipfile.ZipFile(archive) as zipped, vision.HandLandmarker.create_from_options(options) as detector:
        names = zipped.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate archive paths are not accepted")
        selected = select_members(args.source, names, args.per_cell, args.priority_extra)
        if not selected:
            raise ValueError("No matching static images found")
        for index, member in enumerate(selected):
            row = source_metadata(args.source, member)
            # Read only regex-validated entries directly; no archive extraction,
            # symlink following or source-repository code execution.
            entry = zipped.getinfo(member)
            if entry.file_size > 20_000_000:
                raise ValueError("Unexpected image size")
            image_bytes = zipped.read(entry)
            pixels = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            reason = None
            if pixels is None:
                reason = "decode_failed"
            else:
                result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))
                if len(result.hand_landmarks) != 1:
                    reason = "requires_one_detected_hand"
                else:
                    points = [value for point in result.hand_landmarks[0] for value in (point.x, point.y)]
                    try:
                        normalize(points)
                    except ValueError:
                        reason = "invalid_geometry"
            if reason:
                rejected.append({"file": member, "letter": row["letter"], "reason": reason})
            else:
                observation = result.handedness[0][0]
                records.append({**row, "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
                                "observed_hand": observation.category_name.lower(),
                                "observed_hand_score": observation.score,
                                "observed_hand_source": "mediapipe_estimate_image_mirroring_unknown",
                                "points": points})
            if (index+1) % 400 == 0:
                print(f"{args.source}: {index+1}/{len(selected)} processed; {len(records)} accepted", flush=True)
    output.write_text(json.dumps({"source": source["name"], "version": 1, "samples": records}, separators=(",", ":")), encoding="utf-8")
    report = {"source": source, "selection": {"method": "SHA-256 filename order within source/signer/letter (and light direction when documented)",
              "per_cell": args.per_cell, "priority_extra": args.priority_extra, "priority_letters": sorted(PRIORITY)},
              "selected": len(selected), "accepted": len(records), "rejected": len(rejected),
              "counts_by_letter": dict(sorted(Counter(r["letter"] for r in records).items())),
              "participant_ids": sorted({r["participant"] for r in records if r["participant"]}),
              "physical_hands": dict(Counter(r["hand"] for r in records)),
              "lighting_directions": dict(Counter(r["lighting_direction"] for r in records)),
              "estimated_hands_not_ground_truth": dict(Counter(r["observed_hand"] for r in records)),
              "rejections": rejected}
    (ROOT / f"datasets/{args.source}-import-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"source": source["name"], "selected": len(selected), "accepted": len(records), "rejected": len(rejected), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
