"""Import actual ASLYset images with publisher signer IDs and honest metadata."""
import hashlib
import json
import re
import subprocess
from collections import Counter
from pathlib import Path, PurePosixPath
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .download_aslyset import ROOT, SHA256, SOURCE
from .features import normalize, LABELS


def main():
    directory = ROOT / "datasets" / "public" / "aslyset"
    archive = directory / "ASLYset.rar"
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != SHA256:
            raise ValueError("Run ml.download_aslyset to obtain the verified archive")
    listing = subprocess.check_output(["tar", "-tf", str(archive)], text=True).splitlines()
    pattern = re.compile(r"^ASLYset/images/(User[1-4])/([a-z]+)/\1_([A-Z]+)_([0-9]+)\.png$")
    selected = []
    for name in listing:
        match = pattern.fullmatch(name.replace("\\", "/"))
        if match and match[3] in LABELS:
            member = PurePosixPath(name)
            target = (directory / Path(*member.parts)).resolve()
            if not target.is_relative_to(directory.resolve()):
                raise ValueError("Archive path leaves the intended directory")
            selected.append(name)
    if len(selected) != 4800:
        raise ValueError(f"Expected 4 signers x 24 static letters x 50 images; found {len(selected)}")
    verbose = subprocess.check_output(["tar", "-tvf", str(archive)], text=True).splitlines()
    regular = {line[line.index("ASLYset/"):] for line in verbose if line.startswith("-") and "ASLYset/" in line}
    if not set(selected).issubset(regular):
        raise ValueError("Selected archive entries must all be regular files")
    if any(not (directory / member).exists() for member in selected):
        members = directory / "selected-members.txt"
        members.write_text("\n".join(selected)+"\n", encoding="utf-8")
        subprocess.run(["tar", "-xf", str(archive), "-C", str(directory), "-T", str(members), "--safe-writes"], check=True)
    model_path = ROOT / "datasets" / "videos" / "hand_landmarker.task"
    if not model_path.exists():
        raise ValueError("Download Google's hand_landmarker.task model as described in README.md")
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model_path)), running_mode=vision.RunningMode.IMAGE, num_hands=2, min_hand_detection_confidence=.6, min_hand_presence_confidence=.6)
    records, rejected = [], []
    raw_counts, counts = Counter(), Counter()
    with vision.HandLandmarker.create_from_options(options) as detector:
        for index, member in enumerate(selected):
            match = pattern.fullmatch(member)
            person, letter = match[1], match[3]
            raw_counts[letter] += 1
            image_bytes = (directory/member).read_bytes()
            pixels = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            if pixels is None:
                rejected.append({"file": member, "letter": letter, "reason": "decode_failed"})
                continue
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))
            if len(result.hand_landmarks) != 1:
                rejected.append({"file": member, "letter": letter, "reason": "requires_one_detected_hand"})
                continue
            points = [value for p in result.hand_landmarks[0] for value in (p.x, p.y)]
            try:
                normalize(points)
            except ValueError:
                rejected.append({"file": member, "letter": letter, "reason": "invalid_geometry"})
                continue
            handedness = result.handedness[0][0]
            records.append({"letter": letter, "participant": "aslyset_"+person.lower(), "session": None,
                            "hand": "right", "hand_source": "publisher_description", "lighting": "unknown", "angle": "unknown",
                            "label_review": "dataset_provided", "source": "ASLYset", "source_url": SOURCE,
                            "license": "CC BY 4.0", "source_file": member, "source_sha256": hashlib.sha256(image_bytes).hexdigest(),
                            "observed_hand": handedness.category_name.lower(), "observed_hand_score": handedness.score, "points": points})
            counts[letter] += 1
            if (index+1) % 400 == 0:
                print(f"Processed {index+1}/4800; kept {len(records)} actual hand samples", flush=True)
    output = ROOT / "datasets" / "aslyset-landmarks.json"
    output.write_text(json.dumps({"source": "ASLYset", "version": 1, "samples": records}, separators=(",", ":")), encoding="utf-8")
    report = {"source": SOURCE, "license": "CC BY 4.0", "image_count": len(selected), "accepted": len(records), "rejected": len(rejected),
              "counts_by_letter": dict(sorted(counts.items())), "raw_counts_by_letter": dict(sorted(raw_counts.items())),
              "participant_ids": sorted({r["participant"] for r in records}), "hands": {"right": len(records), "left": 0},
              "unknown_metadata": ["lighting", "angle", "session"], "rejections": rejected}
    (ROOT / "datasets" / "aslyset-import-report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"accepted": len(records), "Q": counts["Q"], "participants": report["participant_ids"], "rejected": len(rejected)}, indent=2))


if __name__ == "__main__":
    main()
