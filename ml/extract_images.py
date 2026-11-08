"""Extract real image landmarks. Layout: INPUT/participant/letter/*.jpg (or .png)."""
import argparse
import json
from pathlib import Path
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--hand-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("datasets/private/images.json"))
    args = parser.parse_args()
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(args.hand_model)), running_mode=vision.RunningMode.IMAGE, num_hands=2)
    samples, skipped = [], 0
    with vision.HandLandmarker.create_from_options(options) as detector:
        for path in sorted(args.input.glob("*/*/*")):
            if path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            letter, participant = path.parent.name.upper(), path.parent.parent.name
            result = detector.detect(mp.Image.create_from_file(str(path)))
            if len(result.hand_landmarks) != 1:
                skipped += 1
                continue
            samples.append({"letter": letter, "participant": participant, "points": [value for p in result.hand_landmarks[0] for value in (p.x, p.y)]})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"samples": samples}), encoding="utf-8")
    print(f"Extracted {len(samples)} samples; skipped {skipped} images without exactly one hand.")


if __name__ == "__main__":
    main()
