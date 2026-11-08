"""Extract timestamped one-hand sequences from the downloaded J/Z clips."""
import json
from pathlib import Path
from urllib.request import urlopen
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from .motion_features import motion_features

ROOT = Path(__file__).resolve().parents[1]


def main():
    model = ROOT / "datasets" / "videos" / "hand_landmarker.task"
    if not model.exists():
        with urlopen("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task", timeout=60) as response:
            model.write_bytes(response.read())
    records, failures = [], []
    for path in sorted((ROOT / "datasets" / "videos").glob("*/*.avi")):
        options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(model)), running_mode=vision.RunningMode.VIDEO, num_hands=2)
        capture = cv2.VideoCapture(str(path))
        fps = capture.get(cv2.CAP_PROP_FPS) or 30
        stride = max(1, round(fps/12))
        frames, total, index = [], 0, 0
        with vision.HandLandmarker.create_from_options(options) as detector:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if index % stride == 0:
                    total += 1
                    timestamp = round(index/fps*1000)
                    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    result = detector.detect_for_video(image, timestamp)
                    if len(result.hand_landmarks) == 1:
                        frames.append({"t": timestamp, "points": [value for p in result.hand_landmarks[0] for value in (p.x, p.y)]})
                index += 1
        capture.release()
        clip = f"{path.parent.name}/{path.name}"
        try:
            if total == 0 or len(frames)/total < .6:
                raise ValueError("Too few frames with exactly one tracked hand")
            features = motion_features(frames)
            records.append({"letter": path.parent.name, "clip": clip, "participant": None, "frames": frames, "features": features.tolist()})
            print(f"Extracted {clip}: {len(frames)} frames", flush=True)
        except ValueError as error:
            failures.append({"clip": clip, "reason": str(error)})
            print(f"Skipped {clip}: {error}", flush=True)
    (ROOT / "datasets" / "motion.json").write_text(json.dumps({"samples": records, "failures": failures}, separators=(",", ":")), encoding="utf-8")
    print(f"Extracted {len(records)} real sequences; {len(failures)} rejected clips.")


if __name__ == "__main__":
    main()
