"""Deduplicate before splitting; retain IDs when given. No image or augmentation leakage."""
import argparse
import hashlib
import json
from pathlib import Path
from .features import normalize

ROOT = Path(__file__).resolve().parents[1]


def prepare(paths):
    samples, seen = [], set()
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(data, dict) and "samples" not in data:
            raw = [{"letter": letter, "points": points, "participant": None} for letter, rows in data.items() for points in rows]
        else:
            raw = data["samples"] if isinstance(data, dict) else data
        for row in raw:
            points = row["points"]
            features = normalize(points)
            # Fingerprint canonical coordinates, so translated/mirrored duplicates stay together.
            fingerprint = hashlib.sha256(features.round(5).tobytes()).hexdigest()
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            samples.append({"letter": row["letter"], "features": features.tolist(),
                            "participant": row.get("participant"), "fingerprint": fingerprint,
                            "points": points})
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "prepared.json")
    args = parser.parse_args()
    samples = prepare(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(samples, separators=(",", ":")), encoding="utf-8")
    print(f"Prepared {len(samples)} unique, normalized samples -> {args.output}")


if __name__ == "__main__":
    main()

