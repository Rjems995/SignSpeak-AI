"""Combine consented browser sequence exports for motion training."""
import argparse
import json
from pathlib import Path
from .motion_features import motion_features


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/private/motion.json"))
    args = parser.parse_args()
    records = []
    for path in args.inputs:
        for record in json.loads(path.read_text(encoding="utf-8"))["samples"]:
            if record.get("letter") not in {"J", "Z"} or "frames" not in record:
                raise ValueError("Use separate J/Z motion exports, not static-coordinate datasets")
            motion_features(record["frames"])
            records.append(record)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"samples": records}, separators=(",", ":")), encoding="utf-8")
    print(f"Combined {len(records)} recorded motion samples")


if __name__ == "__main__":
    main()
