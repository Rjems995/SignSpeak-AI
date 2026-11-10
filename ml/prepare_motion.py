"""Combine consented browser sequence exports for motion training."""
import argparse
import json
from pathlib import Path
from .data_quality import curate, load_records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("datasets/private/motion.json"))
    args = parser.parse_args()
    raw = load_records(args.inputs)
    if any(record.get("letter") not in {"J", "Z", "_"} or "frames" not in record for record in raw):
        raise ValueError("Use separate J/Z motion exports, not static-coordinate datasets")
    records, report = curate(raw)
    if report["invalid"] or report["conflicts"]:
        raise ValueError("Invalid or conflicting movement samples. Run ml.audit_dataset for details.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"samples": records}, separators=(",", ":")), encoding="utf-8")
    print(f"Combined {len(records)} recorded motion samples")


if __name__ == "__main__":
    main()
