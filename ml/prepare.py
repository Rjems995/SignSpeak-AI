"""Deduplicate before splitting; retain IDs when given. No image or augmentation leakage."""
import argparse
import json
from pathlib import Path
from .data_quality import curate, load_records

ROOT = Path(__file__).resolve().parents[1]


def prepare(paths, drop_conflicts=False):
    records = load_records(paths)
    if any("frames" in row for row in records):
        raise ValueError("Use ml.prepare_motion for movement datasets")
    samples, report = curate(records)
    if report["invalid"]:
        raise ValueError(f"{len(report['invalid'])} invalid sample(s). Run ml.audit_dataset for details.")
    if report["conflicts"] and not drop_conflicts:
        raise ValueError(f"{len(report['conflicts'])} coordinate groups have conflicting labels. Audit them, then use --drop-conflicts to quarantine all conflicting groups.")
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "datasets" / "prepared.json")
    parser.add_argument("--drop-conflicts", action="store_true", help="Exclude every coordinate group assigned conflicting labels")
    args = parser.parse_args()
    samples = prepare(args.inputs, args.drop_conflicts)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(samples, separators=(",", ":")), encoding="utf-8")
    print(f"Prepared {len(samples)} unique, normalized samples -> {args.output}")


if __name__ == "__main__":
    main()

