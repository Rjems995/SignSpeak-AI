"""Audit real sample counts, conflicts, signer/hand balance and condition gaps."""
import argparse
import json
from pathlib import Path
from .data_quality import curate, coverage, load_records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--kind", choices=["all", "static", "motion"], default="all")
    parser.add_argument("--output", type=Path, default=Path("datasets/private/coverage.json"))
    parser.add_argument("--require-targets", action="store_true")
    args = parser.parse_args()
    clean, curation = curate(load_records(args.inputs))
    report = {"curation": curation, "coverage": coverage(clean, args.kind)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"usable_unique": len(clean), "label_conflicts": len(curation["conflicts"]), "invalid": len(curation["invalid"]), "participants": report["coverage"]["participant_count"], "Q": report["coverage"]["letters"].get("Q"), "targets_met": report["coverage"]["collection_targets_met"]}, indent=2))
    if args.require_targets and not report["coverage"]["collection_targets_met"]:
        raise SystemExit("Collection targets are not met. See the coverage report for remaining gaps.")


if __name__ == "__main__":
    main()
