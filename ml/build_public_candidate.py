"""Balance only verified raw sources by signer/letter; never fill missing cells."""
import argparse
import json
from collections import defaultdict
from .data_quality import ROOT, curate, coverage, load_records
from .features import LABELS


def select_balanced(records, cap=20):
    if cap < 1:
        raise ValueError("cap must be positive")
    clean, audit = curate(records)
    if audit["invalid"] or audit["conflicts"]:
        raise ValueError("Resolve invalid geometry or conflicting labels before selecting data")
    if any(not row.get("participant") or row.get("training_eligible") is False for row in clean):
        raise ValueError("Only identified signers from training-eligible original sources are permitted")
    people = sorted({row["participant"] for row in clean})
    groups = defaultdict(list)
    for row in clean:
        if row["letter"] in LABELS:
            groups[(row["participant"], row["letter"])].append(row)
    if not people:
        raise ValueError("No identified samples")
    per_cell = min(cap, min(len(groups[(person, letter)]) for person in people for letter in LABELS))
    if not per_cell:
        raise ValueError("Every signer must have each static letter; collect missing cells instead of repeating images")
    selected = []
    for person in people:
        for letter in LABELS:
            selected.extend(sorted(groups[(person, letter)], key=lambda r: r["fingerprint"])[:per_cell])
    return selected, {"per_signer_letter": per_cell, "participants": people,
                      "balance_axes": ["participant", "letter"],
                      "warning": "This is signer/letter balance, not physical-hand or condition balance.", "curation": audit}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=int, default=20)
    args = parser.parse_args()
    records = load_records([ROOT / "datasets/aslyset-landmarks.json", ROOT / "datasets/asl_hg-landmarks.json"])
    selected, report = select_balanced(records, args.cap)
    output = ROOT / "datasets/public-balanced.json"
    output.write_text(json.dumps(selected, separators=(",", ":")), encoding="utf-8")
    report["coverage"] = coverage(selected, kind="static")
    (ROOT / "datasets/public-quality-audit.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"samples": len(selected), "participants": len(report["participants"]), "per_signer_letter": report["per_signer_letter"], "per_letter": len(selected)//len(LABELS)}, indent=2))


if __name__ == "__main__":
    main()
