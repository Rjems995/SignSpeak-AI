"""Select equal real examples per signer/letter/hand. Never duplicate or fabricate."""
import argparse
import itertools
import json
import random
from collections import defaultdict
from pathlib import Path
from .data_quality import curate, load_records, UNKNOWN
from .features import LABELS


def balance(records, per_cell, hands, letters):
    clean, report = curate(records)
    if report["invalid"] or report["conflicts"]:
        raise ValueError("Audit/quarantine invalid or conflicting samples before balancing")
    people = sorted({r.get("participant") for r in clean if r.get("participant") not in UNKNOWN})
    if not people:
        raise ValueError("Real participant identifiers are required for signer-balanced selection")
    groups = defaultdict(list)
    for row in clean:
        if row.get("participant") in people and row.get("hand") in hands and row["letter"] in letters:
            groups[(row["participant"], row["hand"], row["letter"])].append(row)
    cells = list(itertools.product(people, hands, letters))
    missing = [(person, hand, letter, len(groups[(person, hand, letter)])) for person, hand, letter in cells if len(groups[(person, hand, letter)]) < per_cell]
    if missing:
        raise ValueError(f"{len(missing)} cells have fewer than {per_cell} real samples. Examples: {missing[:8]}")
    rng = random.Random(42)
    selected = []
    for cell in cells:
        rows = sorted(groups[cell], key=lambda r: r["fingerprint"])
        selected.extend(rng.sample(rows, per_cell))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--per-cell", type=int, default=20)
    parser.add_argument("--hands", nargs="+", choices=["left", "right"], default=["left", "right"])
    parser.add_argument("--kind", choices=["static", "motion"], default="static")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.per_cell < 1:
        parser.error("--per-cell must be positive")
    rows = balance(load_records(args.inputs), args.per_cell, list(dict.fromkeys(args.hands)), LABELS if args.kind == "static" else ["J", "Z"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, separators=(",", ":")), encoding="utf-8")
    print(f"Selected {len(rows)} original examples with equal signer/letter/hand counts. No examples were repeated.")


if __name__ == "__main__":
    main()
