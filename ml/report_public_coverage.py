"""Publish measured source audits, distinguishing training data from review-only data."""
import json
from collections import Counter
from .data_quality import ROOT, curate, load_records
from .import_public import SOURCES


def main():
    specs = [("aslyset", "ASLYset", ROOT / "datasets/aslyset-landmarks.json", "https://data.mendeley.com/datasets/xs6mvhx6rh/1", True,
              "CC BY 4.0. Right-hand images only; per-image lighting, camera angle and session labels are unknown.")]
    for key, source in SOURCES.items():
        directory = ROOT / ("datasets" if source["training_eligible"] else "artifacts/public-review")
        limitation = source["condition_metadata"]
        if key == "massey":
            limitation += " No explicit redistribution license located; coordinates remain in the ignored local review cache."
        specs.append((key, source["name"], directory / f"{key}-landmarks.json", source["url"], source["training_eligible"], limitation))
    reports = []
    for key, name, path, url, eligible, limitation in specs:
        if not path.exists():
            raise ValueError(f"Import {key} before updating the published coverage audit")
        clean, audit = curate(load_records([path]))
        if audit["invalid"] or audit["conflicts"]:
            raise ValueError(f"Resolve invalid/conflicting records in {key}")
        hands = Counter(r.get("hand", "unknown") for r in clean)
        directions = Counter(r["lighting_direction"] for r in clean if r.get("lighting_direction", "unknown") != "unknown")
        reports.append({"name": name, "source_key": key, "samples": len(clean),
                        "participants": len({r["participant"] for r in clean if r.get("participant")}),
                        "q_samples": sum(r["letter"] == "Q" for r in clean),
                        "left": hands["left"], "right": hands["right"], "hand_unknown": hands["unknown"],
                        "lighting_directions": dict(sorted(directions.items())),
                        "usage": "Training pool" if eligible else "Review only — excluded from training",
                        "url": url, "limitations": limitation,
                        "curation": {**{k: v for k, v in audit.items() if k not in {"invalid", "conflicts", "cross_participant_duplicates"}},
                                     "cross_participant_quarantined_rows": sum(group["rows"] for group in audit["cross_participant_duplicates"])}})
    (ROOT / "signspeak/static/data/source-coverage.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    print(json.dumps([{k: r[k] for k in ["name", "samples", "participants", "q_samples", "usage"]} for r in reports], indent=2))


if __name__ == "__main__":
    main()
