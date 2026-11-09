"""Metadata-preserving curation and coverage checks. Unknown fields stay unknown."""
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from .features import normalize
from .motion_features import motion_features

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "signspeak" / "static" / "data" / "collection-plan.json"
UNKNOWN = {None, "", "unknown"}


def load_records(paths):
    rows = []
    for path in paths:
        data = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        if isinstance(data, dict) and "samples" not in data:
            data = [{"letter": label, "points": points} for label, samples in data.items() for points in samples]
        elif isinstance(data, dict):
            data = data["samples"]
        if not isinstance(data, list):
            raise ValueError(f"{path}: expected a list of samples")
        rows.extend(data)
    return rows


def curate(records):
    grouped = defaultdict(list)
    invalid = []
    for index, record in enumerate(records):
        try:
            if not isinstance(record, dict) or record.get("letter") not in [*"ABCDEFGHIJKLMNOPQRSTUVWXYZ", "_"]:
                raise ValueError("unsupported label")
            if any(record.get(key) is not None and not isinstance(record[key], str) for key in ["participant", "session", "hand", "lighting", "angle"]):
                raise ValueError("Participant/session/condition metadata must be strings or null")
            motion = "frames" in record
            if motion != (record["letter"] in {"J", "Z"}):
                raise ValueError("J/Z require sequences; static letters require points")
            features = motion_features(record["frames"]) if motion else normalize(record["points"])
            fingerprint = hashlib.sha256(features.round(5).tobytes()).hexdigest()
            grouped[(motion, fingerprint)].append(dict(record, features=features.tolist(), fingerprint=fingerprint))
        except (ValueError, TypeError, KeyError, IndexError) as error:
            invalid.append({"index": index, "reason": str(error)})
    conflicts, cross_participant, clean = [], [], []
    duplicate_rows = 0
    for (_, fingerprint), group in grouped.items():
        labels = sorted({r["letter"] for r in group})
        people = {r.get("participant") for r in group if r.get("participant") not in UNKNOWN}
        if len(labels) > 1:
            conflicts.append({"fingerprint": fingerprint, "labels": labels, "rows": len(group)})
        elif len(people) > 1:
            cross_participant.append({"fingerprint": fingerprint, "rows": len(group)})
        else:
            # Prefer the record with actual provenance/metadata, never add it to
            # a row that did not supply it.
            clean.append(max(group, key=lambda r: sum(r.get(k) not in UNKNOWN for k in ["participant", "session", "hand", "lighting", "angle"])))
            duplicate_rows += len(group)-1
    return clean, {"input_rows": len(records), "usable_unique": len(clean), "duplicate_rows": duplicate_rows,
                   "conflicts": conflicts, "cross_participant_duplicates": cross_participant, "invalid": invalid}


def coverage(records, kind="all", plan=None):
    plan = plan or json.loads(PLAN.read_text(encoding="utf-8"))
    letters = [l for l in plan["alphabet"] if kind == "all" or ((l in {"J", "Z"}) == (kind == "motion"))]
    result = {}
    for letter in letters:
        rows = [r for r in records if r["letter"] == letter]
        people = {r.get("participant") for r in rows if r.get("participant") not in UNKNOWN}
        conditions = {key: dict(Counter(r.get(key) if r.get(key) not in UNKNOWN else "unknown" for r in rows)) for key in ["hand", "lighting", "angle"]}
        missing = []
        if len(people) < plan["target_participants"]:
            missing.append(f"participants {len(people)}/{plan['target_participants']}")
        for key, expected in [("hand", plan["hands"]), ("lighting", plan["lighting"]), ("angle", plan["angles"])]:
            absent = [value for value in expected if not conditions[key].get(value)]
            if absent:
                missing.append(f"missing {key}: {', '.join(absent)}")
        cells = Counter((r.get("participant"), r.get("hand")) for r in rows if r.get("participant") not in UNKNOWN and r.get("hand") in plan["hands"])
        if any(cells[(person, hand)] < plan["samples_per_participant_letter_hand"] for person in people for hand in plan["hands"]):
            missing.append("some signer/hand cells are below the per-letter sample target")
        sessions = {person: {r.get("session") for r in rows if r.get("participant") == person and r.get("session") not in UNKNOWN} for person in people}
        if any(len(values) < plan["sessions_per_participant"] for values in sessions.values()):
            missing.append("some signers lack the requested independent sessions")
        result[letter] = {"samples": len(rows), "participants": len(people), **conditions, "needs": missing}
    return {"kind": kind, "samples": len(records), "participant_count": len({r.get('participant') for r in records if r.get('participant') not in UNKNOWN}),
            "missing_metadata": {key: sum(r.get(key) in UNKNOWN for r in records) for key in ["participant", "session", "hand", "lighting", "angle"]},
            "letters": result, "collection_targets_met": all(not row["needs"] for row in result.values()),
            "warning": "Targets describe collection coverage, not recognition accuracy. Self-reported conditions and dataset-provided labels need review."}
