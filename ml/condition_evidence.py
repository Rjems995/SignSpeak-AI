"""Summarize previously reviewed source annotations, not arbitrary provenance claims.

Records must first be checked against original source documents. Schema checks
alone cannot authenticate an annotation or make detector estimates capture truth.
"""
from collections import Counter

TRUSTED_BASES = {"publisher_per_sample", "publisher_camera", "publisher_protocol"}
MISSING = {None, "", "unknown"}


def documented_value(row, field):
    annotations = row.get("metadata_evidence", {})
    evidence = annotations.get(field, {}) if isinstance(annotations, dict) else {}
    value = row.get(field)
    if not isinstance(value, str) or value in MISSING or not isinstance(evidence, dict) or evidence.get("value") != value:
        return None
    if evidence.get("basis") not in TRUSTED_BASES or not evidence.get("source", "").startswith("https://") or not evidence.get("locator"):
        return None
    if field == "hand" and evidence.get("assertion") != "physical_hand_used":
        return None
    return value


def summarize_conditions(rows):
    # Augmentations and repeated views cannot fill quotas for original people or
    # performances. Preview mirroring alone is not an augmentation of the input.
    originals = [r for r in rows if not r.get("augmentation_of") and not r.get("synthetic", False)]
    hands, directions, views, setups, dominance = Counter(), Counter(), Counter(), Counter(), Counter()
    complete = 0
    complete_hands = set()
    for row in originals:
        hand = documented_value(row, "hand")
        direction = documented_value(row, "lighting_direction")
        view = documented_value(row, "camera_view")
        setup = documented_value(row, "camera_setup")
        dominant = documented_value(row, "signer_dominance")
        if hand in {"left", "right"}:
            hands[hand] += 1
        if direction:
            directions[direction] += 1
        if view:
            views[view] += 1
        if setup:
            setups[setup] += 1
        if dominant:
            dominance[dominant] += 1
        if hand in {"left", "right"} and direction and view:
            complete += 1
            complete_hands.add(hand)
    identities = [r.get("performance_id") or r.get("source_sha256") or r.get("source_file") for r in originals]
    return {"sample_views": len(originals),
            "independent_performances": len({value for value in identities if value}),
            "performance_id_unknown": sum(not value for value in identities),
            "documented_physical_hands": {"left": hands["left"], "right": hands["right"], "unknown": len(originals)-sum(hands.values())},
            "documented_lighting_directions": dict(sorted(directions.items())),
            "documented_camera_views": dict(sorted(views.items())),
            "documented_camera_setups": dict(sorted(setups.items())),
            "documented_signer_dominance": dict(sorted(dominance.items())),
            "hand_lighting_view_together": complete,
            "both_hands_with_joint_annotations": complete_hands == {"left", "right"},
            "warning": "Signer dominance is not the actual hand used. Camera setup is not a per-image angle; normalized brightness is processing, not physical lighting. Missing and detector-estimated fields remain unverified. View counts are not independent performance counts."}
