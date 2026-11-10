"""Describe observed duration/trajectory variation without inventing capture labels."""
import numpy as np


def motion_observations(frames):
    time = np.array([f["t"] for f in frames], dtype=float)
    points = np.array([f["points"] for f in frames]).reshape(-1, 21, 2)
    scale = np.median(np.linalg.norm(points[:, 9]-points[:, 0], axis=1))
    if len(frames) < 2 or not np.isfinite(scale) or scale < 1e-6 or np.any(np.diff(time) <= 0):
        raise ValueError("Cannot measure malformed movement")
    relative = (points-points[0, 0])/scale
    # Absolute fingertip travel includes hand/wrist movement, as does browser
    # inference. These are landmark-derived proxies, not measured signing intent.
    speed = np.linalg.norm(np.diff(relative[:, [8, 20]], axis=0), axis=2).max(axis=1)
    active = np.flatnonzero(speed >= max(.035, speed.max()*.12))
    lo, hi = (max(0, int(active[0])-1), min(len(frames)-1, int(active[-1])+2)) if len(active) else (0, len(frames)-1)
    active_points = relative[lo:hi+1]
    duration = (time[hi]-time[lo])/1000
    extent = max(float(np.linalg.norm(np.ptp(active_points[:, j], axis=0))) for j in [8, 20])
    return {"recorded_duration_ms": float(time[-1]-time[0]), "active_duration_ms": float(duration*1000),
            "trajectory_extent_palm_lengths": extent,
            "travel_speed_palm_lengths_per_second": float(max(np.linalg.norm(np.diff(active_points[:, j], axis=0), axis=1).sum() for j in [8, 20])/duration) if duration else 0,
            "measurement_source": "derived_from_detected_landmarks_not_capture_annotations"}


def ranges(records):
    result = {}
    for key in ["active_duration_ms", "trajectory_extent_palm_lengths", "travel_speed_palm_lengths_per_second"]:
        values = np.array([r["observations"][key] for r in records])
        result[key] = {"min": float(values.min()), "median": float(np.median(values)), "max": float(values.max())} if len(values) else None
    return result
