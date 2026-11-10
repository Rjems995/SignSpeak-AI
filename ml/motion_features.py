"""Temporal features shared with the browser; actual movement is mandatory."""
import numpy as np

MOTION_FEATURE_VERSION = "temporal-palm-scale-16x5xy-v1"
STEPS = 16
JOINTS = [0, 4, 8, 12, 20]


def motion_features(frames):
    if len(frames) < 8:
        raise ValueError("At least eight tracked frames are needed")
    time = np.array([r["t"] for r in frames], dtype=float)
    points = np.array([r["points"] for r in frames], dtype=float).reshape(-1, 21, 2)
    if not np.isfinite(points).all() or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("Motion needs finite coordinates and increasing timestamps")
    if time[-1]-time[0] < 250 or np.diff(time).max() > 500:
        raise ValueError("Keep one hand in view throughout the movement")
    scale = np.median(np.linalg.norm(points[:, 9]-points[:, 0], axis=1))
    if scale < 1e-6:
        raise ValueError("Hand is too small")
    centered = (points-points[0, 0])/scale
    palm = points[:, [5, 17]]-points[:, 0, None, :]
    cross = np.median(palm[:, 0, 0]*palm[:, 1, 1]-palm[:, 0, 1]*palm[:, 1, 0])
    if cross < 0:
        centered[:, :, 0] *= -1
    movement = max(np.linalg.norm(np.ptp(centered[:, joint], axis=0)) for joint in [8, 20])
    if movement < 0.45:
        raise ValueError("Trace a clear movement; a stationary hand is not J or Z")
    speed = np.linalg.norm(np.diff(centered[:, [8, 20]], axis=0), axis=2).max(axis=1)
    active = np.flatnonzero(speed >= max(0.035, speed.max()*0.12))
    lo, hi = (max(0, int(active[0])-1), min(len(frames)-1, int(active[-1])+2)) if len(active) else (0, len(frames)-1)
    time, centered = time[lo:hi+1], centered[lo:hi+1]
    centered -= centered[0, 0].copy()
    grid = np.linspace(time[0], time[-1], STEPS)
    selected = centered[:, JOINTS, :].reshape(len(time), -1)
    return np.stack([np.interp(grid, time, selected[:, i]) for i in range(selected.shape[1])], axis=1).ravel()
