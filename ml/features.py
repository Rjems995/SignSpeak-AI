import numpy as np

FEATURE_VERSION = "wrist-scale-orientation-xy-v2"
LABELS = list("ABCDEFGHIKLMNOPQRSTUVWXY")


def normalize(points):
    """42 features. Matches static/js/recognition.js, including mirror handling."""
    p = np.asarray(points, dtype=float).reshape(21, -1)[:, :2].copy()
    if not np.isfinite(p).all():
        raise ValueError("Landmarks must be finite")
    p -= p[0]
    palm = p[9]
    size = np.linalg.norm(palm)
    if size < 1e-6:
        raise ValueError("Degenerate hand geometry")
    # Preserve image-plane direction: rotating H/U, K/P, or G/Q into the same
    # palm-axis frame erases meaningful differences between alphabet letters.
    cross = p[5, 0] * p[17, 1] - p[5, 1] * p[17, 0]
    if cross < 0:
        p[:, 0] *= -1
    return (p / size).ravel()

