import numpy as np

FEATURE_VERSION = "wrist-palm-axis-mirror-xy-v1"
LABELS = ["A", "B", "C", "L", "V", "Y", "_"]


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
    x = (p[:, 0] * -palm[1] + p[:, 1] * palm[0]) / size ** 2
    y = -(p @ palm) / size ** 2
    if x[5] > x[17]:
        x *= -1
    return np.column_stack([x, y]).ravel()

