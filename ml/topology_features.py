"""Small, explicit crossing and finger-extension features from canonical XY.

Inputs are batches of the first 42 already-normalized static features. Mirroring,
translation and scale handling stay with ``ml.features.normalize``; do not pass
raw image coordinates here.
"""
import numpy as np

TOPOLOGY_VERSION = 'finger-crossing-r-v1'
FEATURE_NAMES = [
    'tip_mcp_order_product', 'tip_aligned_x_difference',
    'pip_aligned_x_difference', 'mcp_aligned_x_difference',
    'index_extension', 'middle_extension', 'ring_extension', 'pinky_extension',
    'thumb_index_mcp_distance', 'thumb_middle_mcp_distance',
    'index_middle_tip_distance', 'tip_aligned_y_difference',
]


def topology_features(xy):
    """Return 12 descriptors per hand without interpreting its letter."""
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 42 or not np.isfinite(xy).all():
        raise ValueError('A finite N x 42 canonical XY batch is required')
    p = xy.reshape(-1, 21, 2)
    vx, vy = p[:, 9, 0], p[:, 9, 1]
    ax = p[:, :, 0]*vy[:, None]-p[:, :, 1]*vx[:, None]
    ay = p[:, :, 0]*vx[:, None]+p[:, :, 1]*vy[:, None]
    def distance(a, b):
        return np.linalg.norm(p[:, a]-p[:, b], axis=1)
    def radius(a):
        return np.linalg.norm(p[:, a], axis=1)
    return np.column_stack([
        (ax[:, 8]-ax[:, 12])*(ax[:, 5]-ax[:, 9]),
        ax[:, 8]-ax[:, 12], ax[:, 6]-ax[:, 10], ax[:, 5]-ax[:, 9],
        *[radius(a)-radius(b) for a, b in [(8, 6), (12, 10), (16, 14), (20, 18)]],
        distance(4, 5), distance(4, 9), distance(8, 12), ay[:, 8]-ay[:, 12],
    ])


def topology_probabilities(expert, xy):
    """Binary R score from the exported logistic model, with stable sigmoid."""
    if expert.get('version') != TOPOLOGY_VERSION:
        raise ValueError('Unknown topology expert version')
    threshold = expert.get('threshold')
    if (not isinstance(threshold, (int, float)) or not np.isfinite(threshold)
            or not .85 <= threshold <= 1
            or expert.get('allowed_primary_labels') != ['R', 'U', 'V', 'D']):
        raise ValueError('Invalid topology threshold or allowed primary labels')
    vectors = [np.asarray(expert.get(key), dtype=float) for key in ['mean', 'scale', 'weights']]
    if any(v.shape != (12,) or not np.isfinite(v).all() for v in vectors):
        raise ValueError('The topology expert needs 12 finite coefficients per vector')
    mean, scale, weights = vectors
    if np.any(scale <= 0):
        raise ValueError('Topology scales must be positive')
    bias = expert.get('bias')
    if not isinstance(bias, (int, float)) or not np.isfinite(bias):
        raise ValueError('A finite topology bias is required')
    score = ((topology_features(xy)-mean)/scale)@weights+bias
    probability = np.empty_like(score)
    positive = score >= 0
    probability[positive] = 1/(1+np.exp(-score[positive]))
    exp = np.exp(score[~positive])
    probability[~positive] = exp/(1+exp)
    return probability
