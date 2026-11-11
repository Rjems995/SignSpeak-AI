"""Deterministic 64px grayscale HOG used for reviewed texture experiments.

No photographs are stored here. Patches are uint8 grayscale pixels produced by
the documented source-verified square crop; the descriptor is ordinary HOG.
"""
import numpy as np

TEXTURE_VERSION = 'hog-r-v1'
PATCH_SIZE = 64
FEATURE_COUNT = 1764


def hog_batch(patches, batch_size=256):
    pixels = np.asarray(patches)
    if pixels.ndim != 3 or pixels.shape[1:] != (PATCH_SIZE, PATCH_SIZE):
        raise ValueError('Expected grayscale patches shaped N x 64 x 64')
    if not np.isfinite(pixels).all() or ((pixels < 0) | (pixels > 255)).any():
        raise ValueError('Grayscale pixels must be finite values from 0 to 255')
    if batch_size < 1:
        raise ValueError('batch_size must be positive')
    if not len(pixels):
        return np.empty((0, FEATURE_COUNT), dtype=float)
    cell = ((np.arange(64)[:, None]//8)*8+(np.arange(64)[None, :]//8)).ravel()
    output = []
    for start in range(0, len(pixels), batch_size):
        gray = pixels[start:start+batch_size].astype(float)/255
        dx, dy = np.zeros_like(gray), np.zeros_like(gray)
        dx[:, 1:-1, 1:-1] = gray[:, 1:-1, 2:]-gray[:, 1:-1, :-2]
        dy[:, 1:-1, 1:-1] = gray[:, 2:, 1:-1]-gray[:, :-2, 1:-1]
        magnitude = np.hypot(dx, dy).reshape(-1, 4096)
        position = (np.arctan2(dy, dx) % np.pi).reshape(-1, 4096)*9/np.pi
        lower = np.floor(position).astype(int)
        fraction = position-lower
        lower %= 9
        upper = (lower+1) % 9
        offset = np.arange(len(gray))[:, None]*576+cell[None, :]*9
        histogram = (np.bincount((offset+lower).ravel(),
            weights=(magnitude*(1-fraction)).ravel(), minlength=len(gray)*576) +
            np.bincount((offset+upper).ravel(), weights=(magnitude*fraction).ravel(),
                minlength=len(gray)*576)).reshape(-1, 8, 8, 9)
        blocks = np.concatenate([histogram[:, :7, :7, :], histogram[:, :7, 1:, :],
            histogram[:, 1:, :7, :], histogram[:, 1:, 1:, :]], axis=3)
        blocks /= np.sqrt((blocks*blocks).sum(axis=3, keepdims=True)+1e-10)
        blocks = np.minimum(blocks, .2)
        blocks /= np.sqrt((blocks*blocks).sum(axis=3, keepdims=True)+1e-10)
        output.append(blocks.reshape(-1, FEATURE_COUNT))
    return np.concatenate(output)


def hog_features(patch):
    pixels = np.asarray(patch)
    if pixels.size != PATCH_SIZE*PATCH_SIZE:
        raise ValueError('Expected one 64 x 64 grayscale patch')
    return hog_batch(pixels.reshape(1, PATCH_SIZE, PATCH_SIZE))[0]


def texture_probabilities(expert, features):
    if expert.get('version') != TEXTURE_VERSION:
        raise ValueError('Unknown texture expert version')
    x = np.asarray(features, dtype=float)
    if x.ndim != 2 or x.shape[1] != FEATURE_COUNT or not np.isfinite(x).all():
        raise ValueError('Expected N x 1764 finite HOG features')
    mean, scale = np.asarray(expert['mean']), np.asarray(expert['scale'])
    weights = np.asarray(expert['weights'])
    if (mean.shape != (FEATURE_COUNT,) or scale.shape != (FEATURE_COUNT,) or
            not np.isfinite(mean).all() or
            not np.isfinite(scale).all() or not np.isfinite(weights).all() or (scale <= 0).any() or
            not np.isfinite(expert['bias'])):
        raise ValueError('Invalid texture expert parameters')
    normalized = (x-mean)/scale
    if expert.get('kernel_basis'):
        basis = expert['kernel_basis']
        centers = np.asarray(basis['centers'], dtype=float)
        gamma = basis['gamma']
        if (centers.ndim != 2 or centers.shape[1] != FEATURE_COUNT or
                not np.isfinite(centers).all() or not np.isfinite(gamma) or gamma <= 0):
            raise ValueError('Invalid texture kernel basis')
        squared = ((normalized*normalized).sum(axis=1)[:, None]+
            (centers*centers).sum(axis=1)[None, :]-2*normalized@centers.T)
        normalized = np.exp(-gamma*np.maximum(squared, 0))
    if weights.shape != (normalized.shape[1],):
        raise ValueError('Texture output weights do not match their input features')
    logits = normalized @ weights+expert['bias']
    # Stable sigmoid without overflow for strongly rejected patches.
    result = np.empty_like(logits)
    positive = logits >= 0
    result[positive] = 1/(1+np.exp(-logits[positive]))
    exp = np.exp(logits[~positive])
    result[~positive] = exp/(1+exp)
    return result
