"""Image/world geometry shared with the browser, with XY orientation retained."""
import numpy as np
from .features import FEATURE_VERSION, normalize

DEPTH_VERSION = 'wrist-scale-image-xyz-v3'
WORLD_VERSION = 'wrist-scale-image-xyz-world-distances-v3'
ALIGN_VERSION = 'wrist-scale-image-xyz-palm-xy-v4'
WORLD_ALIGN_VERSION = 'wrist-scale-image-xyz-world-palm-v4'
PAIRS = [(a,b) for a in range(21) for b in range(a+1,21)]


def static_features(points, world_points=None, version=WORLD_VERSION):
    xy = normalize(points)
    if version==FEATURE_VERSION:
        return xy
    if version not in {DEPTH_VERSION,WORLD_VERSION,ALIGN_VERSION,WORLD_ALIGN_VERSION}:
        raise ValueError('Unknown static feature version')
    p = np.asarray(points,dtype=float)
    if p.size!=63 or not np.isfinite(p).all():
        raise ValueError('21 finite image XYZ landmarks are required')
    p = p.reshape(21,3)
    palm = np.linalg.norm(p[9,:2]-p[0,:2])
    features = np.concatenate([xy,(p[:,2]-p[0,2])/palm])
    if version==ALIGN_VERSION:
        plane = xy.reshape(21,2)
        vx,vy = plane[9]
        aligned = np.column_stack([plane[:,0]*vy-plane[:,1]*vx,plane[:,0]*vx+plane[:,1]*vy])
        return np.concatenate([features,aligned.ravel()])
    if version in {WORLD_VERSION,WORLD_ALIGN_VERSION}:
        world = np.asarray(world_points,dtype=float)
        if world.size!=63 or not np.isfinite(world).all():
            raise ValueError('21 finite world XYZ landmarks are required')
        world = world.reshape(21,3).copy()
        world -= world[0].copy()
        scale = np.linalg.norm(world[9])
        if scale<1e-6:
            raise ValueError('Degenerate world hand geometry')
        world /= scale
        features = np.concatenate([features,[np.linalg.norm(world[a]-world[b]) for a,b in PAIRS]])
    if version==WORLD_ALIGN_VERSION:
        plane = xy.reshape(21,2)
        vx,vy = plane[9]
        features = np.concatenate([features,np.column_stack([plane[:,0]*vy-plane[:,1]*vx,plane[:,0]*vx+plane[:,1]*vy]).ravel()])
    return features


def record_features(row, model):
    return static_features(row['points'],row.get('world_points'),model['feature_version'])
