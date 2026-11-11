"""Extract hash-verified public hand patches without classifier predictions.

Raw pixel outputs stay in ignored artifacts. Training and validation are the
default; a frozen evaluation explicitly requests fresh_test or test later.
"""
import argparse
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from .data_quality import ROOT
from .import_static_depth import SourceImages, crop_image, digest


def extract_patch(pixels, points):
    """Use a landmark-bounded square, black padding, linear resize and gray RGB."""
    xy = np.asarray(points, dtype=float).reshape(21, -1)[:, :2]
    if not np.isfinite(xy).all():
        raise ValueError('Non-finite hand image coordinates')
    height, width = pixels.shape[:2]
    lo = xy.min(axis=0) * [width, height]
    hi = xy.max(axis=0) * [width, height]
    side = max(1, math.ceil(float(np.max(hi-lo)) * 1.4))
    centre = (lo+hi) / 2
    x0 = math.floor(float(centre[0]) - side/2)
    y0 = math.floor(float(centre[1]) - side/2)
    square = np.zeros((side, side, 3), dtype=np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(width, x0+side), min(height, y0+side)
    if sx1 > sx0 and sy1 > sy0:
        square[sy0-y0:sy1-y0, sx0-x0:sx1-x0] = pixels[sy0:sy1, sx0:sx1]
    rgb = cv2.cvtColor(square, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (64, 64), interpolation=cv2.INTER_LINEAR)
    gray = np.rint(resized[:, :, 0]*.299 + resized[:, :, 1]*.587 + resized[:, :, 2]*.114).clip(0, 255).astype(np.uint8)
    return gray, {'x0': x0, 'y0': y0, 'side': side, 'input_width': width, 'input_height': height}


def select_rows(splits):
    positive_path = ROOT / 'datasets/focus-depth.json'
    unknown_path = ROOT / 'datasets/unknown-depth.json'
    positives = json.loads(positive_path.read_text())['samples']
    unknown = json.loads(unknown_path.read_text())['samples']
    rows = [dict(row, kind='positive') for row in positives if row['split'] in splits]
    rows += [dict(row, kind='unknown') for row in unknown if row['split'] in splits]
    if any(row.get('training_eligible') is False for row in rows):
        raise ValueError('Benchmark/review-only records cannot enter texture training/evaluation')
    if len({row['fingerprint'] for row in rows}) != len(rows):
        raise ValueError('Duplicate image geometry in selected texture records')
    for field in ['participant', 'source_sha256', 'fingerprint']:
        groups = {split: {row[field] for row in rows if row['split'] == split} for split in splits}
        # fresh_test and test deliberately contain the same previously assessed
        # people, so source identity isolation applies to train/validation only.
        for a, b in [('train', 'validation'), ('train', 'fresh_test'), ('train', 'test'),
                     ('validation', 'fresh_test'), ('validation', 'test')]:
            if a in groups and b in groups and groups[a] & groups[b]:
                raise ValueError(f'Texture selection leaks {field} across {a}/{b}')
    return rows, {'focus': digest(positive_path), 'unknown': digest(unknown_path)}


def extract(rows, output, dataset_hashes):
    images = SourceImages(rows)
    patches = np.empty((len(rows), 64, 64), dtype=np.uint8)
    metadata = []
    try:
        for index, row in enumerate(rows):
            data = images.read(row)  # Verifies pinned archive and original image bytes.
            decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError('Verified image cannot decode: ' + row['source_file'])
            # Reviewed unknown coordinates refer to this exact prior source crop.
            decoded = crop_image(decoded, row.get('crop_bbox'))
            patches[index], bounds = extract_patch(decoded, row['points'])
            metadata.append({key: row[key] for key in ['fingerprint', 'source_sha256', 'source_file',
                'source', 'split', 'letter', 'participant', 'kind']} | {'crop_bbox': row.get('crop_bbox'), 'patch_bounds': bounds})
            if (index+1) % 1000 == 0:
                print(f'Patches {index+1}/{len(rows)}', flush=True)
        archive_hashes = images.archives
    finally:
        images.close()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, pixels=patches, fingerprints=np.array([row['fingerprint'] for row in rows]),
        source_sha256=np.array([row['source_sha256'] for row in rows]), split=np.array([row['split'] for row in rows]),
        letter=np.array([row['letter'] for row in rows]), kind=np.array([row['kind'] for row in rows]))
    algorithm = {'version': 'hand-square64-gray-v1', 'opencv_version': cv2.__version__,
        'bbox': 'min/max of all21 imageXY landmarks in detector input pixels',
        'side': 'ceil(max(pixel_width,pixel_height)*1.4)', 'origin': 'floor(centre-side/2)',
        'outside_frame': 'black RGB0 padding', 'resize': '64x64 cv2.INTER_LINEAR',
        'grayscale': 'np.rint(.299*R+.587*G+.114*B),uint8; ties to even',
        'order': 'focus positives then reviewed unknowns; original row order within selected splits',
        'augmentation': False, 'unknown_original_crop': 'Apply prior reviewed crop_bbox+25%padding first, matching landmark extraction',
        'source_dataset_sha256': dataset_hashes, 'source_archives': archive_hashes,
        'pixel_artifact_sha256': digest(output), 'counts': dict(Counter(row['split'] for row in rows)), 'samples': metadata}
    output.with_suffix('.json').write_text(json.dumps(algorithm, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return algorithm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', nargs='+', default=['train', 'validation'], choices=['train', 'validation', 'fresh_test', 'test'])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/weak-texture/trainval-patches.npz')
    args = parser.parse_args()
    rows, dataset_hashes = select_rows(list(dict.fromkeys(args.split)))
    if not rows:
        raise ValueError('No source images match the requested partitions')
    print('Selected', len(rows), dict(Counter(row['split'] for row in rows)), flush=True)
    report = extract(rows, args.output, dataset_hashes)
    print('WROTE', args.output, (len(rows), 64, 64), 'uint8', report['pixel_artifact_sha256'], flush=True)


if __name__ == '__main__':
    main()
