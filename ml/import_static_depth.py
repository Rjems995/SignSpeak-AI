"""Re-extract real image/world XYZ from verified public static hand images.

The default opens only the frozen expanded training and validation participants.
Test extraction is an explicit later step, after model selection is frozen.
MediaPipe depth is estimated geometry, not a measured physical-depth annotation.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

from .data_quality import ROOT
from .download_aslyset import SHA256 as ASLYSET_SHA256, SIZE as ASLYSET_SIZE
from .weak_letter_data import public_partitions

CACHE_ROOT = ROOT / 'artifacts/weak-depth/image-cache'
EXTRACTOR_SCHEMA = 'mediapipe-image-world-xyz-v1'
MAX_IMAGE_BYTES = 20_000_000


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def regular_member(name):
    """Reject absolute, alternate-shell, parent and non-image archive paths."""
    member = PurePosixPath(name)
    if (not isinstance(name, str) or '\\' in name or ':' in name or member.is_absolute()
            or '..' in member.parts or member.suffix.lower() not in {'.jpg', '.png'}):
        raise ValueError('Unsafe source image member')
    return member


def regular_cached_file(root, name):
    regular_member(name)
    root = root.resolve()
    path = root / name
    if not path.resolve().is_relative_to(root) or path.is_symlink() or not path.is_file():
        raise ValueError(f'Expected a regular image inside {root}: {name}')
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise ValueError('Unexpected cached image size')
    return path


class SourceImages:
    """Verify archives once, then verify each original image before decoding."""
    def __init__(self, rows):
        self.zipped = None
        self.archives = {}
        sources = {row['source'] for row in rows}
        try:
            if 'ASLYset' in sources:
                directory = ROOT / 'datasets/public/aslyset'
                archive = directory / 'ASLYset.rar'
                if archive.stat().st_size != ASLYSET_SIZE or digest(archive) != ASLYSET_SHA256:
                    raise ValueError('ASLYset archive hash/size mismatch; run ml.download_aslyset')
                self.archives['ASLYset'] = {'sha256': ASLYSET_SHA256, 'bytes': archive.stat().st_size}
                members = sorted({row['source_file'] for row in rows if row['source'] == 'ASLYset'})
                pattern = re.compile(r'ASLYset/images/(User[1-4])/([a-z]+)/\1_([A-Z]+)_([1-9][0-9]*)\.png')
                for name in members:
                    regular_member(name)
                    match = pattern.fullmatch(name)
                    if not match or match[2] != match[3].lower():
                        raise ValueError('Unexpected ASLYset source filename')
                    if not (directory / name).resolve().is_relative_to(directory.resolve()):
                        raise ValueError('ASLYset extraction target leaves the verified cache')
                # Inspect even cached members: originals must be regular archive entries.
                listing = subprocess.check_output(['tar', '-tvf', str(archive)], text=True).splitlines()
                regular = {line[line.index('ASLYset/'):] for line in listing if line.startswith('-') and 'ASLYset/' in line}
                if not set(members).issubset(regular):
                    raise ValueError('ASLYset selected members must be regular archive images')
                missing = [name for name in members if not (directory / name).exists()]
                if missing:
                    CACHE_ROOT.mkdir(parents=True, exist_ok=True)
                    key = hashlib.sha256('\n'.join(missing).encode()).hexdigest()
                    member_list = CACHE_ROOT / f'aslyset-members-{key}.txt'
                    member_list.write_text('\n'.join(missing) + '\n', encoding='utf-8')
                    subprocess.run(['tar', '--safe-writes', '-xf', str(archive), '-C', str(directory), '-T', str(member_list)], check=True)
            if 'ASL-HG' in sources:
                source = json.loads((ROOT / 'datasets/public-sources.json').read_text())['asl_hg']
                archive = ROOT / 'datasets/public/asl_hg' / source['archive']
                if archive.stat().st_size != source['bytes'] or digest(archive) != source['sha256']:
                    raise ValueError('ASL-HG archive hash/size mismatch; run scripts/download-public-source.mjs')
                self.archives['ASL-HG'] = {'sha256': source['sha256'], 'bytes': archive.stat().st_size}
                self.zipped = zipfile.ZipFile(archive)
                if len(self.zipped.namelist()) != len(set(self.zipped.namelist())):
                    raise ValueError('Duplicate ZIP paths are not accepted')
            self.hagrid_lock = (json.loads((ROOT / 'datasets/unknown-source-lock.json').read_text())
                                if 'HaGRIDv2' in sources else None)
        except Exception:
            self.close()
            raise

    def close(self):
        if self.zipped is not None:
            self.zipped.close()

    def read(self, row):
        name = row['source_file']
        regular_member(name)
        if row['source'] == 'ASLYset':
            data = regular_cached_file(ROOT / 'datasets/public/aslyset', name).read_bytes()
        elif row['source'] == 'ASL-HG':
            if not re.fullmatch(r'asl_dataset/[A-Z]/P(?:10|[1-9])_[A-Z]_[0-9]+\.jpg', name):
                raise ValueError('Unexpected ASL-HG source filename')
            info = self.zipped.getinfo(name)
            kind = stat.S_IFMT(info.external_attr >> 16)
            if info.is_dir() or kind not in {0, stat.S_IFREG} or info.file_size > MAX_IMAGE_BYTES:
                raise ValueError('ASL-HG selected member must be a bounded regular image')
            data = self.zipped.read(info)
        elif row['source'] == 'HaGRIDv2':
            image_id = PurePosixPath(name).stem
            if not re.fullmatch(r'[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}', image_id):
                raise ValueError('Unexpected reviewed HaGRID image identifier')
            if self.hagrid_lock['images'].get(image_id) != row['source_sha256']:
                raise ValueError('HaGRID image is not in the pinned reviewed source set')
            data = regular_cached_file(ROOT / 'datasets/public/hagrid', image_id + '.jpg').read_bytes()
        else:
            raise ValueError(f'Unsupported static depth source: {row["source"]}')
        if hashlib.sha256(data).hexdigest() != row['source_sha256']:
            raise ValueError(f'Original source image hash mismatch: {name}')
        return data


@contextlib.contextmanager
def native_log(path):
    """Suppress native C++ telemetry while restoring visible Python failures."""
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = os.dup(2)
    with path.open('w', encoding='utf-8') as log:
        try:
            os.dup2(log.fileno(), 2)
            yield
        finally:
            sys.stderr.flush()
            os.dup2(saved, 2)
            os.close(saved)


def valid_geometry(row):
    return all(isinstance(row.get(key), list) and len(row[key]) == 63
               and all(isinstance(value, (int, float)) and math.isfinite(value) for value in row[key])
               for key in ['points', 'world_points'])


def existing_geometry(extractor, output):
    """Reuse prior local extraction only when its complete detector config matches."""
    entries = {}
    paths = {ROOT / 'artifacts/weak-depth/aslyset-depth.json', ROOT / 'artifacts/weak-depth/asl_hg-depth.json', output}
    for path in paths:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        matches = data.get('extractor') == extractor
        # The initial real extraction predates this module's cache schema but
        # records all of the same detector/model settings independently.
        if not matches:
            fields = ['mediapipe_version', 'mode', 'num_hands', 'detection_confidence', 'presence_confidence', 'task_sha256']
            matches = all(data.get(key) == extractor[key] for key in fields)
        if not matches:
            continue
        for row in data.get('samples', []):
            if not valid_geometry(row):
                continue
            spec = {'source_sha256': row['source_sha256'], 'crop_bbox': row.get('crop_bbox'), 'extractor': extractor}
            key = hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            entries[key] = {'cache_spec': spec, 'points': row['points'], 'world_points': row['world_points'],
                            'image_width': row.get('image_width'), 'image_height': row.get('image_height')}
    return entries


def crop_image(pixels, bbox):
    if bbox is None:
        return pixels
    if (not isinstance(bbox, list) or len(bbox) != 4 or
            not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox)):
        raise ValueError('Invalid reviewed crop bounding box')
    x, y, w, h = bbox
    if min(x, y) < 0 or min(w, h) <= 0 or x + w > 1.000001 or y + h > 1.000001:
        raise ValueError('Reviewed crop lies outside the source image')
    height, width = pixels.shape[:2]
    margin = .25 * max(w, h)
    cropped = pixels[max(0, int((y-margin)*height)):min(height, int((y+h+margin)*height)),
                     max(0, int((x-margin)*width)):min(width, int((x+w+margin)*width))].copy()
    if not cropped.size:
        raise ValueError('Empty reviewed source crop')
    return cropped


def select_rows(splits, unknown=False):
    if unknown:
        path = ROOT / 'datasets/unknown-hands.json'
        data = json.loads(path.read_text())
        rows = [row for row in data['samples'] if row['split'] in splits]
        audit = {'unknown_dataset_sha256': digest(path),
                 'source_tracking_failures': [row for row in data['tracking_failures'] if row['split'] in splits]}
    else:
        parts, audit = public_partitions()
        rows = [dict(row, split=split) for split in splits for row in parts[split]]
    if any(row.get('training_eligible') is False for row in rows):
        raise ValueError('Benchmark/review-only rows cannot enter depth training or evaluation')
    if len({row['fingerprint'] for row in rows}) != len(rows):
        raise ValueError('Duplicate selected source fingerprints')
    for field in ['participant', 'source_sha256', 'fingerprint']:
        for a, b in [('train', 'validation'), ('train', 'test'), ('validation', 'test')]:
            if {r[field] for r in rows if r['split'] == a} & {r[field] for r in rows if r['split'] == b}:
                raise ValueError(f'Depth selection leaks {field} across partitions')
    return rows, audit


def extract(rows, output, selection, log_path):
    samples, failures, cache_hits = [], [], 0
    with native_log(log_path):
        import cv2
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        task = ROOT / 'datasets/videos/hand_landmarker.task'
        extractor = {'schema': EXTRACTOR_SCHEMA, 'mediapipe_version': mp.__version__, 'mode': 'IMAGE',
                     'num_hands': 2, 'detection_confidence': .6, 'presence_confidence': .6, 'task_sha256': digest(task)}
        options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(task)),
            running_mode=vision.RunningMode.IMAGE, num_hands=2,
            min_hand_detection_confidence=.6, min_hand_presence_confidence=.6)
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        prior_geometry = existing_geometry(extractor, output)
        sources = SourceImages(rows)
        try:
            with vision.HandLandmarker.create_from_options(options) as detector:
                for index, row in enumerate(rows):
                    data = sources.read(row)  # Verify current originals even on a geometry-cache hit.
                    bbox = row.get('crop_bbox')
                    cache_spec = {'source_sha256': row['source_sha256'], 'crop_bbox': bbox, 'extractor': extractor}
                    cache_key = hashlib.sha256(json.dumps(cache_spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
                    cache_path = CACHE_ROOT / (cache_key + '.json')
                    try:
                        cached = json.loads(cache_path.read_text()) if cache_path.exists() else prior_geometry.get(cache_key)
                    except (ValueError, OSError):
                        cached = None  # An interrupted cache write can be re-extracted safely.
                    if cached and cached.get('cache_spec') == cache_spec and (cached.get('reason') or valid_geometry(cached)):
                        geometry = cached
                        if not geometry.get('reason') and not (geometry.get('image_width') and geometry.get('image_height')):
                            pixels = crop_image(cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR), bbox)
                            geometry.update(image_width=int(pixels.shape[1]), image_height=int(pixels.shape[0]))
                        if not cache_path.exists():
                            cache_path.write_text(json.dumps(geometry, separators=(',', ':'), allow_nan=False), encoding='utf-8')
                        cache_hits += 1
                    else:
                        pixels = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                        geometry = {'cache_spec': cache_spec}
                        if pixels is None:
                            geometry['reason'] = 'decode_failed'
                        else:
                            pixels = crop_image(pixels, bbox)
                            geometry.update(image_width=int(pixels.shape[1]), image_height=int(pixels.shape[0]))
                            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(pixels, cv2.COLOR_BGR2RGB)))
                            if len(result.hand_landmarks) != 1:
                                geometry['reason'] = 'requires_one_detected_hand'
                            else:
                                geometry['points'] = [v for point in result.hand_landmarks[0] for v in (point.x, point.y, point.z)]
                                geometry['world_points'] = ([v for point in result.hand_world_landmarks[0] for v in (point.x, point.y, point.z)]
                                                            if result.hand_world_landmarks else None)
                                if not valid_geometry(geometry):
                                    geometry.pop('points', None)
                                    geometry.pop('world_points', None)
                                    geometry['reason'] = 'invalid_3d_geometry'
                        cache_path.write_text(json.dumps(geometry, separators=(',', ':'), allow_nan=False), encoding='utf-8')
                    metadata = {key: value for key, value in row.items() if key not in {'points', 'features', 'world_points'}}
                    if geometry.get('reason'):
                        failures.append(dict(metadata, reason=geometry['reason']))
                    else:
                        # Do not silently change the XY geometry behind a frozen fingerprint.
                        xy = np.array(geometry['points']).reshape(21, 3)[:, :2].ravel()
                        difference = float(np.max(np.abs(xy - np.array(row['points']))))
                        if difference > 1e-12:
                            failures.append(dict(metadata, reason='original_xy_changed', xy_max_absolute_difference=difference))
                        else:
                            samples.append(dict(metadata, points=geometry['points'], world_points=geometry['world_points'],
                                                image_width=geometry['image_width'], image_height=geometry['image_height']))
                    if (index + 1) % 400 == 0:
                        print(f'Depth: {index+1}/{len(rows)}; kept {len(samples)}; failures {len(failures)}; cache hits {cache_hits}', flush=True)
            archive_audit = sources.archives
        finally:
            sources.close()
    result = {**extractor, 'extractor': extractor, 'selection': selection, 'selected': len(rows),
              'samples': samples, 'failures': failures, 'archives': archive_audit, 'cache_hits': cache_hits,
              'xy_preserved_tolerance': 1e-12, 'by_split': dict(Counter(row['split'] for row in samples)),
              'failure_counts': dict(Counter(row['reason'] for row in failures)),
              'warning': 'Image z and world coordinates are MediaPipe estimates, not measured depth or new physical-hand annotations.'}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(',', ':'), allow_nan=False), encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', nargs='+', choices=['train', 'validation', 'test'], default=['train', 'validation'])
    parser.add_argument('--output', type=Path, default=ROOT / 'artifacts/weak-depth/selected-depth.json')
    parser.add_argument('--unknown', action='store_true', help='Extract existing reviewed unknown rows instead of known letters.')
    parser.add_argument('--source', nargs='+', choices=['ASLYset', 'ASL-HG', 'HaGRIDv2'], help='Optionally select disjoint source subsets for parallel extraction.')
    args = parser.parse_args()
    splits = list(dict.fromkeys(args.split))
    log_name = hashlib.sha256(str(args.output.resolve()).encode()).hexdigest()[:12]
    log_path = ROOT / 'artifacts/weak-depth' / f'import-static-depth-{log_name}.log'
    print(f'Native MediaPipe log: {log_path}', flush=True)
    try:
        rows, selection = select_rows(splits, unknown=args.unknown)
        if args.source:
            rows = [row for row in rows if row['source'] in args.source]
        selection = dict(selection, requested_splits=splits, requested_sources=args.source)
        if not rows:
            raise ValueError('No eligible source images match the selected partitions')
        print(f'Selected {len(rows)} real images: {dict(Counter(row["split"] for row in rows))}', flush=True)
        report = extract(rows, args.output, selection, log_path)
    except Exception:
        print(f'Static depth import failed. Native diagnostic log: {log_path}', file=sys.stderr)
        raise
    print(json.dumps({'output': str(args.output), 'samples': len(report['samples']), 'failures': len(report['failures']),
                      'cache_hits': report['cache_hits'], 'by_split': report['by_split'], 'extractor': report['extractor']}, indent=2))


if __name__ == '__main__':
    main()
