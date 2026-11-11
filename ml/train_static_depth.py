"""Train the static XYZ/world candidate using training and validation rows only.

This command never installs a model or evaluates the final test partition.
The fixed default rescue thresholds replay the prior validation-only selection;
--calibrate repeats its declared 36-combination validation grid.
"""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .compare_public_models import predict, probabilities
from .data_quality import ROOT
from .features import LABELS
from .static_features import PAIRS, WORLD_VERSION, static_features
from .weak_letter_data import public_partitions

GRID = [1e-5, 1e-4, .001, .01, .05, .1]
FIXED_RESCUE = {'E': .01, 'R': .0001}
FOCUS_FLOORS = {'E': 49, 'M': 28, 'R': 38, 'T': 29}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if isinstance(data, list):
        return data, []
    if not isinstance(data, dict) or not isinstance(data.get('samples'), list):
        raise ValueError(f'{path}: expected a list or an object containing samples')
    return data['samples'], data.get('failures', [])


def feature_arrays(rows):
    """Canonical image XY, relative image Z, and normalized world coordinates."""
    points = np.asarray([row['points'] for row in rows], dtype=float)
    world = np.asarray([row['world_points'] for row in rows], dtype=float)
    if points.shape != (len(rows), 63) or world.shape != (len(rows), 63):
        raise ValueError('Every training/validation row needs 21 image/world XYZ landmarks')
    if not np.isfinite(points).all() or not np.isfinite(world).all():
        raise ValueError('Image and world coordinates must be finite')
    points = points.reshape(-1, 21, 3)
    relative = points - points[:, :1, :]
    xy = relative[:, :, :2].copy()
    size = np.linalg.norm(xy[:, 9, :], axis=1)
    if (size < 1e-6).any():
        raise ValueError('Degenerate image hand geometry')
    cross = xy[:, 5, 0]*xy[:, 17, 1] - xy[:, 5, 1]*xy[:, 17, 0]
    xy[:, :, 0] *= np.where(cross < 0, -1, 1)[:, None]
    xy /= size[:, None, None]
    z = relative[:, :, 2] / size[:, None]
    world = world.reshape(-1, 21, 3)
    world -= world[:, :1, :]
    scale = np.linalg.norm(world[:, 9, :], axis=1)
    if (scale < 1e-6).any():
        raise ValueError('Degenerate world hand geometry')
    world /= scale[:, None, None]
    return xy.reshape(-1, 42), z, world


def compose_features(arrays, rng=None):
    """Match static_features exactly; perturb training geometry only when asked."""
    xy, z, world = arrays
    if rng is not None:
        # Keep the draw order stable to reproduce the selected training run.
        xy = xy + rng.normal(0, .015, xy.shape)
        z = z + rng.normal(0, .015, z.shape)
        world = world + rng.normal(0, .015, world.shape)
    first = np.concatenate([xy, z], axis=1)
    distances = np.linalg.norm(
        world[:, [a for a, _ in PAIRS], :] - world[:, [b for _, b in PAIRS], :], axis=2)
    return np.concatenate([first, distances], axis=1)


def features_batch(rows):
    return compose_features(feature_arrays(rows))


def nearest_distance(features, support):
    """Bound temporary memory while retaining the browser's RMS distance."""
    chunks = np.array_split(features, max(1, (len(features)+63)//64))
    return np.concatenate([
        np.sqrt(((chunk[:, None, :]-support[None, :, :])**2).mean(axis=2)).min(axis=1)
        for chunk in chunks])


def representative_indices(rows, features, per_cell=8):
    """Select real training rows: central first, then farthest from those chosen."""
    selected = []
    for person in sorted({row['participant'] for row in rows}):
        for letter in sorted({row['letter'] for row in rows}):
            ids = np.array([i for i, row in enumerate(rows)
                            if row['participant'] == person and row['letter'] == letter], dtype=int)
            if not len(ids):
                continue
            vectors = features[ids]
            chosen = [int(np.argmin(np.linalg.norm(vectors-np.median(vectors, axis=0), axis=1)))]
            distances = ((vectors-vectors[chosen[0]])**2).mean(axis=1)
            while len(chosen) < min(per_cell, len(ids)):
                distances[chosen] = -1
                index = int(distances.argmax())
                chosen.append(index)
                distances = np.minimum(distances, ((vectors-vectors[index])**2).mean(axis=1))
            selected.extend(ids[chosen].tolist())
    return selected


def selected_partitions(positive, unknown, public):
    """Use frozen metadata to reject relabeled holdout people and review-only data."""
    people = {split: {r['participant'] for r in rows} for split, rows in public.items()}
    expected = {split: {r['fingerprint']: r for r in rows} for split, rows in public.items()}
    parts = {}
    for split in ['train', 'validation']:
        rows = [row for row in positive if row.get('split') == split]
        if not rows or any(row.get('training_eligible') is False for row in rows):
            raise ValueError('Only eligible real training/validation rows may enter this command')
        if {r['fingerprint'] for r in rows} != set(expected[split]):
            raise ValueError(f'{split}: does not match the frozen eligible public selection')
        if len({r['fingerprint'] for r in rows}) != len(rows):
            raise ValueError(f'{split}: duplicate source fingerprints')
        for row in rows:
            reference = expected[split][row['fingerprint']]
            if row['participant'] not in people[split] or any(
                    row[key] != reference[key] for key in ['letter', 'participant', 'source_sha256', 'source']):
                raise ValueError('Public source/participant metadata changed behind a frozen fingerprint')
        if set(r['letter'] for r in rows) != set(LABELS):
            raise ValueError('Every static class is required in training and validation')
        parts[split] = rows
    original = json.loads((ROOT/'datasets/unknown-hands.json').read_text())['samples']
    for split in ['train', 'validation']:
        expected_negative = {r['fingerprint']: r for r in original if r['split'] == split}
        rows = [row for row in unknown if row.get('split') == split]
        if len(rows) != len(expected_negative) or {r['fingerprint'] for r in rows} != set(expected_negative):
            raise ValueError(f'{split}: negative data differs from the reviewed frozen selection')
        for row in rows:
            reference = expected_negative[row['fingerprint']]
            if row.get('training_eligible') is False or row['letter'] != '_' or any(
                    row[key] != reference[key] for key in ['participant', 'source_sha256', 'source']):
                raise ValueError('Negative metadata changed behind a reviewed fingerprint')
            if row['participant'] in people['test']:
                raise ValueError('A final-test participant entered negative training/validation')
            for other in people.keys()-{split}:
                if row['participant'] in people[other]:
                    raise ValueError('A public source participant crosses positive/negative splits')
        parts['unknown_'+split] = rows
    combined_train = parts['train']+parts['unknown_train']
    combined_val = parts['validation']+parts['unknown_validation']
    for key in ['participant', 'fingerprint', 'source_sha256']:
        if {r[key] for r in combined_train} & {r[key] for r in combined_val}:
            raise ValueError(f'Training/validation leakage: {key}')
    return parts


def known_metrics(truth, raw, gated):
    correct = gated == truth
    accepted = gated != '_'
    return {'samples': len(truth), 'raw_accuracy': float((raw == truth).mean()),
            'gated_accuracy': float(correct.mean()), 'correct_accepted': int(correct.sum()),
            'accepted': int(accepted.sum()), 'incorrect_accepted': int((accepted & ~correct).sum()),
            'accuracy_when_accepted': float(correct.sum()/accepted.sum()) if accepted.any() else None,
            'by_letter': {letter: {'samples': int((truth == letter).sum()),
                'raw_correct': int(((raw == truth) & (truth == letter)).sum()),
                'correct_accepted': int((correct & (truth == letter)).sum()),
                'false_accepted': int(((gated == letter) & (truth != letter)).sum())}
                for letter in sorted(set(truth))}}


def apply_rescue(primary, probability, legacy, labels, thresholds):
    result = primary.copy()
    for letter in ['E', 'R']:
        result[(legacy == letter) & (probability[:, labels.index(letter)] >= thresholds[letter])] = letter
    return result


def policy_score(metrics, false_unknown, thresholds):
    focus = [metrics['by_letter'][letter]['correct_accepted'] for letter in FOCUS_FLOORS]
    eligible = (all(metrics['by_letter'][letter]['correct_accepted'] >= floor
                    for letter, floor in FOCUS_FLOORS.items())
                and metrics['gated_accuracy'] >= .85 and false_unknown == 0)
    score = (min(focus), sum(focus), metrics['gated_accuracy'],
             min(thresholds.values()), thresholds['E'], thresholds['R'])
    return eligible, score


def train_candidate(dataset, unknown_path, output, baseline_path, calibrate=False):
    output = Path(output).resolve()
    if (output.is_relative_to((ROOT/'signspeak/static/models').resolve()) or
            output.is_relative_to((ROOT/'datasets/baselines').resolve())):
        raise ValueError('This command only writes candidate artifacts; choose a separate output directory')
    positive, positive_failures = load(dataset)
    negative, negative_failures = load(unknown_path)
    public, audit = public_partitions()
    parts = selected_partitions(positive, negative, public)
    train, val = parts['train'], parts['validation']
    utrain, uval = parts['unknown_train'], parts['unknown_validation']
    arrays = feature_arrays(train)
    x = compose_features(arrays)
    xv, nt, nv = map(features_batch, [val, utrain, uval])
    # The vectorized fit must match the single-record production feature math.
    for rows, matrix in [(train, x), (val, xv), (utrain, nt), (uval, nv)]:
        for index in sorted({0, len(rows)//2, len(rows)-1}):
            np.testing.assert_allclose(matrix[index], static_features(rows[index]['points'], rows[index]['world_points']), atol=1e-12, rtol=1e-12)
    y = np.array([r['letter'] for r in train])
    truth = np.array([r['letter'] for r in val])
    scaler = StandardScaler().fit(x)
    rng = np.random.default_rng(17)
    augmented = np.concatenate([x]+[compose_features(arrays, rng) for _ in range(2)])
    classifier = MLPClassifier(hidden_layer_sizes=(96, 64), solver='lbfgs', alpha=.3,
        max_iter=1200, max_fun=25000, random_state=17)
    print(f'Training {len(train)} real originals; {len(val)} validation rows; final test excluded.', flush=True)
    with threadpool_limits(limits=2):
        classifier.fit(scaler.transform(augmented), np.tile(y, 3))
    labels = classifier.classes_.tolist()
    ids = representative_indices(train, x)
    vectors = np.round(x[ids], 6)
    support_y = y[ids]
    support = {letter: vectors[support_y == letter] for letter in labels}
    gates = {letter: float(max(.12, np.quantile(nearest_distance(xv[truth == letter], support[letter]), .95))) for letter in labels}
    baseline = json.loads(Path(baseline_path).read_text())
    if baseline['feature_version'] == WORLD_VERSION or baseline.get('rescue'):
        raise ValueError('Preserve the pre-depth deployed model as the fixed legacy baseline')
    gate = copy.deepcopy(baseline['rejection'])
    gate.update(support=np.round(nt, 6).tolist(), ratio=1,
        training_fingerprints=[r['fingerprint'] for r in utrain])
    model = {'feature_version': WORLD_VERSION, 'labels': labels,
        'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
        'layers': [{'kernel': kernel.tolist(), 'bias': bias.tolist()}
                   for kernel, bias in zip(classifier.coefs_, classifier.intercepts_)],
        'support': {letter: rows.tolist() for letter, rows in support.items()},
        'distance_thresholds': gates, 'rejection': gate}
    raw, primary = predict(model, xv)
    _, primary_unknown = predict(model, nv)
    primary_probability, unknown_probability = probabilities(model, xv), probabilities(model, nv)
    _, legacy = predict(baseline, xv[:, :42])
    _, legacy_unknown = predict(baseline, nv[:, :42])
    trials, winner = [], None
    candidates = [dict(E=e, R=r) for e in GRID for r in GRID] if calibrate else [FIXED_RESCUE.copy()]
    for thresholds in candidates:
        accepted = apply_rescue(primary, primary_probability, legacy, labels, thresholds)
        accepted_unknown = apply_rescue(primary_unknown, unknown_probability, legacy_unknown, labels, thresholds)
        metrics = known_metrics(truth, raw, accepted)
        false_unknown = int((accepted_unknown != '_').sum())
        eligible, score = policy_score(metrics, false_unknown, thresholds)
        trial = {'min_primary_probabilities': thresholds, 'eligible': eligible,
            'known': metrics, 'unknown': {'tracked': len(uval), 'false_acceptances': false_unknown,
                'false_acceptance_rate_tracked': false_unknown/len(uval)}}
        trials.append(trial)
        if eligible and (winner is None or score > winner[0]):
            winner = (score, trial)
    if winner is None:
        raise ValueError('No validation candidate met the frozen promotion policy; no model exported')
    selected = winner[1]
    model['rescue'] = {'version': 'gated-legacy-agreement-v1', 'labels': ['E', 'R'],
        'min_primary_probabilities': selected['min_primary_probabilities'], 'model': baseline}
    output.mkdir(parents=True, exist_ok=True)
    model_bytes = json.dumps(model, separators=(',', ':'), allow_nan=False).encode('utf-8')
    report = {'version': 1, 'feature_version': WORLD_VERSION,
        'method': 'Training/validation-only static depth candidate; final test not evaluated.',
        'test_predictions_used_for_selection': False,
        'split': 'Primary model public-source participants are disjoint; legacy rescue signer identities are unavailable.',
        'input_sha256': {'dataset': digest(dataset), 'unknown': digest(unknown_path),
            'legacy_baseline': digest(baseline_path), 'public_balanced': audit['dataset_sha256'],
            **audit['source_hashes']},
        'classifier_sha256': hashlib.sha256(model_bytes).hexdigest(),
        'hyperparameters': {'hidden_layers': [96, 64], 'solver': 'lbfgs', 'alpha': .3,
            'max_iter': 1200, 'max_fun': 25000, 'seed': 17, 'augmentation_seed': 17,
            'augmentation_copies_including_original': 3, 'training_noise_stddev': .015,
            'support_per_participant_letter': 8, 'support_decimal_places': 6,
            'distance_gate_quantile': .95, 'distance_gate_minimum': .12,
            'confidence_threshold': .85, 'negative_distance_ratio': 1,
            'iterations': int(classifier.n_iter_)},
        'counts': {'known_training_originals': len(train), 'known_training_with_augmentation': len(augmented),
            'known_validation': len(val), 'negative_training': len(utrain), 'negative_validation': len(uval),
            'positive_support': len(ids)},
        'participant_ids': audit['participant_ids'],
        'split_fingerprints': {split: [r['fingerprint'] for r in parts[split]] for split in ['train', 'validation']},
        'negative_split_fingerprints': {split: [r['fingerprint'] for r in parts['unknown_'+split]] for split in ['train', 'validation']},
        'support_fingerprints': [train[i]['fingerprint'] for i in ids],
        'source_counts': {split: dict(Counter(r['source'] for r in parts[split])) for split in ['train', 'validation']},
        'selected_extraction_failures': {'positive': [r for r in positive_failures if r.get('split') in ['train', 'validation']],
            'negative': [r for r in negative_failures if r.get('split') in ['train', 'validation']]},
        'policy': {'rescue_grid': GRID, 'focused_correct_acceptance_floors': FOCUS_FLOORS,
            'validation_samples_per_letter': 51, 'minimum_all_letter_gated_accuracy': .85,
            'maximum_unknown_validation_false_acceptances': 0,
            'selection': 'Maximize minimum focused recall, then mean focused recall, then all-letter accuracy; ties favor higher minimum primary probability, then E, then R.',
            'calibration_replayed': calibrate, 'fixed_selected_thresholds': FIXED_RESCUE},
        'primary_validation': known_metrics(truth, raw, primary),
        'primary_unknown_validation': {'tracked': len(uval), 'false_acceptances': int((primary_unknown != '_').sum())},
        'selected': selected, 'validation_trials': trials,
        'limitations': ['MediaPipe XYZ/world coordinates estimate depth; they are not measured depth.',
            'Public image participants are source identifiers, not independently reconciled identities.',
            'The legacy rescue retains the older model with unavailable training signer identities.',
            'Per-image actual hand, lighting and camera-angle coverage remains incomplete.',
            'Public still-image validation does not establish live-webcam reliability.']}
    (output/'classifier.json').write_bytes(model_bytes)
    (output/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'counts': report['counts'], 'selected': selected,
        'classifier_sha256': report['classifier_sha256'], 'final_test_evaluated': False}, indent=2), flush=True)
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/static-depth.json')
    parser.add_argument('--unknown', type=Path, default=ROOT/'datasets/unknown-depth.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/static-depth-reproduced')
    parser.add_argument('--baseline', type=Path, default=ROOT/'datasets/baselines/static-v2/classifier.json')
    parser.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()
    train_candidate(args.dataset, args.unknown, args.output, args.baseline, args.calibrate)


if __name__ == '__main__':
    main()
