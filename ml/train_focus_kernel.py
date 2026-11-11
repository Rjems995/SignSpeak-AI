"""Reproduce the focused static kernel candidate without evaluating holdouts.

Only rows marked train/validation are transformed or inferred. Fresh-test and
previously assessed test images remain outside fitting, gates and calibration.
This command writes candidate artifacts and cannot install them.
"""
import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .compare_public_models import predict, probabilities
from .data_quality import ROOT
from .features import LABELS
from .static_features import WORLD_ALIGN_VERSION, static_features
from .train_static_depth import (
    GRID, FOCUS_FLOORS, apply_rescue, compose_features, digest, feature_arrays,
    known_metrics, nearest_distance, representative_indices)
from .weak_letter_data import public_partitions

FIXED_RESCUE = {'E': .01, 'R': .01}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def features_batch(rows):
    base = compose_features(feature_arrays(rows))
    xy = base[:, :42].reshape(-1, 21, 2)
    vx, vy = xy[:, 9, 0], xy[:, 9, 1]
    aligned = np.stack([xy[:, :, 0]*vy[:, None]-xy[:, :, 1]*vx[:, None],
                        xy[:, :, 0]*vx[:, None]+xy[:, :, 1]*vy[:, None]], axis=2)
    return np.concatenate([base, aligned.reshape(-1, 42)], axis=1)


def validated_partitions(data, selection, unknown):
    public, audit = public_partitions()
    people = {split: {r['participant'] for r in rows} for split, rows in public.items()}
    original = {split: {(r['participant'], r['source_sha256'], r['letter']): r
                        for r in rows} for split, rows in public.items()}
    additions = {(r['participant'], r['source_sha256'], r['letter']): r
                 for r in selection['samples'] if r['split'] == 'train'}
    allowed = original['train'].copy()
    allowed.update(additions)
    parts = {split: [r for r in data['samples'] if r['split'] == split]
             for split in ['train', 'validation']}
    if len(parts['validation']) != 1224:
        raise ValueError('Keep the complete frozen 1,224-row validation partition')
    for split, rows in parts.items():
        if not rows or any(r.get('training_eligible') is False for r in rows):
            raise ValueError('Review-only rows cannot enter fitting or calibration')
        if set(r['letter'] for r in rows) != set(LABELS):
            raise ValueError('Every static class must occur in each selected partition')
        if len({r['fingerprint'] for r in rows}) != len(rows):
            raise ValueError('Duplicate selected source fingerprints')
        for r in rows:
            key = (r['participant'], r['source_sha256'], r['letter'])
            reference = (allowed if split == 'train' else original['validation']).get(key)
            if (r['participant'] not in people[split] or reference is None or
                    r['source'] != reference['source'] or r['source_file'] != reference['source_file']):
                raise ValueError('Selected source identity lies outside the frozen eligible partition')
        if split == 'validation' and {r['fingerprint'] for r in rows} != {
                r['fingerprint'] for r in public['validation']}:
            raise ValueError('The original validation fingerprints changed')
    # Source/image identities are checked before any 3D transformation.
    groups = defaultdict(set)
    for r in parts['train']:
        groups[r['source_sha256']].add((r['participant'], r['letter']))
    if any(len(values) > 1 for values in groups.values()):
        raise ValueError('A training image is shared by conflicting labels/participants')
    reviewed = read(ROOT/'datasets/unknown-hands.json')['samples']
    for split in ['train', 'validation']:
        rows = [r for r in unknown['samples'] if r['split'] == split]
        expected = {r['fingerprint']: r for r in reviewed if r['split'] == split}
        if len(rows) != len(expected) or {r['fingerprint'] for r in rows} != set(expected):
            raise ValueError('Unknown data differs from the reviewed frozen selection')
        for r in rows:
            ref = expected[r['fingerprint']]
            if r.get('training_eligible') is False or r['letter'] != '_' or any(
                    r[k] != ref[k] for k in ['participant', 'source', 'source_sha256']):
                raise ValueError('Reviewed negative identity changed')
            if any(r['participant'] in people[other] for other in people.keys()-{split}):
                raise ValueError('Known/negative participant split leakage')
        parts['unknown_'+split] = rows
    combined_train = parts['train']+parts['unknown_train']
    combined_val = parts['validation']+parts['unknown_validation']
    for field in ['participant', 'fingerprint', 'source_sha256']:
        train_values = {r[field] for r in combined_train}
        if train_values & {r[field] for r in combined_val}:
            raise ValueError('Training/validation leakage: '+field)
        if train_values & {r[field] for r in public['test']}:
            raise ValueError('Previously held-out test identity entered training: '+field)
        # Read only holdout identity metadata, never its coordinate features.
        if train_values & {r[field] for r in data['samples'] if r['split'] == 'fresh_test'}:
            raise ValueError('Fresh holdout identity entered training: '+field)
    return parts, audit


def participant_metrics(rows, raw, gated):
    result = known_metrics(np.array([r['letter'] for r in rows]), raw, gated)
    result['by_participant'] = {}
    for person in sorted({r['participant'] for r in rows}):
        ids = np.array([i for i, r in enumerate(rows) if r['participant'] == person])
        result['by_participant'][person] = known_metrics(
            np.array([rows[i]['letter'] for i in ids]), raw[ids], gated[ids])
    result['worst_participant_focus_recall'] = min(
        value['correct_accepted']/value['samples']
        for metrics in result['by_participant'].values()
        for letter, value in metrics['by_letter'].items() if letter in FOCUS_FLOORS)
    return result


def candidate_score(metrics, false_unknown, thresholds):
    focus = [metrics['by_letter'][l] for l in FOCUS_FLOORS]
    precision = [v['correct_accepted']/(v['correct_accepted']+v['false_accepted'])
                 if v['correct_accepted']+v['false_accepted'] else 0 for v in focus]
    eligible = (false_unknown == 0 and metrics['gated_accuracy'] >= .85 and
        all(metrics['by_letter'][l]['correct_accepted'] >= floor for l, floor in FOCUS_FLOORS.items())
        and min(precision) >= .95)
    correct = [v['correct_accepted'] for v in focus]
    score = (metrics['worst_participant_focus_recall'], min(precision), min(correct),
        sum(correct), metrics['gated_accuracy'], min(thresholds.values()), thresholds['E'], thresholds['R'])
    return eligible, score


def train_candidate(dataset, unknown_path, selection_path, baseline_path, output, calibrate=False):
    output = Path(output).resolve()
    if (output.is_relative_to((ROOT/'signspeak/static/models').resolve()) or
            output.is_relative_to((ROOT/'datasets/baselines').resolve())):
        raise ValueError('Choose a candidate output directory; installed/baseline models are protected')
    data, selection, unknown = read(dataset), read(selection_path), read(unknown_path)
    if data.get('selection_manifest_sha256') != digest(selection_path):
        raise ValueError('Focused data does not match its verified source-selection manifest')
    parts, audit = validated_partitions(data, selection, unknown)
    train, val = parts['train'], parts['validation']
    utrain, uval = parts['unknown_train'], parts['unknown_validation']
    x, xv, nt, nv = map(features_batch, [train, val, utrain, uval])
    for rows, features in [(train, x), (val, xv), (utrain, nt), (uval, nv)]:
        for i in sorted({0, len(rows)//2, len(rows)-1}):
            np.testing.assert_allclose(features[i], static_features(
                rows[i]['points'], rows[i]['world_points'], WORLD_ALIGN_VERSION), atol=1e-12, rtol=1e-12)
    y = np.array([r['letter'] for r in train])
    scaler = StandardScaler().fit(x)
    transform = Nystroem(kernel='rbf', gamma=1/315, n_components=640, random_state=17)
    classifier = LogisticRegression(C=1000, max_iter=1500, solver='lbfgs', tol=1e-5)
    print(f'Training kernel on {len(train)} originals; {len(val)} validation rows; holdouts excluded.', flush=True)
    with threadpool_limits(limits=2):
        kernel_features = transform.fit_transform(scaler.transform(x))
        classifier.fit(kernel_features, y)
    labels = classifier.classes_.tolist()
    folded = transform.normalization_.T @ classifier.coef_.T
    ids = representative_indices(train, x)
    vectors, support_y = np.round(x[ids], 6), y[ids]
    support = {l: vectors[support_y == l] for l in labels}
    truth = np.array([r['letter'] for r in val])
    gates = {l: float(max(.12, np.quantile(nearest_distance(xv[truth == l], s), .95)))
             for l, s in support.items()}
    baseline = read(baseline_path)
    if baseline.get('rescue') or len(baseline['mean']) != 42:
        raise ValueError('Use the preserved pre-depth 42-feature static-v2 expert')
    rejector = copy.deepcopy(baseline['rejection'])
    rejector.update(support=np.round(nt, 6).tolist(), ratio=1)
    model = {'feature_version': WORLD_ALIGN_VERSION, 'labels': labels,
        'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
        'kernel_basis': {'centers': transform.components_.tolist(), 'gamma': 1/315},
        'layers': [{'kernel': folded.tolist(), 'bias': classifier.intercept_.tolist()}],
        'support': {l: s.tolist() for l, s in support.items()}, 'distance_thresholds': gates,
        'rejection': rejector}
    np.testing.assert_allclose(probabilities(model, xv),
        classifier.predict_proba(transform.transform(scaler.transform(xv))), atol=1e-10)
    raw, primary = predict(model, xv)
    _, primary_unknown = predict(model, nv)
    prob, uprob = probabilities(model, xv), probabilities(model, nv)
    _, legacy = predict(baseline, xv[:, :42])
    _, ulegacy = predict(baseline, nv[:, :42])
    choices = [dict(E=e, R=r) for e in GRID for r in GRID] if calibrate else [FIXED_RESCUE.copy()]
    trials, winner = [], None
    for thresholds in choices:
        accepted = apply_rescue(primary, prob, legacy, labels, thresholds)
        negative = apply_rescue(primary_unknown, uprob, ulegacy, labels, thresholds)
        metrics = participant_metrics(val, raw, accepted)
        false_unknown = int((negative != '_').sum())
        eligible, score = candidate_score(metrics, false_unknown, thresholds)
        trial = {'min_primary_probabilities': thresholds, 'eligible': eligible,
            'known': metrics, 'unknown': {'tracked': len(uval), 'false_acceptances': false_unknown,
                'false_acceptance_rate_tracked': false_unknown/len(uval)}}
        trials.append(trial)
        if eligible and (winner is None or score > winner[0]):
            winner = (score, trial)
    if winner is None:
        raise ValueError('The validation policy failed; no candidate model exported')
    selected = winner[1]
    model['rescue'] = {'version': 'gated-legacy-agreement-v1', 'labels': ['E', 'R'],
        'min_primary_probabilities': selected['min_primary_probabilities'], 'model': baseline}
    model['train_fingerprints'] = [r['fingerprint'] for r in train]
    model['support_fingerprints'] = [train[i]['fingerprint'] for i in ids]
    output.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(model, separators=(',', ':'), allow_nan=False).encode('utf-8')
    import hashlib
    report = {'version': 2, 'feature_version': WORLD_ALIGN_VERSION,
        'method': 'Validation-only focused kernel reproduction; no final/fresh-test inference.',
        'test_predictions_used_for_selection': False, 'fresh_test_predictions_used_for_selection': False,
        'input_sha256': {'dataset': digest(dataset), 'selection_manifest': digest(selection_path),
            'unknown_depth': digest(unknown_path), 'legacy_baseline': digest(baseline_path)},
        'classifier_sha256': hashlib.sha256(encoded).hexdigest(),
        'counts': {'training_originals': len(train), 'validation_originals': len(val),
            'training_augmentation': 0, 'negative_training': len(utrain), 'negative_validation': len(uval),
            'positive_support': len(ids)},
        'settings': {'scaler': 'StandardScaler fitted on originals only',
            'kernel': {'method': 'Nystroem RBF', 'gamma': 1/315, 'n_components': 640, 'random_state': 17},
            'classifier': {'method': 'LogisticRegression', 'C': 1000, 'max_iter': 1500,
                'solver': 'lbfgs', 'tol': 1e-5, 'iterations': classifier.n_iter_.tolist()},
            'support_per_participant_letter': 8, 'support_decimal_places': 6,
            'distance_gate_quantile': .95, 'distance_gate_minimum': .12,
            'negative_gate_ratio': 1, 'confidence_threshold': .85,
            'Nystroem_normalization_folded_into_output_kernel': True},
        'partition_audit': audit,
        'split_fingerprints': {split: [r['fingerprint'] for r in parts[split]] for split in ['train', 'validation']},
        'negative_split_fingerprints': {split: [r['fingerprint'] for r in parts['unknown_'+split]] for split in ['train', 'validation']},
        'support_fingerprints': model['support_fingerprints'],
        'source_counts': {split: dict(Counter(r['source'] for r in parts[split])) for split in ['train', 'validation']},
        'primary_validation': participant_metrics(val, raw, primary),
        'primary_unknown_false_acceptances': int((primary_unknown != '_').sum()),
        'selected': selected, 'validation_trials': trials,
        'policy': {'fixed_selected_rescue': FIXED_RESCUE, 'grid': GRID, 'calibration_replayed': calibrate,
            'focus_acceptance_floors': FOCUS_FLOORS, 'minimum_focus_precision': .95,
            'minimum_all_letter_gated_accuracy': .85, 'maximum_unknown_false_acceptances': 0,
            'selection': 'Worst-participant focus recall, focus precision, minimum and mean focus recall, then overall accuracy; ties favor higher thresholds.'},
        'limitations': ['The original public test was previously inspected; this command does not re-evaluate it.',
            'Fresh reserved images belong to the same previously assessed test participants, not new signers/sessions.',
            'R remains weak for one validation signer; aggregate recall does not establish uniform robustness.',
            'The frozen legacy E/R expert has unidentified training signers.',
            'MediaPipe image/world depth is estimated; still images do not establish live-webcam reliability.']}
    (output/'classifier.json').write_bytes(encoded)
    (output/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'counts': report['counts'], 'selected': selected,
        'classifier_sha256': report['classifier_sha256'], 'fresh_test_evaluated': False}, indent=2), flush=True)
    return model, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/focus-depth.json')
    parser.add_argument('--unknown', type=Path, default=ROOT/'datasets/unknown-depth.json')
    parser.add_argument('--selection', type=Path, default=ROOT/'datasets/focus-depth-selection.json')
    parser.add_argument('--baseline', type=Path, default=ROOT/'datasets/baselines/static-v2/classifier.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/focus-kernel-reproduced')
    parser.add_argument('--calibrate', action='store_true')
    args = parser.parse_args()
    train_candidate(args.dataset, args.unknown, args.selection, args.baseline, args.output, args.calibrate)


if __name__ == '__main__':
    main()
