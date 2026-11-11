"""Reproduce three bounded R topology fits using train/validation only.

Only a validation winner with greater R recall, no added wrong R labels, no
unknown acceptance, and retained E/M/T counts is exported. This command cannot
install a model and never transforms or predicts either known test partition.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .compare_public_models import predict
from .data_quality import ROOT
from .features import normalize
from .topology_features import FEATURE_NAMES, TOPOLOGY_VERSION, topology_features, topology_probabilities
from .train_focus_kernel import features_batch, validated_partitions
from .train_static_depth import digest
from .train_texture_expert import validation_metrics

THRESHOLDS = [.85, .9, .95, .99]
ALLOWED = ['R', 'U', 'V', 'D']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def fingerprint_digest(rows):
    return hashlib.sha256('\n'.join(r['fingerprint'] for r in rows).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/focus-depth.json')
    parser.add_argument('--unknown', type=Path, default=ROOT/'datasets/unknown-depth.json')
    parser.add_argument('--selection', type=Path, default=ROOT/'datasets/focus-depth-selection.json')
    parser.add_argument('--primary', type=Path, default=ROOT/'datasets/candidates/focus-kernel-v4/classifier.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/topology-expert-reproduced')
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.is_relative_to((ROOT/'signspeak/static/models').resolve()) or
            output.is_relative_to((ROOT/'datasets/baselines').resolve())):
        raise ValueError('This command only writes research candidates, never installed models')
    if any((output/name).exists() for name in ['expert.json', 'classifier.json', 'validation.json']):
        raise ValueError('Choose a fresh output directory; stale candidates cannot survive failed validation')
    inputs = {name: digest(path) for name, path in [
        ('dataset', args.dataset), ('unknown', args.unknown),
        ('selection', args.selection), ('primary', args.primary)]}
    policy = read(ROOT/'datasets/weak-letter-release-policy.json')
    if inputs['primary'] != policy['candidate_classifier_sha256']:
        raise ValueError('Use the sealed primary model without refitting its weights')
    parts, audit = validated_partitions(read(args.dataset), read(args.selection), read(args.unknown))
    primary = read(args.primary)
    if primary.get('topology_expert') or primary.get('texture_expert'):
        raise ValueError('Use a primary model without a supplemental expert')
    if set(primary['train_fingerprints']) != {r['fingerprint'] for r in parts['train']}:
        raise ValueError('Primary training identities differ from the validated dataset')
    train = parts['train']+parts['unknown_train']
    val = parts['validation']+parts['unknown_validation']
    train_xy = np.array([normalize(r['points']) for r in train])
    val_xy = np.array([normalize(r['points']) for r in val])
    x, xv = topology_features(train_xy), topology_features(val_xy)
    scaler = StandardScaler().fit(x)
    y = np.array([r['letter'] == 'R' for r in train])
    known = np.array([r['letter'] != '_' for r in val])
    raw, baseline = predict(primary, features_batch(val))
    eligible_shape = np.isin(raw, ALLOWED) & np.isin(baseline, ALLOWED+['_'])
    known_rows = [r for r, selected in zip(val, known) if selected]
    baseline_known = baseline[known]
    base = validation_metrics(known_rows, baseline_known, baseline_known)
    fits, winner = [], None
    for C in [.1, 1, 10]:
        classifier = LogisticRegression(C=C, class_weight='balanced', max_iter=1500, tol=1e-7)
        with threadpool_limits(limits=2):
            classifier.fit(scaler.transform(x), y)
        expert = {'version': TOPOLOGY_VERSION, 'mean': scaler.mean_.tolist(),
            'scale': scaler.scale_.tolist(), 'weights': classifier.coef_[0].tolist(),
            'bias': float(classifier.intercept_[0]), 'allowed_primary_labels': ALLOWED,
            'threshold': .85}
        probability = topology_probabilities(expert, val_xy)
        expected = classifier.predict_proba(scaler.transform(xv))[:, list(classifier.classes_).index(True)]
        np.testing.assert_allclose(probability, expected, atol=1e-12)
        trials = []
        for threshold in THRESHOLDS:
            prediction = baseline.copy()
            prediction[eligible_shape & (probability >= threshold)] = 'R'
            metrics = validation_metrics(known_rows, prediction[known], baseline_known)
            false_unknown = int((prediction[~known] != '_').sum())
            eligible = (metrics['R_correct'] > base['R_correct'] and metrics['new_wrong_R'] == 0
                and false_unknown == 0 and metrics['EMT_correct'] == base['EMT_correct'])
            trial = {'threshold': threshold, 'eligible': eligible, 'known': metrics,
                'unknown_false_acceptances': false_unknown, 'overrides': int((prediction != baseline).sum())}
            trials.append(trial)
            score = (metrics['R_correct'], metrics['gated_accuracy'], threshold)
            if eligible and (winner is None or score > winner[0]):
                winner = (score, copy.deepcopy(expert), threshold, C, trial)
        fits.append({'C': C, 'iterations': classifier.n_iter_.tolist(), 'trials': trials})
    report = {'version': 1, 'method': 'Three bounded binary-R topology fits using train/validation only.',
        'test_predictions_used': False, 'fresh_test_predictions_used': False,
        'input_sha256': inputs, 'source_audit': audit,
        'feature_spec': {'version': TOPOLOGY_VERSION, 'features': 12,
            'input': 'First 42 canonical normalized image XY values', 'ordered_names': FEATURE_NAMES},
        'counts': {'train': len(train), 'validation': len(val), 'R_training': int(y.sum())},
        'split_fingerprint_sha256': {key: fingerprint_digest(parts[key]) for key in parts},
        'baseline': base, 'baseline_unknown_false_acceptances': int((baseline[~known] != '_').sum()),
        'policy': {'C': [.1, 1, 10], 'thresholds': THRESHOLDS, 'require_strict_R_recall_gain': True,
            'maximum_new_wrong_R': 0, 'maximum_unknown_false_acceptances': 0,
            'retain_known_EMT_correct_acceptances': True, 'override_raw_labels': ALLOWED,
            'override_accepted_labels': ALLOWED+['_'],
            'selection_order': ['R_correct descending', 'gated_accuracy descending', 'threshold descending']},
        'fits': fits, 'candidate_exported': winner is not None,
        'limitations': ['A binary R score does not independently verify an ASL sign.',
            'Still-image geometry and validation gains do not establish live or new-signer accuracy.',
            'The primary weights and E/R legacy rescue remain fixed.']}
    output.mkdir(parents=True, exist_ok=True)
    if winner:
        expert = winner[1]
        expert['threshold'] = winner[2]
        candidate = copy.deepcopy(primary)
        candidate['topology_expert'] = expert
        vector_ids = [next(i for i, row in enumerate(val) if row['letter'] == label)
            for label in ['R', 'U', 'V', 'D', 'E', '_']]
        selected_probability = topology_probabilities(expert, val_xy)
        report['test_vectors'] = [{'fingerprint': val[i]['fingerprint'], 'letter': val[i]['letter'],
            'xy': val_xy[i].tolist(), 'features': xv[i].tolist(),
            'probability': float(selected_probability[i])} for i in vector_ids]
        (output/'expert.json').write_text(json.dumps(expert, separators=(',', ':'), allow_nan=False), encoding='utf-8')
        (output/'classifier.json').write_text(json.dumps(candidate, separators=(',', ':'), allow_nan=False), encoding='utf-8')
        report['selected'] = {'C': winner[3], **winner[4]}
    (output/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'candidate_exported': report['candidate_exported'], 'counts': report['counts'],
        'baseline': base, 'selected': report.get('selected')}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    main()
