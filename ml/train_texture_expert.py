"""Reproduce bounded R texture trials; export an expert only after validation gains.

The image patches are local, verified originals. This tool uses training and
validation patches only, never installs a model, and requires strictly greater
R recall with no added wrong-R acceptances before exporting candidate.json.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .compare_public_models import predict
from .data_quality import ROOT
from .texture_features import TEXTURE_VERSION, hog_batch, texture_probabilities
from .train_focus_kernel import features_batch
from .train_static_depth import digest

THRESHOLDS = [.85, .9, .95, .99]
ALLOWED = ['R', 'U', 'V', 'D']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def validation_metrics(rows, prediction, baseline):
    truth = np.array([r['letter'] for r in rows])
    correct, positive, accepted = prediction == truth, truth == 'R', prediction != '_'
    result = {'samples': len(rows), 'gated_accuracy': float(correct.mean()),
        'correct_accepted': int(correct.sum()), 'incorrect_accepted': int((accepted & ~correct).sum()),
        'R_correct': int((correct & positive).sum()),
        'R_false_acceptances': int(((prediction == 'R') & ~positive).sum()),
        'new_wrong_R': int(((prediction == 'R') & ~positive & (baseline != 'R')).sum()),
        'EMT_correct': {l: int((correct & (truth == l)).sum()) for l in ['E', 'M', 'T']},
        'by_participant': {}}
    for person in sorted({r['participant'] for r in rows}):
        ids = np.array([i for i, r in enumerate(rows) if r['participant'] == person])
        r_truth = truth[ids] == 'R'
        result['by_participant'][person] = {'R_samples': int(r_truth.sum()),
            'R_correct': int((correct[ids] & r_truth).sum()),
            'R_false_acceptances': int(((prediction[ids] == 'R') & ~r_truth).sum())}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--patches', type=Path, default=ROOT/'artifacts/weak-texture/trainval-patches.npz')
    parser.add_argument('--metadata', type=Path, default=ROOT/'artifacts/weak-texture/trainval-patches.json')
    parser.add_argument('--dataset', type=Path, default=ROOT/'datasets/focus-depth.json')
    parser.add_argument('--unknown', type=Path, default=ROOT/'datasets/unknown-depth.json')
    parser.add_argument('--primary', type=Path, default=ROOT/'artifacts/focus-kernel-reproduced/classifier.json')
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/texture-expert-reproduced')
    args = parser.parse_args()
    output = args.output.resolve()
    if (output.is_relative_to((ROOT/'signspeak/static/models').resolve()) or
            output.is_relative_to((ROOT/'datasets/baselines').resolve())):
        raise ValueError('This command only writes research candidates, never installed models')
    if (output/'candidate.json').exists():
        raise ValueError('Choose a fresh output directory so a previous candidate cannot survive a failed validation run')
    patches = np.load(args.patches, allow_pickle=False)
    metadata = read(args.metadata)
    rows = metadata['samples']
    if len(rows) != len(patches['pixels']) or set(patches['split'])-{'train', 'validation'}:
        raise ValueError('Only correctly matched train/validation patches are permitted')
    for key in ['fingerprints', 'source_sha256', 'split', 'letter', 'kind']:
        row_key = 'fingerprint' if key == 'fingerprints' else key
        if not np.array_equal(patches[key], np.array([r[row_key] for r in rows])):
            raise ValueError('Patch metadata order differs: '+key)
    expected_hashes = metadata['source_dataset_sha256']
    if expected_hashes['focus'] != digest(args.dataset) or expected_hashes['unknown'] != digest(args.unknown):
        raise ValueError('Patch source dataset hashes changed; reconstruct verified patches')
    depth = [r for path in [args.dataset, args.unknown]
             for r in read(path)['samples'] if r['split'] in ['train', 'validation']]
    mapping = {r['fingerprint']: r for r in depth}
    for row in rows:
        source = mapping.get(row['fingerprint'])
        if (source is None or source.get('training_eligible') is False or
                any(row[key] != source[key] for key in ['source_sha256', 'letter', 'participant', 'split'])):
            raise ValueError('Patch does not identify an eligible source row')
    train, val = patches['split'] == 'train', patches['split'] == 'validation'
    for field in ['participant', 'fingerprint', 'source_sha256']:
        if ({r[field] for r, selected in zip(rows, train) if selected} &
                {r[field] for r, selected in zip(rows, val) if selected}):
            raise ValueError('Texture training/validation leakage: '+field)
    y = (patches['letter'] == 'R').astype(int)
    if not set(y[train]) == {0, 1}:
        raise ValueError('Collect both R and non-R training patches')
    descriptors = hog_batch(patches['pixels'])
    scaler = StandardScaler().fit(descriptors[train])
    x, xv = scaler.transform(descriptors[train]), scaler.transform(descriptors[val])
    valrows = [r for r, selected in zip(rows, val) if selected]
    selected_depth = [mapping[r['fingerprint']] for r in valrows]
    model = read(args.primary)
    if model.get('texture_expert'):
        raise ValueError('Compare against the fixed primary model without a texture expert')
    raw, baseline = predict(model, features_batch(selected_depth))
    eligible_shape = np.isin(raw, ALLOWED) & np.isin(baseline, ['_']+ALLOWED)
    known = np.array([r['letter'] != '_' for r in valrows])
    known_rows = [r for r, selected in zip(valrows, known) if selected]
    baseline_known = baseline[known]
    baseline_metrics = validation_metrics(known_rows, baseline_known, baseline_known)
    output.mkdir(parents=True, exist_ok=True)
    fits, winner = [], None
    settings = [('linear', .1), ('linear', 1), ('linear', 10), ('rbf256', 100)]
    for method, C in settings:
        transform = None
        classifier = LogisticRegression(C=C, class_weight='balanced', solver='lbfgs', max_iter=1500, tol=1e-5)
        with threadpool_limits(limits=2):
            if method == 'rbf256':
                transform = Nystroem(kernel='rbf', gamma=1/1764, n_components=256, random_state=17)
                inputs = transform.fit_transform(x)
            else:
                inputs = x
            classifier.fit(inputs, y[train])
        expert = {'version': TEXTURE_VERSION, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
            'weights': (transform.normalization_.T@classifier.coef_[0] if transform else classifier.coef_[0]).tolist(),
            'bias': float(classifier.intercept_[0]), 'patch_size': 64, 'padding': .2,
            'allowed_primary_labels': ALLOWED}
        if transform:
            expert['kernel_basis'] = {'centers': transform.components_.tolist(), 'gamma': 1/1764}
        probability = texture_probabilities(expert, descriptors[val])
        expected = classifier.predict_proba(transform.transform(xv) if transform else xv)[:, list(classifier.classes_).index(1)]
        np.testing.assert_allclose(probability, expected, atol=1e-10)
        trials = []
        for threshold in THRESHOLDS:
            prediction = baseline.copy()
            prediction[eligible_shape & (probability >= threshold)] = 'R'
            metrics = validation_metrics(known_rows, prediction[known], baseline_known)
            false_unknown = int((prediction[~known] != '_').sum())
            eligible = (metrics['R_correct'] > baseline_metrics['R_correct'] and
                metrics['new_wrong_R'] == 0 and false_unknown == 0 and
                metrics['EMT_correct'] == baseline_metrics['EMT_correct'])
            trial = {'threshold': threshold, 'eligible': eligible, 'known': metrics,
                'unknown_false_acceptances': false_unknown, 'overrides': int((prediction != baseline).sum())}
            trials.append(trial)
            score = (metrics['R_correct'], metrics['gated_accuracy'], threshold)
            if eligible and (winner is None or score > winner[0]):
                winner = (score, copy.deepcopy(expert), threshold, method, C, trial)
        filename = f'{method}-C{C}'.replace('.', 'p')+'.json'
        (output/filename).write_text(json.dumps(expert, separators=(',', ':'), allow_nan=False), encoding='utf-8')
        fits.append({'method': method, 'C': C, 'iterations': classifier.n_iter_.tolist(), 'trials': trials})
    report = {'version': 1, 'method': 'Four bounded binary-R texture trials using train/validation only.',
        'test_predictions_used': False, 'fresh_test_predictions_used': False,
        'input_sha256': {'patches': digest(args.patches), 'metadata': digest(args.metadata),
            'dataset': digest(args.dataset), 'unknown': digest(args.unknown), 'primary': digest(args.primary)},
        'feature_spec': {'version': TEXTURE_VERSION, 'features': 1764, 'patch_size': 64,
            'gradients': 'central differences on interior62x62; outer border zero',
            'orientation': 'unsigned modulo pi; 9 angular-interpolated bins', 'cell_size': 8,
            'block_cells': 2, 'stride_cells': 1, 'block_order': 'TL/TR/BL/BR then9bins',
            'normalization': 'sqrt(sum²+1e-10), clip.2, renormalize with sameepsilon'},
        'counts': {'train': int(train.sum()), 'validation': int(val.sum()), 'R_training': int(y[train].sum())},
        'baseline': baseline_metrics, 'baseline_unknown_false_acceptances': int((baseline[~known] != '_').sum()),
        'policy': {'thresholds': THRESHOLDS, 'require_strict_R_recall_gain': True,
            'maximum_new_wrong_R': 0, 'maximum_unknown_false_acceptances': 0,
            'retain_known_EMT_correct_acceptances': True, 'override_raw_labels': ALLOWED,
            'override_accepted_labels': ALLOWED+['_']},
        'fits': fits, 'candidate_exported': winner is not None,
        'limitations': ['Hand texture and landmarks can shift across people, cameras and viewpoints.',
            'A threshold with no recall gain is not an improvement and cannot qualify.',
            'No raw photographs are bundled; patches remain in ignored local storage.']}
    if winner:
        expert = winner[1]
        expert['threshold'] = winner[2]
        (output/'candidate.json').write_text(json.dumps(expert, separators=(',', ':'), allow_nan=False), encoding='utf-8')
        report['selected'] = {'method': winner[3], 'C': winner[4], **winner[5]}
    (output/'validation.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'candidate_exported': report['candidate_exported'], 'counts': report['counts'],
        'baseline': baseline_metrics, 'selected': report.get('selected')}, indent=2), flush=True)
    return report


if __name__ == '__main__':
    main()
