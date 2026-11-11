"""Crossing geometry, invariance and safe logistic export math."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from ml.features import normalize
from ml.topology_features import TOPOLOGY_VERSION, topology_features, topology_probabilities

ROOT = Path(__file__).resolve().parents[1]


def training_rows():
    data = json.loads((ROOT/'datasets/focus-depth.json').read_text())
    return [r for r in data['samples'] if r['split'] == 'train']


def test_crossing_feature_tracks_tip_order_relative_to_knuckles():
    points = np.zeros((21, 2))
    points[9] = [0, 1]
    points[5] = [-.2, .8]
    points[8] = [.1, 1.6]
    points[12] = [-.1, 1.7]
    crossed = topology_features(points.reshape(1, 42))[0]
    points[[8, 12]] = points[[12, 8]]
    uncrossed = topology_features(points.reshape(1, 42))[0]
    assert crossed.shape == (12,)
    assert crossed[0] == pytest.approx(-.04)
    assert uncrossed[0] == pytest.approx(.04)
    assert crossed[10] == pytest.approx(uncrossed[10])


def test_topology_uses_canonical_translation_scale_and_mirror_invariance():
    points = np.array(training_rows()[0]['points']).reshape(21, 3)
    expected = topology_features([normalize(points)])
    np.testing.assert_allclose(topology_features([normalize(points*2+[.3, -.2, 1])]), expected, atol=1e-12)
    np.testing.assert_allclose(topology_features([normalize(points*[-1, 1, 1])]), expected, atol=1e-12)
    with pytest.raises(ValueError):
        topology_features(np.zeros((1, 63)))
    with pytest.raises(ValueError):
        topology_features(np.full((1, 42), np.nan))


def test_exported_logistic_scores_match_sklearn_on_real_training_geometry():
    rows = training_rows()
    # Fit a small, deterministic selection of real training R/U images only.
    chosen = [next(r for r in rows if r['participant'] == person and r['letter'] == letter)
        for person in sorted({r['participant'] for r in rows}) for letter in ['R', 'U']]
    xy = np.array([normalize(r['points']) for r in chosen])
    x = topology_features(xy)
    scaler = StandardScaler().fit(x)
    classifier = LogisticRegression(class_weight='balanced').fit(scaler.transform(x), [r['letter'] == 'R' for r in chosen])
    expert = {'version': TOPOLOGY_VERSION, 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
        'weights': classifier.coef_[0].tolist(), 'bias': float(classifier.intercept_[0]),
        'threshold': .99, 'allowed_primary_labels': ['R', 'U', 'V', 'D']}
    np.testing.assert_allclose(topology_probabilities(expert, xy), classifier.predict_proba(scaler.transform(x))[:, 1], atol=1e-12)
    for field, value in [('version', 'unknown'), ('scale', [0]*12), ('weights', [1]*11),
                         ('bias', float('inf')), ('threshold', .5), ('threshold', float('nan')),
                         ('allowed_primary_labels', ['R', 'U', 'V', 'E'])]:
        broken = copy.deepcopy(expert)
        broken[field] = value
        with pytest.raises(ValueError):
            topology_probabilities(broken, xy)
    extreme = copy.deepcopy(expert)
    extreme['weights'] = [0]*12
    for bias, expected in [(1e6, 1), (-1e6, 0)]:
        extreme['bias'] = bias
        with np.errstate(over='raise'):
            np.testing.assert_array_equal(topology_probabilities(extreme, xy), np.full(len(xy), expected))


def test_frozen_topology_candidate_retains_release_guard_and_unknown_rejection():
    from ml.compare_public_models import predict
    def read(name):
        return json.loads((ROOT/name).read_text())
    def digest(name):
        return hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
    candidate_path = 'datasets/candidates/focus-topology-v5/classifier.json'
    validation_path = 'datasets/weak-letter-topology-validation.json'
    candidate = read(candidate_path)
    report = read('datasets/weak-letter-topology-evaluation.json')
    policy = read('datasets/weak-letter-topology-release-policy.json')
    validation = read(validation_path)
    assert digest(candidate_path) == policy['candidate_classifier_sha256'] == report['classifier_sha256']
    assert digest(validation_path) == policy['validation_choice_sha256'] == report['validation_choice_sha256']
    assert report['release_policy'] == policy
    assert not report['release_passed']
    assert report['release_failures'] == [{'criterion': 'overall_gated_accuracy',
        'actual': 657/840, 'minimum': .8}]
    assert policy['minimum_gated_test_accuracy'] == .8
    assert 'not an independent holdout' in policy['test_use']
    for filename in ['classifier.json', 'evaluation.json', 'metadata.json', 'references.json']:
        assert (ROOT/'signspeak/static/models'/filename).read_bytes() == (ROOT/'datasets/baselines/static-v2'/filename).read_bytes()
    expert = candidate.pop('topology_expert')
    assert candidate == read('datasets/candidates/focus-kernel-v4/classifier.json')
    train_ids = {r['fingerprint'] for r in training_rows()}
    assert len(train_ids) == 8980
    assert set(candidate['train_fingerprints']) == train_ids
    assert set(candidate['support_fingerprints']) <= train_ids
    selected = validation['selected']
    assert selected['C'] == 1 and selected['threshold'] == expert['threshold'] == .99
    assert selected['eligible'] and selected['known']['R_correct'] == 45
    assert selected['known']['new_wrong_R'] == selected['unknown_false_acceptances'] == 0
    assert selected['known']['EMT_correct'] == validation['baseline']['EMT_correct']
    assert not validation['test_predictions_used'] and not validation['fresh_test_predictions_used']
    vectors = validation['test_vectors']
    xy = np.array([v['xy'] for v in vectors])
    np.testing.assert_allclose(topology_features(xy), [v['features'] for v in vectors], atol=1e-12)
    np.testing.assert_allclose(topology_probabilities(expert, xy), [v['probability'] for v in vectors], atol=1e-12)
    known = report['candidate']['test']['known']
    assert {label: known['by_letter'][label]['correct_accepted'] for label in 'EMRT'} == {'E': 60, 'M': 39, 'R': 39, 'T': 30}
    assert known['gated_accuracy'] == 657/840 and report['r_precision'] == 1
    unknown = report['unknown_test_examples']
    candidate['topology_expert'] = expert
    _, gated = predict(candidate, np.array([r['features'] for r in unknown]))
    assert len(gated) == report['candidate']['test']['unknown']['tracked'] == 48
    assert [None if label == '_' else label for label in gated] == [r['accepted'] for r in unknown]
    assert np.all(gated == '_')
