"""Real negative regression coverage, split isolation and validation-only choice."""
import hashlib
import json
from pathlib import Path
import numpy as np
from ml.compare_public_models import predict
from ml.features import normalize
from ml.unknown_rejection import frozen_rows

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text())


def test_negative_provenance_geometry_and_participants_never_cross_splits():
    data = read('datasets/unknown-hands.json')
    report = read('datasets/unknown-evaluation.json')
    assert hashlib.sha256((ROOT/'datasets/unknown-hands.json').read_bytes()).hexdigest()==report['unknown_dataset_sha256']
    assert {r['category'] for r in data['samples']}=={'relaxed_hand','partial_flexion','unrelated_gesture'}
    assert all(r['label_review'].startswith('Codex visual review') for r in data['samples'])
    for row in data['samples']:
        np.testing.assert_allclose(normalize(row['points']),row['features'],atol=1e-12)
        assert row['letter']=='_' and len(row['source_sha256'])==64
    for key in ['participant','source_sha256','fingerprint']:
        rows = data['samples'] if key=='fingerprint' else data['samples']+data['tracking_failures']
        parts = [{r[key] for r in rows if r['split']==split} for split in ['train','validation','test']]
        assert not parts[0]&parts[1] and not parts[0]&parts[2] and not parts[1]&parts[2]
    known = read('datasets/prepared.json')+read('datasets/public-balanced.json')
    assert not {r['fingerprint'] for r in known}&{r['fingerprint'] for r in data['samples']}
    parts = report['public_positive_participants']
    for split in parts:
        for other in parts.keys()-{split}:
            assert not set(parts[split])&set(report['unknown_participants'][other])


def test_installed_support_only_uses_unknown_training_and_keeps_class_weights():
    baseline = read('datasets/baselines/static-v1/classifier.json')
    model = read('signspeak/static/models/classifier.json')
    report = read('datasets/unknown-evaluation.json')
    assert {k:v for k,v in model.items() if k!='rejection'}==baseline
    gate = model['rejection']
    assert gate['training_fingerprints']==report['unknown_split_fingerprints']['train']
    train = [r for r in read('datasets/unknown-hands.json')['samples'] if r['split']=='train']
    assert gate['support']==[r['features'] for r in train]
    trials = report['validation_trials']
    eligible = [r for r in trials if r['positive_retention_passed'] and r['unknown_false_acceptances']<report['baseline']['validation']['unknown']['false_acceptances']]
    chosen = min(eligible,key=lambda row:(row['unknown_false_acceptances'],row['ratio']))
    assert chosen['ratio']==gate['ratio']==report['selected_ratio']
    assert all(report['candidate']['validation'][k]['correct_accepted']==report['baseline']['validation'][k]['correct_accepted'] for k in ['legacy_known','public_known'])


def test_frozen_unknown_and_known_tests_match_report_including_tracking_failures():
    model = read('signspeak/static/models/classifier.json')
    report = read('datasets/unknown-evaluation.json')
    data = read('datasets/unknown-hands.json')
    rows = [r for r in data['samples'] if r['split']=='test']
    _, gated = predict(model,np.array([r['features'] for r in rows]))
    result = report['candidate']['test']['unknown']
    assert int((gated!='_').sum())==result['false_acceptances']==2
    assert len(rows)==result['tracked']==48
    assert result['selected']==len(rows)+len([r for r in data['tracking_failures'] if r['split']=='test'])
    assert result['false_acceptance_rate_tracked']<report['baseline']['test']['unknown']['false_acceptance_rate_tracked']
    legacy_report = read('datasets/baselines/static-v1/evaluation.json')
    legacy = frozen_rows(ROOT/'datasets/prepared.json',legacy_report,'test')
    _, gated = predict(model,np.array([r['features'] for r in legacy]))
    truth = np.array([r['letter'] for r in legacy])
    assert int((gated==truth).sum())==report['candidate']['test']['legacy_known']['correct_accepted']==85
    current = read('signspeak/static/models/evaluation.json')
    assert current['gated_accuracy_at_threshold_085']==85/125
    assert current['rejection_benchmark']['false_acceptances']==2
