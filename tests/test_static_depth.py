"""Depth feature math, source isolation, and the sealed weak-letter release."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pytest
from ml.features import normalize
from ml.static_features import static_features, record_features, WORLD_VERSION

ROOT = Path(__file__).resolve().parents[1]


def read(name):
    return json.loads((ROOT/name).read_text())


def test_depth_features_preserve_xy_and_position_scale_mirror_invariance():
    row = read('datasets/static-depth.json')['samples'][0]
    points = np.array(row['points']).reshape(21,3)
    world = np.array(row['world_points']).reshape(21,3)
    expected = static_features(points,world)
    assert expected.shape==(273,) and np.isfinite(expected).all()
    np.testing.assert_allclose(expected[:42],normalize(points),atol=1e-12)
    np.testing.assert_allclose(static_features(points*3+[.2,-.7,.1],world*2+[1,2,3]),expected,atol=1e-10)
    np.testing.assert_allclose(static_features(points*[-1,1,1],world*[-1,1,1]),expected,atol=1e-10)
    assert not np.allclose(static_features(points*[-1,-1,1],world),expected)
    with pytest.raises(ValueError):
        static_features(points[:,:2],world)
    with pytest.raises(ValueError):
        static_features(points,np.zeros((21,3)))


def test_expanded_data_and_representatives_never_cross_frozen_splits():
    from ml.weak_letter_data import public_partitions
    parts, audit = public_partitions()
    assert {k:len(v) for k,v in parts.items()}=={'train':5828,'validation':1224,'test':1224}
    rows = read('datasets/static-depth.json')['samples']
    for key in ['participant','source_sha256','fingerprint']:
        sets = {split:{r[key] for r in rows if r['split']==split} for split in parts}
        assert not sets['train']&sets['validation'] and not sets['train']&sets['test'] and not sets['validation']&sets['test']
    original = {r['fingerprint']:r for v in parts.values() for r in v}
    for row in rows:
        assert row.get('training_eligible') is not False
        assert row['source_sha256']==original[row['fingerprint']]['source_sha256']
        np.testing.assert_array_equal(np.array(row['points']).reshape(21,3)[:,:2].ravel(),original[row['fingerprint']]['points'])


def test_failed_release_keeps_deployed_weights_and_reports_frozen_results():
    from ml.compare_public_models import predict
    model = read('datasets/candidates/focus-kernel-v4/classifier.json')
    report = read('datasets/weak-letter-evaluation.json')
    assert model['feature_version']=='wrist-scale-image-xyz-world-palm-v4'
    assert hashlib.sha256((ROOT/'datasets/candidates/focus-kernel-v4/classifier.json').read_bytes()).hexdigest()==report['classifier_sha256']
    assert not report['release_passed']
    assert read('signspeak/static/models/classifier.json')==read('datasets/baselines/static-v2/classifier.json')
    assert {r['criterion'] for r in report['release_failures']}=={'overall_gated_accuracy','R_correct_accepted'}
    assert len(model['support_fingerprints'])==1520
    train = {r['fingerprint'] for r in read('datasets/focus-depth.json')['samples'] if r['split']=='train'}
    assert set(model['support_fingerprints']).issubset(train)
    assert model['rescue']['model']==read('datasets/baselines/static-v2/classifier.json')
    assert model['rescue']['min_primary_probabilities']=={'E':.01,'R':.01}
    examples = report['unknown_test_examples']
    _, predictions = predict(model,np.array([r['features'] for r in examples]))
    assert [None if p=='_' else p for p in predictions]==[r['accepted'] for r in examples]
    assert int((predictions!='_').sum())<=report['release_policy']['maximum_unknown_test_false_acceptances']
    for label in 'EMT':
        assert report['candidate']['test']['known']['by_letter'][label]['gated_accuracy']>report['old_public_candidate']['test']['known']['by_letter'][label]['gated_accuracy']
    assert report['candidate']['test']['known']['by_letter']['R']['gated_accuracy']<report['old_public_candidate']['test']['known']['by_letter']['R']['gated_accuracy']


@pytest.fixture
def sealed_cohort(tmp_path):
    from ml.evaluate_static_depth import cohort_manifest
    positive = {'samples':[{'split':split,'fingerprint':f'known-{split}-{i}','points':[i]}
        for split in ['train','validation','fresh_test'] for i in range(2)]}
    negative = {'samples':[{'split':split,'fingerprint':f'unknown-{split}-{i}','points':[i]}
        for split in ['train','validation','test'] for i in range(2)],
        'tracking_failures':[{'split':split,'source':'example','source_file':split+'.jpg',
            'source_sha256':split,'participant':split,'category':'relaxed_hand','letter':'_'}
            for split in ['train','validation','test']]}
    paths = [tmp_path/'known.json',tmp_path/'unknown.json']
    for path,data in zip(paths,[positive,negative]):
        path.write_text(json.dumps(data),encoding='utf-8')
    policy = {'evaluation_cohort':cohort_manifest(*paths,'fresh_test')}
    return paths,policy


def test_cohort_guard_accepts_complete_seal_without_inference(sealed_cohort,monkeypatch):
    import ml.evaluate_static_depth as evaluator
    paths,policy = sealed_cohort
    def inference_forbidden(*args,**kwargs):
        pytest.fail('Cohort validation must precede every model inference')
    monkeypatch.setattr(evaluator,'predict',inference_forbidden)
    monkeypatch.setattr(evaluator,'probabilities',inference_forbidden)
    evaluator.validate_evaluation_cohort(policy,*paths,'fresh_test')
    with pytest.raises(ValueError,match='no supported frozen'):
        evaluator.validate_evaluation_cohort({},*paths,'fresh_test')
    with pytest.raises(ValueError,match='Requested test split'):
        evaluator.validate_evaluation_cohort(policy,*paths,'test')


@pytest.mark.parametrize('file_index',[0,1])
def test_cohort_guard_rejects_changed_coordinate_bytes(sealed_cohort,file_index):
    from ml.evaluate_static_depth import validate_evaluation_cohort
    paths,policy = sealed_cohort
    data = json.loads(paths[file_index].read_text())
    data['samples'][0]['points'] = [999]
    paths[file_index].write_text(json.dumps(data),encoding='utf-8')
    with pytest.raises(ValueError,match='dataset bytes'):
        validate_evaluation_cohort(policy,*paths,'fresh_test')


@pytest.mark.parametrize('change',['drop_known','replace_known','move_unknown','drop_failure','duplicate_known'])
def test_cohort_guard_rejects_changed_membership_even_if_byte_pin_is_refreshed(sealed_cohort,change):
    from ml.evaluate_static_depth import sha,validate_evaluation_cohort
    paths,policy = sealed_cohort
    file_index = 1 if change in {'move_unknown','drop_failure'} else 0
    data = json.loads(paths[file_index].read_text())
    if change=='drop_known':
        data['samples'].pop()
    elif change=='replace_known':
        data['samples'][-1]['fingerprint'] = 'cherry-picked-easy-image'
    elif change=='move_unknown':
        data['samples'][-1]['split'] = 'validation'
    elif change=='drop_failure':
        data['tracking_failures'].pop()
    else:
        data['samples'].append(data['samples'][-1].copy())
    paths[file_index].write_text(json.dumps(data),encoding='utf-8')
    key = 'unknown_dataset_sha256' if file_index else 'positive_dataset_sha256'
    policy['evaluation_cohort'][key] = sha(paths[file_index])
    with pytest.raises(ValueError,match='identities/counts|Duplicate evaluation'):
        validate_evaluation_cohort(policy,*paths,'fresh_test')


def test_frozen_policy_pins_complete_original_focus_and_unknown_cohorts():
    from ml.evaluate_static_depth import validate_evaluation_cohort
    policy = read('datasets/weak-letter-release-policy.json')
    validate_evaluation_cohort(policy,ROOT/'datasets/focus-depth.json',ROOT/'datasets/unknown-depth.json','fresh_test')
    cohort = policy['evaluation_cohort']
    assert {key:value['count'] for key,value in cohort['known'].items()}=={'train':8980,'validation':1224,'test':840}
    assert {key:value['count'] for key,value in cohort['unknown'].items()}=={'train':76,'validation':27,'test':48}
    assert {key:value['count'] for key,value in cohort['unknown_tracking_failures'].items()}=={'train':6,'validation':2,'test':3}
