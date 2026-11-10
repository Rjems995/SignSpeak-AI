"""Calibrate a non-letter support veto on validation, then evaluate frozen tests."""
import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from .data_quality import ROOT
from .compare_public_models import predict
from .train import split_indices
from sklearn.metrics import confusion_matrix

MODEL = ROOT / 'signspeak/static/models'
BASELINE = ROOT / 'datasets/baselines/static-v1'


def read(path):
    return json.loads(path.read_text())


def frozen_rows(dataset, evaluation, split):
    rows = read(dataset)
    if hashlib.sha256(dataset.read_bytes()).hexdigest()!=evaluation['dataset_sha256']:
        raise ValueError('Frozen positive dataset hash mismatch')
    mapping = {r['fingerprint']:r for r in rows}
    return [mapping[key] for key in evaluation['split_fingerprints'][split]]


def labels(model, rows):
    return predict(model, np.array([r['features'] for r in rows]))[1]


def positive_metrics(model, rows):
    prediction = labels(model, rows)
    truth = np.array([r['letter'] for r in rows])
    return {'samples':len(rows),'correct_accepted':int((prediction==truth).sum()),
            'gated_accuracy':float((prediction==truth).mean()),'rejected':int((prediction=='_').sum()),
            'incorrect_accepted':int(((prediction!='_')&(prediction!=truth)).sum()),
            'by_letter':{letter:{'samples':int((truth==letter).sum()),'correct_accepted':int(((prediction==truth)&(truth==letter)).sum())} for letter in sorted(set(truth))}}


def unknown_metrics(model, rows, failures):
    prediction = labels(model, rows)
    def score(ids, lost):
        count = len(ids)+lost
        accepted = int((prediction[ids]!='_').sum())
        return {'selected':count,'tracked':len(ids),'tracking_failures':lost,'false_acceptances':accepted,
                'false_acceptance_rate_tracked':accepted/len(ids) if len(ids) else None,
                'false_acceptance_rate_end_to_end':accepted/count if count else None}
    result = score(list(range(len(rows))),len(failures))
    for key in ['category','source']:
        result['by_'+key] = {value:score([i for i,r in enumerate(rows) if r[key]==value],sum(r[key]==value for r in failures))
                                for value in sorted({r[key] for r in rows+failures})}
    return result


def retention_ok(base, candidate, legacy, public, policy):
    for rows, minimum in [(legacy,policy['minimum_legacy_correct_acceptance_retention']),
                          (public,policy['minimum_public_correct_acceptance_retention'])]:
        truth = np.array([r['letter'] for r in rows])
        before, after = labels(base,rows), labels(candidate,rows)
        was_correct = before==truth
        if not was_correct.any() or (after[was_correct]==truth[was_correct]).mean()+1e-12<minimum:
            return False
        if rows is public:
            for letter in set(truth):
                ids = was_correct & (truth==letter)
                if ids.any() and (after[ids]==truth[ids]).mean()+1e-12<policy['minimum_public_per_letter_correct_acceptance_retention']:
                    return False
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--install',action='store_true')
    args = parser.parse_args()
    BASELINE.mkdir(parents=True,exist_ok=True)
    for file in ['classifier.json','evaluation.json','metadata.json']:
        path = BASELINE/file
        if not path.exists():
            if read(MODEL/'classifier.json').get('rejection'):
                raise ValueError('Preserve the original static model before installing a rejector')
            path.write_bytes((MODEL/file).read_bytes())
    base = read(BASELINE/'classifier.json')
    legacy_report = read(BASELINE/'evaluation.json')
    # Recreate the already frozen source/signer split from the tracked dataset;
    # do not depend on ignored training artifacts being present on a new clone.
    public_path = ROOT/'datasets/public-balanced.json'
    public_rows = read(public_path)
    public_sha = hashlib.sha256(public_path.read_bytes()).hexdigest()
    comparison = read(ROOT/'datasets/public-candidate-evaluation.json')
    if public_sha != comparison['dataset_sha256']:
        raise ValueError('Public positive dataset differs from the frozen comparison')
    train,val,test,split_name = split_indices(public_rows,np.array([r['letter'] for r in public_rows]))
    public_report = {'dataset_sha256':public_sha,'split_fingerprints':{},'participant_ids':{}}
    for split, ids in [('train',train),('validation',val),('test',test)]:
        public_report['split_fingerprints'][split] = [public_rows[i]['fingerprint'] for i in ids]
        public_report['participant_ids'][split] = sorted({public_rows[i]['participant'] for i in ids})
    if split_name!='participant-disjoint' or public_report['participant_ids']['test']!=comparison['test_participants']:
        raise ValueError('Cannot reproduce the frozen public signer split')
    review = read(ROOT/'datasets/unknown-review.json')
    data = read(ROOT/'datasets/unknown-hands.json')
    samples, failures = data['samples'],data['tracking_failures']
    parts = {split:[r for r in samples if r['split']==split] for split in ['train','validation','test']}
    legacy = {split:frozen_rows(ROOT/'datasets/prepared.json',legacy_report,split) for split in ['validation','test']}
    public = {split:frozen_rows(ROOT/'datasets/public-balanced.json',public_report,split) for split in ['validation','test']}
    # Confirm people in known and unknown public sources never cross partitions.
    for split in parts:
        people = {r['participant'] for r in parts[split]}
        for other in parts.keys()-{split}:
            known_people = set(public_report['participant_ids'][other])
            unknown_people = {r['participant'] for r in parts[other]}
            if people & (known_people|unknown_people):
                raise ValueError('Participant leakage in rejection benchmark')
    support = [r['features'] for r in parts['train']]
    candidate = copy.deepcopy(base)
    policy = review['validation_policy']
    baseline_accepted = int((labels(base,parts['validation'])!='_').sum())
    trials = []
    chosen = None
    for ratio in policy['ratio_grid']:
        candidate['rejection'] = {'version':'negative-distance-ratio-v1','ratio':ratio,'support':support,
            'license':'HaGRID custom public license (datasets/HAGRID-LICENSE.pdf) and ASLYset CC BY 4.0; see UNKNOWN_HANDS.md',
            'attribution':[
                {'source':'HaGRID / HaGRIDv2','creators':'Alexander Kapitanov; Andrey Makhlyarchuk; Karina Kvanchiani; Aleksandr Nagaev; Roman Kraynov; Anton Nuzhdin',
                 'url':'https://github.com/hukenovs/hagrid','license_url':'https://raw.githubusercontent.com/hukenovs/hagrid/master/license/en_us.pdf',
                 'license_notice':'Custom public license with attribution and conditions reserved; not a Creative Commons license. Supplied as-is, without warranties.',
                 'changes':'Image selection, visual review, resting-hand crops, MediaPipe extraction, normalization, deduplication and non-alphabet support export.'},
                {'source':'ASLYset V1 (2019), doi:10.17632/xs6mvhx6rh.1','creators':'Miguel Rivera',
                 'url':'https://data.mendeley.com/datasets/xs6mvhx6rh/1','license_url':'https://creativecommons.org/licenses/by/4.0/',
                 'changes':'Selection of FN controls, MediaPipe extraction, normalization and non-alphabet support export.'}],
            'training_fingerprints':[r['fingerprint'] for r in parts['train']]}
        accepted = int((labels(candidate,parts['validation'])!='_').sum())
        eligible = retention_ok(base,candidate,legacy['validation'],public['validation'],policy)
        trials.append({'ratio':ratio,'unknown_false_acceptances':accepted,'positive_retention_passed':eligible})
        if eligible and accepted<baseline_accepted and (chosen is None or accepted<chosen[0]):
            chosen = (accepted,copy.deepcopy(candidate))
    promoted = chosen is not None
    candidate = chosen[1] if promoted else base
    # Only after the complete validation decision, open/evaluate final partitions.
    result = {'version':1,'method':'Frozen class network plus nearest reviewed negative/positive support ratio veto. Threshold selected on validation only.',
        'confidence_threshold':.85,'policy':policy,'validation_trials':trials,'eligible_for_install':promoted,
        'selected_ratio':candidate.get('rejection',{}).get('ratio'),'unknown_counts':{k:len(v) for k,v in parts.items()},
        'unknown_dataset_sha256':hashlib.sha256((ROOT/'datasets/unknown-hands.json').read_bytes()).hexdigest(),
        'positive_datasets_sha256':{'legacy':legacy_report['dataset_sha256'],'public':public_sha},
        'public_positive_participants':public_report['participant_ids'],
        'unknown_participants':{split:sorted({r['participant'] for r in rows}) for split,rows in parts.items()},
        'unknown_split_fingerprints':{split:[r['fingerprint'] for r in rows] for split,rows in parts.items()},
        'baseline_classifier_sha256':hashlib.sha256((BASELINE/'classifier.json').read_bytes()).hexdigest(),
        'candidate_classifier_sha256':hashlib.sha256(json.dumps(candidate,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
        'limitations':['Visual review by Codex, not ASL expert certification.','Partial flexion is a proxy; intended incomplete signs are not annotated.',
          'Still images/cropped resting hands do not establish live webcam reliability.','Unknown physical hands, lighting and camera angles.',
          'Legacy known-letter split lacks signer IDs. Public known/unknown splits are source-participant-disjoint.',
          'Dataset rejection labels cover the alphabet only; some unsupported shapes are valid numbers or signs.',
          'A handshape identical to a valid letter cannot be rejected based on intent alone.','Small, correlated categories; no universal open-set accuracy claim.'],
        'test_examples':[]}
    for name, model in [('baseline',base),('candidate',candidate)]:
        result[name] = {split:{'unknown':unknown_metrics(model,parts[split],[r for r in failures if r['split']==split]),
                              'legacy_known':positive_metrics(model,legacy[split]),'public_known':positive_metrics(model,public[split])}
                       for split in ['validation','test']}
    test_predictions = labels(candidate,parts['test'])
    result['test_examples'] = [{k:r[k] for k in ['features','fingerprint','category','source','source_sha256','participant']}
                              | {'expected':'_','accepted':None if p=='_' else str(p)} for r,p in zip(parts['test'],test_predictions)]
    directory = ROOT/'artifacts/rejection/candidate'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'classifier.json').write_text(json.dumps(candidate,separators=(',',':'),allow_nan=False),encoding='utf-8')
    (ROOT/'datasets/unknown-evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    if args.install:
        if not promoted:
            raise ValueError('No candidate met validation promotion criteria; baseline retained')
        (MODEL/'classifier.json').write_bytes((directory/'classifier.json').read_bytes())
        metadata = read(BASELINE/'metadata.json')
        metadata['rejection'] = {'version':candidate['rejection']['version'],'ratio':candidate['rejection']['ratio'],
                                 'report':'datasets/unknown-evaluation.json','experimental':True}
        (MODEL/'metadata.json').write_text(json.dumps(metadata,separators=(',',':')),encoding='utf-8')
        updated_report = copy.deepcopy(legacy_report)
        updated_report['gated_labels'] = candidate['labels']+['_']
        updated_report['previous_gated_accuracy_at_threshold_085'] = legacy_report['gated_accuracy_at_threshold_085']
        updated_report['gated_accuracy_at_threshold_085'] = result['candidate']['test']['legacy_known']['gated_accuracy']
        updated_report['gated_confusion_matrix'] = confusion_matrix([r['letter'] for r in legacy['test']],labels(candidate,legacy['test']),labels=updated_report['gated_labels']).tolist()
        updated_report['rejection_evaluated'] = True
        updated_report['rejection_benchmark'] = {'report':'datasets/unknown-evaluation.json',
            **result['candidate']['test']['unknown'], 'warning':'Small visually reviewed still-image benchmark; partial flexion is only a proxy for incomplete signs.'}
        (MODEL/'evaluation.json').write_text(json.dumps(updated_report,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print(json.dumps({k:result[k] for k in ['eligible_for_install','selected_ratio','unknown_counts','validation_trials']},indent=2))
    for name in ['baseline','candidate']:
        for split in ['validation','test']:
            print(name,split,json.dumps({key:{k:v for k,v in value.items() if not k.startswith('by_')} for key,value in result[name][split].items()}))


if __name__=='__main__':
    main()
