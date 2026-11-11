"""Evaluate a sealed candidate on a frozen cohort; record prior use explicitly."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from .data_quality import ROOT
from .static_features import record_features
from .weak_letter_data import public_partitions
from .compare_public_models import predict, probabilities, metrics


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def partition_signature(rows, field='fingerprint'):
    """Seal complete cohort identities, including duplicates and row counts."""
    values = [r[field] for r in rows]
    if any(not isinstance(value,str) or not value for value in values):
        raise ValueError('Missing evaluation source identity')
    if len(set(values)) != len(values):
        raise ValueError('Duplicate evaluation source identity')
    return {'count':len(rows), field+'_sha256':hashlib.sha256(
        '\n'.join(sorted(values)).encode('utf-8')).hexdigest()}


def cohort_manifest(positive_path, negative_path, test_split):
    """Build a reproducible seal before scoring, without transforming landmarks."""
    if test_split not in {'test','fresh_test'}:
        raise ValueError('Unknown evaluation test split')
    positive, negative = read(positive_path),read(negative_path)
    known, unknown, failures = {},{},{}
    for split in ['train','validation','test']:
        selected = test_split if split=='test' else split
        rows = [r for r in positive['samples'] if r['split']==selected]
        if not rows:
            raise ValueError('Empty known evaluation partition: '+selected)
        known[split] = partition_signature(rows)
        rows = [r for r in negative['samples'] if r['split']==split]
        if not rows:
            raise ValueError('Empty unknown evaluation partition: '+split)
        unknown[split] = partition_signature(rows)
        # Failed tracks have no coordinate fingerprint. Preserve their source
        # identity separately so failures cannot disappear from denominators.
        rows = [r for r in negative.get('tracking_failures',[]) if r['split']==split]
        identities = [{'source_identity':json.dumps({key:r.get(key) for key in
            ['source','source_file','source_annotation','source_sha256','participant','category','letter']},
            sort_keys=True,separators=(',',':'))} for r in rows]
        failures[split] = partition_signature(identities,'source_identity')
    return {'version':'frozen-static-depth-cohort-v1',
        'positive_dataset_sha256':sha(positive_path),'unknown_dataset_sha256':sha(negative_path),
        'test_split':test_split,'known':known,'unknown':unknown,'unknown_tracking_failures':failures}


def validate_evaluation_cohort(release, positive_path, negative_path, test_split):
    """Fail closed on changed source bytes, split choices or cohort membership."""
    sealed = release.get('evaluation_cohort')
    if not sealed or sealed.get('version')!='frozen-static-depth-cohort-v1':
        raise ValueError('Release policy has no supported frozen evaluation cohort')
    if test_split != sealed.get('test_split'):
        raise ValueError('Requested test split differs from frozen evaluation cohort')
    if (sha(positive_path)!=sealed.get('positive_dataset_sha256') or
            sha(negative_path)!=sealed.get('unknown_dataset_sha256')):
        raise ValueError('Evaluation dataset bytes differ from frozen cohort')
    if cohort_manifest(positive_path,negative_path,test_split) != sealed:
        raise ValueError('Evaluation identities/counts differ from frozen cohort')


def known_metrics(model, rows, features):
    raw, gated = predict(model,features)
    result = metrics(rows,raw,gated)
    for label, value in result['by_letter'].items():
        value['correct_accepted'] = int(sum(gated[i]==label for i,r in enumerate(rows) if r['letter']==label))
    result['incorrect_accepted'] = int(sum(p!='_' and p!=r['letter'] for r,p in zip(rows,gated)))
    return result, raw, gated


def unknown_metrics(model, rows, features, failures):
    _, gated = predict(model,features)
    result = {'tracked':len(rows),'selected':len(rows)+len(failures),'tracking_failures':len(failures),
              'false_acceptances':int((gated!='_').sum()),'by_category':{}}
    result['false_acceptance_rate_tracked'] = result['false_acceptances']/len(rows)
    result['false_acceptance_rate_end_to_end'] = result['false_acceptances']/result['selected']
    for category in sorted({r['category'] for r in rows+failures}):
        ids = [i for i,r in enumerate(rows) if r['category']==category]
        result['by_category'][category] = {'tracked':len(ids),'false_acceptances':int((gated[ids]!='_').sum()),
                                           'tracking_failures':sum(r['category']==category for r in failures)}
    return result, gated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',type=Path,default=ROOT/'artifacts/static-depth-candidate')
    parser.add_argument('--dataset',type=Path,default=ROOT/'datasets/static-depth.json')
    parser.add_argument('--test-split',choices=['test','fresh_test'],default='test')
    parser.add_argument('--policy',type=Path,default=ROOT/'datasets/weak-letter-release-policy.json')
    parser.add_argument('--validation',type=Path,default=ROOT/'datasets/weak-letter-validation.json')
    parser.add_argument('--report',type=Path,default=ROOT/'datasets/weak-letter-evaluation.json')
    parser.add_argument('--install',action='store_true')
    args = parser.parse_args()
    path = args.candidate/'classifier.json'
    model = read(path)
    release = read(args.policy)
    if sha(path)!=release['candidate_classifier_sha256'] or sha(args.validation)!=release['validation_choice_sha256']:
        raise ValueError('Sealed candidate or validation decision changed after selection')
    frozen_parts, source_audit = public_partitions()
    positive_path = args.dataset.resolve()
    negative_path = ROOT/'datasets/unknown-depth.json'
    validate_evaluation_cohort(release,positive_path,negative_path,args.test_split)
    positive, negative = read(positive_path),read(negative_path)
    parts = {'train':[r for r in positive['samples'] if r['split']=='train'],
             'validation':[r for r in positive['samples'] if r['split']=='validation'],
             'test':[r for r in positive['samples'] if r['split']==args.test_split]}
    if {r['fingerprint'] for r in parts['validation']}!={r['fingerprint'] for r in frozen_parts['validation']}:
        raise ValueError('Validation cohort changed during candidate selection')
    expected_people = source_audit['participant_ids']
    for split in parts:
        if {r['participant'] for r in parts[split]}!=set(expected_people[split]):
            raise ValueError('Unexpected participants in '+split)
    by_fp = {r['fingerprint']:r for r in positive['samples']}
    baseline = read(ROOT/'datasets/baselines/static-v2/classifier.json')
    comparison = read(ROOT/'datasets/public-candidate-evaluation.json')
    report = {'feature_version':model['feature_version'],'classifier_sha256':sha(path),'release_policy':release,
        'positive_dataset_sha256':sha(positive_path),'unknown_dataset_sha256':sha(negative_path),
        'validation_choice_sha256':sha(args.validation),
        'source_audit':source_audit,'counts':{k:len(v) for k,v in parts.items()},
        'positive_dataset':str(positive_path.relative_to(ROOT)).replace('\\','/'),'test_split':args.test_split,
        'test_definition':release.get('test_use', 'Previously frozen public test images; prior use must be checked before interpreting results.'),
        'primary_support_fingerprints':model['support_fingerprints'],
        'previous_candidate':comparison['public_candidate'],
        'baseline':{},'candidate':{},'unknown_test_examples':[],
        'limitations':['Image z and world geometry are MediaPipe estimates, not measured depth.',
          'Source-identified public train/validation/test participants are disjoint; frozen SigNN E/R expert signers are unknown.',
          'Still images do not establish live webcam accuracy; Python IMAGE 0.10.35 and browser VIDEO 0.10.21 differ.',
          'Physical left-hand, lighting, camera-angle and session coverage remain incomplete.',
          'Negative visual review is not ASL expert certification; partial flexion is only a proxy for incomplete signs.']}
    for split in ['validation','test']:
        rows = [by_fp[r['fingerprint']] for r in parts[split]]
        expected_split = args.test_split if split=='test' else split
        if any(r['split']!=expected_split for r in rows):
            raise ValueError('Depth rows changed frozen partitions')
        x = np.array([record_features(r,model) for r in rows])
        unknown = [r for r in negative['samples'] if r['split']==split]
        nx = np.array([record_features(r,model) for r in unknown])
        failures = [r for r in negative.get('tracking_failures',[]) if r['split']==split]
        old_path = ROOT/'datasets/baselines/public-candidate-v1/classifier.json'
        models = [('baseline',baseline,x[:,:42],nx[:,:42]),('candidate',model,x,nx)]
        if old_path.exists():
            report.setdefault('old_public_candidate',{})
            models.append(('old_public_candidate',read(old_path),x[:,:42],nx[:,:42]))
        for name, current, px, ux in models:
            known, raw, gated = known_metrics(current,rows,px)
            rejects, unknown_gated = unknown_metrics(current,unknown,ux,failures)
            report[name][split] = {'known':known,'unknown':rejects}
            if name=='candidate' and split=='test':
                candidate_test_gated = gated.copy()
                p = probabilities(model,x)
                labels = model['labels']
                evaluation = {'dataset_sha256':sha(positive_path),'seed':{'model':17},'feature_version':model['feature_version'],
                    'split':report['test_definition']+' Legacy expert signers unidentified.',
                    'counts':{'unique_total':sum(len(v) for v in parts.values()),'train':len(parts['train']),'validation':len(parts['validation']),'test':len(rows)},
                    'classes':labels,'gated_labels':labels+['_'],'raw_classifier_accuracy':known['raw_accuracy'],
                    'supported_letter_accuracy':known['raw_accuracy'],'supported_letter_test_samples':len(rows),
                    'gated_accuracy_at_threshold_085':known['gated_accuracy'],'classification_report':classification_report([r['letter'] for r in rows],raw,labels=labels,output_dict=True,zero_division=0),
                    'confusion_matrix':confusion_matrix([r['letter'] for r in rows],raw,labels=labels).tolist(),
                    'gated_confusion_matrix':confusion_matrix([r['letter'] for r in rows],gated,labels=labels+['_']).tolist(),
                    'split_fingerprints':{k:[r['fingerprint'] for r in v] for k,v in parts.items()},
                    'rejection_evaluated':True,'rejection_benchmark':rejects,'warning':'Experimental source-image benchmark; see WEAK_LETTERS.md.',
                    'test_examples':[{'features':x[i].tolist(),'expected':r['letter'],'probabilities':p[i].tolist(),'accepted':None if gated[i]=='_' else str(gated[i])} for i,r in enumerate(rows)]}
                report['unknown_test_examples'] = [{'fingerprint':r['fingerprint'],'features':nx[i].tolist(),'category':r['category'],
                    'source':r['source'],'accepted':None if unknown_gated[i]=='_' else str(unknown_gated[i])} for i,r in enumerate(unknown)]
    new = report['candidate']['test']
    report['release_passed'] = (new['known']['gated_accuracy']>=release['minimum_gated_test_accuracy'] and
        new['unknown']['false_acceptances']<=release['maximum_unknown_test_false_acceptances'] and
        all(new['known']['by_letter'][label]['correct_accepted']>=minimum for label,minimum in release['minimum_correct_test_by_letter'].items()))
    if release.get('minimum_r_precision') is not None:
        accepted_r = int((candidate_test_gated=='R').sum())
        correct_r = new['known']['by_letter']['R']['correct_accepted']
        report['r_precision'] = correct_r/accepted_r if accepted_r else 0
        report['release_passed'] &= report['r_precision']>=release['minimum_r_precision']
    report['release_failures'] = []
    if new['known']['gated_accuracy']<release['minimum_gated_test_accuracy']:
        report['release_failures'].append({'criterion':'overall_gated_accuracy','actual':new['known']['gated_accuracy'],'minimum':release['minimum_gated_test_accuracy']})
    for label, minimum in release['minimum_correct_test_by_letter'].items():
        actual = new['known']['by_letter'][label]['correct_accepted']
        if actual<minimum:
            report['release_failures'].append({'criterion':label+'_correct_accepted','actual':actual,'minimum':minimum})
    if new['unknown']['false_acceptances']>release['maximum_unknown_test_false_acceptances']:
        report['release_failures'].append({'criterion':'unknown_false_acceptances','actual':new['unknown']['false_acceptances'],'maximum':release['maximum_unknown_test_false_acceptances']})
    if report.get('r_precision',1)<release.get('minimum_r_precision',0):
        report['release_failures'].append({'criterion':'R_precision','actual':report['r_precision'],'minimum':release['minimum_r_precision']})
    references = {}
    train = [r for r in positive['samples'] if r['split']=='train']
    for label in model['labels']:
        candidates = [r for r in train if r['letter']==label and r['fingerprint'] in set(model['support_fingerprints'])]
        vectors = np.array([record_features(r,model) for r in candidates])
        _, accepted = predict(model,vectors)
        valid = [r for r,p in zip(candidates,accepted) if p==label]
        if not valid:
            raise ValueError('No accepted real training reference for '+label)
        center = np.median([record_features(r,model) for r in valid],axis=0)
        reference = min(valid,key=lambda r:np.linalg.norm(record_features(r,model)-center))
        references[label] = np.array(reference['points']).reshape(21,3)[:,:2].ravel().tolist()
    metadata = {'ready':True,'labels':model['labels'],'feature_version':model['feature_version'],'training_samples':len(train),
        'test_samples':evaluation['counts']['test'],'test_accuracy':evaluation['raw_classifier_accuracy'],
        'supported_accuracy':evaluation['supported_letter_accuracy'],'supported_test_samples':evaluation['supported_letter_test_samples'],
        'gated_accuracy':new['known']['gated_accuracy'],'experimental':True,'low_support_warning':'Estimated depth; legacy E/R expert has unidentified signers.',
        'split':evaluation['split'],'source':'ASLYset + ASL-HG; frozen SigNN E/R expert',
        'warning':evaluation['warning'],'rejection':{'report':str(args.report.resolve().relative_to(ROOT)).replace('\\','/'),'experimental':True,
            'test_samples':new['unknown']['tracked'],'false_acceptances':new['unknown']['false_acceptances']}}
    for filename, value in [('evaluation.json',evaluation),('metadata.json',metadata),('references.json',references)]:
        (args.candidate/filename).write_text(json.dumps(value,separators=(',',':'),allow_nan=False),encoding='utf-8')
    args.report.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'release_passed':report['release_passed'],'raw_accuracy':new['known']['raw_accuracy'],
        'gated_accuracy':new['known']['gated_accuracy'],'focus':{k:new['known']['by_letter'][k] for k in 'EMRT'},
        'unknown':new['unknown'],'release_failures':report['release_failures']},indent=2))
    if args.install:
        if not report['release_passed']:
            raise ValueError('Frozen release criteria failed; deployed model retained')
        for name in ['classifier.json','evaluation.json','metadata.json','references.json']:
            (ROOT/'signspeak/static/models'/name).write_bytes((args.candidate/name).read_bytes())


if __name__=='__main__':
    main()
