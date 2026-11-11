"""Freeze the public signer holdout and expand only its training participants."""
import hashlib
import json
from .data_quality import ROOT, curate, load_records
from .features import LABELS
from .train import split_indices
import numpy as np

FOCUS = list('EMRT')


def public_partitions():
    path = ROOT/'datasets/public-balanced.json'
    raw = path.read_bytes()
    balanced = json.loads(raw)
    previous = json.loads((ROOT/'datasets/public-candidate-evaluation.json').read_text())
    if hashlib.sha256(raw).hexdigest()!=previous['dataset_sha256']:
        raise ValueError('The frozen public positive dataset has changed')
    train, val, test, name = split_indices(balanced,np.array([r['letter'] for r in balanced]))
    if name!='participant-disjoint':
        raise ValueError('Public training requires source-participant-disjoint splits')
    parts = {key:[balanced[i] for i in ids] for key,ids in [('train',train),('validation',val),('test',test)]}
    people = {key:sorted({r['participant'] for r in rows}) for key,rows in parts.items()}
    if people['test']!=previous['test_participants']:
        raise ValueError('The frozen public test participants have changed')
    pool, audit = curate(load_records([ROOT/'datasets/aslyset-landmarks.json',ROOT/'datasets/asl_hg-landmarks.json']))
    if audit['conflicts'] or audit['invalid']:
        raise ValueError('Resolve public source conflicts before expanding training')
    expanded = [r for r in pool if r['participant'] in people['train'] and r['letter'] in LABELS]
    if any(r.get('training_eligible') is False for r in expanded):
        raise ValueError('Review-only data cannot enter training')
    parts['train'] = expanded
    for key in ['participant','fingerprint','source_sha256']:
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            if {r[key] for r in parts[a]}&{r[key] for r in parts[b]}:
                raise ValueError(f'Expanded training leaks {key} across splits')
    if not {r['fingerprint'] for r in balanced if r['participant'] in people['train']}.issubset({r['fingerprint'] for r in expanded}):
        raise ValueError('Expansion lost original training examples')
    return parts, {'dataset_sha256':hashlib.sha256(raw).hexdigest(),'participant_ids':people,
                   'source_hashes':{name:hashlib.sha256((ROOT/'datasets'/name).read_bytes()).hexdigest() for name in ['aslyset-landmarks.json','asl_hg-landmarks.json']},
                   'curation':audit}
