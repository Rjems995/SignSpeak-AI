"""Expand fixed training signers and freeze fresh images from assessed test signers.

No classifier predictions enter selection. The new image holdout does not
constitute new unseen signers: its people were assessed in the earlier test.
"""
import argparse
import hashlib
import json
import stat
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from .data_quality import ROOT, curate, load_records
from .features import LABELS, normalize
from .import_public import source_metadata
from .import_static_depth import SourceImages, digest, native_log, valid_geometry, MAX_IMAGE_BYTES
from .weak_letter_data import public_partitions

FOCUS = set('EMRT')
EXPAND = set('EMRTSNUV')
MANIFEST = ROOT / 'datasets/focus-depth-selection.json'
OUTPUT = ROOT / 'datasets/focus-depth.json'
ARTIFACTS = ROOT / 'artifacts/focus-depth'


def file_order(row):
    return hashlib.sha256(row['source_file'].encode()).hexdigest()


def incompatible(group):
    return any(len({row.get(key) for row in group}) > 1 for key in ['letter', 'participant', 'split'])


def quarantine(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    bad = {value for value, group in groups.items() if incompatible(group)}
    audit = [{key: value, 'rows': [{k: row.get(k) for k in ['source_file','letter','participant','split','fingerprint']} for row in groups[value]]}
             for value in sorted(bad)]
    return [row for row in rows if row[key] not in bad], audit


def prepare(reserve=10):
    if reserve < 0:
        raise ValueError('The deterministic tracking/curation reserve cannot be negative')
    old_path = ROOT / 'datasets/static-depth.json'
    old = json.loads(old_path.read_text())
    balanced_path = ROOT / 'datasets/public-balanced.json'
    balanced = json.loads(balanced_path.read_text())
    parts, partition_audit = public_partitions()
    train_people = set(partition_audit['participant_ids']['train'])
    test_people = set(partition_audit['participant_ids']['test'])
    old_files = {row['source_file'] for row in old['samples']}
    balanced_files = {row['source_file'] for row in balanced}
    balanced_hashes = {row['source_sha256'] for row in balanced}
    balanced_fingerprints = {row['fingerprint'] for row in balanced}
    candidates = []
    source = json.loads((ROOT / 'datasets/public-sources.json').read_text())['asl_hg']
    archive = ROOT / 'datasets/public/asl_hg' / source['archive']
    if archive.stat().st_size != source['bytes'] or digest(archive) != source['sha256']:
        raise ValueError('ASL-HG archive must match its pinned source hash/size')
    with zipfile.ZipFile(archive) as zipped:
        if len(zipped.namelist()) != len(set(zipped.namelist())):
            raise ValueError('Duplicate original image paths')
        for info in zipped.infolist():
            metadata = source_metadata('asl_hg', info.filename)
            if not metadata or info.filename in old_files:
                continue
            training = metadata['participant'] in train_people and metadata['letter'] in EXPAND
            fresh = metadata['participant'] in test_people and info.filename not in balanced_files
            if not training and not fresh:
                continue
            if info.is_dir() or stat.S_IFMT(info.external_attr >> 16) not in {0,stat.S_IFREG} or info.file_size > MAX_IMAGE_BYTES:
                raise ValueError('Selected ASL-HG images must be bounded regular files')
            data = zipped.read(info)
            image_hash = hashlib.sha256(data).hexdigest()
            # Fresh images must be absent from every original balanced source hash.
            if fresh and image_hash in balanced_hashes:
                continue
            candidates.append(dict(metadata, source_sha256=image_hash, split='train' if training else 'fresh_test'))
    eligible, audit = curate(load_records([ROOT/'datasets/aslyset-landmarks.json', ROOT/'datasets/asl_hg-landmarks.json']))
    if audit['invalid'] or audit['conflicts']:
        raise ValueError('Resolve source conflicts before selecting fresh images')
    asly = [dict(row, split='fresh_test') for row in eligible if row['source']=='ASLYset' and row['participant'] in test_people
            and row['source_file'] not in balanced_files and row['source_sha256'] not in balanced_hashes
            and row['fingerprint'] not in balanced_fingerprints]
    images = SourceImages(asly)
    try:
        for row in asly:
            images.read(row)
            candidates.append({key:value for key,value in row.items() if key not in {'features','points'}})
    finally:
        images.close()
    # Include all old rows in global image-hash quarantine before selecting caps.
    all_rows = [dict(row, origin='old') for row in old['samples']] + [dict(row, origin='new') for row in candidates]
    compatible, hash_conflicts = quarantine(all_rows, 'source_sha256')
    compatible_hashes = {row['source_sha256'] for row in compatible}
    groups = defaultdict(list)
    for row in compatible:
        groups[row['source_sha256']].append(row)
    deduped = [sorted(group, key=lambda row:(row['origin']!='old',file_order(row)))[0] for group in groups.values()]
    raw_candidates = [row for row in deduped if row['origin']=='new']
    selected = [row for row in raw_candidates if row['split']=='train']
    cells = defaultdict(list)
    for row in raw_candidates:
        if row['split']=='fresh_test':
            cells[(row['participant'],row['letter'])].append(row)
    limits = {}
    for (person,letter), rows in sorted(cells.items()):
        target = 20 if letter in FOCUS else 10
        limit = min(len(rows),target+reserve)
        selected.extend(sorted(rows,key=file_order)[:limit])
        limits[f'{person}/{letter}'] = {'target':target,'available_after_hash_quarantine':len(rows),'extraction_candidates':limit}
    if any((person,letter) not in cells for person in test_people for letter in LABELS):
        raise ValueError('A fresh test signer/letter cell has no eligible original images')
    manifest = {'version':1,'selection_rule':'SHA-256 source filename order; no classifier predictions. Targets20 for E/M/R/T,10 for other static letters per old test person; reserve10 for tracking/fingerprint curation.',
        'holdout_type':'Fresh image holdout on previously assessed test participants; not new unseen signers.',
        'expanded_training_letters':sorted(EXPAND),'focus_letters':sorted(FOCUS),'reserve':reserve,
        'static_depth_sha256':digest(old_path),'balanced_dataset_sha256':digest(balanced_path),
        'partition_audit':partition_audit,'archive_sha256':{'ASL-HG':source['sha256'],**{k:v['sha256'] for k,v in images.archives.items()}},
        'old_training_count':sum(row['split']=='train' for row in old['samples']),
        'old_validation_count':sum(row['split']=='validation' for row in old['samples']),
        'image_hash_quarantine':hash_conflicts,'old_hash_quarantine_files':sorted(row['source_file'] for row in old['samples'] if row['source_sha256'] not in compatible_hashes),
        'limits':limits,'selected_counts':dict(Counter(row['split'] for row in selected)),
        'selected_source_counts':dict(Counter(row['source']+'/'+row['split'] for row in selected)),
        'samples':[{k:v for k,v in row.items() if k!='origin'} for row in selected]}
    MANIFEST.write_text(json.dumps(manifest,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print('Frozen source selection',manifest['selected_counts'],manifest['selected_source_counts'],'hash conflicts',len(hash_conflicts),flush=True)
    return manifest


def extract(source):
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    manifest=json.loads(MANIFEST.read_text())
    rows=[row for row in manifest['samples'] if row['source']==source]
    name={'ASLYset':'aslyset','ASL-HG':'asl_hg'}[source]
    path=ARTIFACTS/(name+'-expanded.json')
    log=ARTIFACTS/(name+'-mediapipe.log')
    samples=[];failures=[]
    with native_log(log):
        import cv2
        import mediapipe as mp
        import numpy as np
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        task=ROOT/'datasets/videos/hand_landmarker.task'
        extractor={'mediapipe_version':mp.__version__,'mode':'IMAGE','num_hands':2,'detection_confidence':.6,'presence_confidence':.6,'task_sha256':digest(task)}
        images=SourceImages(rows)
        options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(task)),running_mode=vision.RunningMode.IMAGE,
            num_hands=2,min_hand_detection_confidence=.6,min_hand_presence_confidence=.6)
        try:
            with vision.HandLandmarker.create_from_options(options) as detector:
                for i,row in enumerate(rows):
                    data=images.read(row)
                    pixels=cv2.imdecode(np.frombuffer(data,dtype=np.uint8),cv2.IMREAD_COLOR)
                    if pixels is None:
                        failures.append(dict(row,reason='decode_failed'));continue
                    result=detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(pixels,cv2.COLOR_BGR2RGB)))
                    if len(result.hand_landmarks)!=1:
                        failures.append(dict(row,reason='requires_one_detected_hand'));continue
                    points=[v for p in result.hand_landmarks[0] for v in (p.x,p.y,p.z)]
                    world=[v for p in result.hand_world_landmarks[0] for v in (p.x,p.y,p.z)] if result.hand_world_landmarks else None
                    geometry={'points':points,'world_points':world}
                    if not valid_geometry(geometry):
                        failures.append(dict(row,reason='invalid_3d_geometry'));continue
                    features=normalize(points)
                    fingerprint=hashlib.sha256(features.round(5).tobytes()).hexdigest()
                    samples.append(dict(row,**geometry,fingerprint=fingerprint))
                    if (i+1)%400==0:print(source,i+1,'/',len(rows),'accepted',len(samples),flush=True)
        finally:images.close()
    report={'selection_manifest_sha256':digest(MANIFEST),'extractor':extractor,'selected':len(rows),'samples':samples,'failures':failures}
    path.write_text(json.dumps(report,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print('WROTE',path,len(samples),'samples',len(failures),'failures',flush=True)
    return report


def finalize():
    manifest=json.loads(MANIFEST.read_text());old_path=ROOT/'datasets/static-depth.json'
    if digest(old_path)!=manifest['static_depth_sha256']:
        raise ValueError('Original frozen static depth dataset changed during expansion')
    old=json.loads(old_path.read_text())
    old_rows=[dict(row,origin='old') for row in old['samples'] if row['source_file'] not in manifest['old_hash_quarantine_files']]
    newly=[];failures=[];extractors=[]
    for source in ['aslyset','asl_hg']:
        data=json.loads((ARTIFACTS/(source+'-expanded.json')).read_text())
        if data['selection_manifest_sha256']!=digest(MANIFEST):raise ValueError('Extraction used a different frozen source manifest')
        newly += [dict(row,origin='new') for row in data['samples']]
        failures += data['failures'];extractors.append(data['extractor'])
    if extractors[0]!=extractors[1]:raise ValueError('Extractors differ between sources')
    rows,hash_conflicts=quarantine(old_rows+newly,'source_sha256')
    rows,fp_conflicts=quarantine(rows,'fingerprint')
    groups=defaultdict(list)
    for row in rows:groups[row['fingerprint']].append(row)
    unique=[sorted(group,key=lambda row:(row['origin']!='old',file_order(row)))[0] for group in groups.values()]
    training=[row for row in unique if row['split'] in {'train','validation'}]
    fresh=[];cells=defaultdict(list)
    for row in unique:
        if row['split']=='fresh_test':cells[(row['participant'],row['letter'])].append(row)
    gaps=[]
    for person in manifest['partition_audit']['participant_ids']['test']:
        for letter in LABELS:
            limit=20 if letter in FOCUS else 10
            chosen=sorted(cells[(person,letter)],key=file_order)[:limit]
            fresh += chosen
            if len(chosen)<limit:gaps.append({'participant':person,'letter':letter,'available':len(chosen),'target':limit})
    balanced=json.loads((ROOT/'datasets/public-balanced.json').read_text())
    for key in ['source_file','source_sha256','fingerprint']:
        if {row[key] for row in fresh}&{row[key] for row in balanced}:
            raise ValueError('Fresh image holdout overlaps the original balanced dataset')
    samples=[{k:v for k,v in row.items() if k!='origin'} for row in training+fresh]
    retained_training_fingerprints = {row['fingerprint'] for row in training}
    if any(row.get('training_eligible') is False for row in samples):raise ValueError('Review-only rows cannot enter expanded training/evaluation')
    for key in ['participant','source_sha256','fingerprint']:
        for a,b in [('train','validation'),('train','fresh_test'),('validation','fresh_test')]:
            if {row[key] for row in samples if row['split']==a}&{row[key] for row in samples if row['split']==b}:
                raise ValueError('Final focus split leakage: '+key)
    result={'samples':samples,'tracking_failures':failures,'extractor':extractors[0],
        'selection_manifest_sha256':digest(MANIFEST),'holdout_type':manifest['holdout_type'],
        'selected_counts':dict(Counter(row['split'] for row in samples)),
        'fresh_test_cells':{person+'/'+letter:sum(row['participant']==person and row['letter']==letter for row in fresh)
                            for person in manifest['partition_audit']['participant_ids']['test'] for letter in LABELS},
        'quarantine':{'source_image_hashes':manifest['image_hash_quarantine']+hash_conflicts,'canonical_xy_fingerprints':fp_conflicts},
        'removed_old_rows':[{k:row[k] for k in ['source_file','fingerprint','source_sha256','split','participant','letter']}
                            for row in old['samples'] if row['split'] in {'train','validation'} and row['fingerprint'] not in retained_training_fingerprints],
        'fresh_test_gaps':gaps,'frozen_old_test':[{k:row[k] for k in ['source_file','fingerprint','source_sha256','participant','letter']} for row in old['samples'] if row['split']=='test']}
    OUTPUT.write_text(json.dumps(result,separators=(',',':'),allow_nan=False),encoding='utf-8')
    train_only=dict(result,samples=[row for row in samples if row['split']!='fresh_test'])
    (ARTIFACTS/'train-validation.json').write_text(json.dumps(train_only,separators=(',',':'),allow_nan=False),encoding='utf-8')
    print('FINAL',result['selected_counts'],'removed_old',len(result['removed_old_rows']),'gaps',gaps,flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','extract','finalize','all'],default='all')
    parser.add_argument('--source',choices=['ASLYset','ASL-HG'])
    parser.add_argument('--reserve',type=int,default=10)
    args=parser.parse_args()
    ARTIFACTS.mkdir(parents=True,exist_ok=True)
    if args.phase in {'prepare','all'}:prepare(args.reserve)
    if args.phase in {'extract','all'}:
        for source in ([args.source] if args.source else ['ASLYset','ASL-HG']):extract(source)
    if args.phase in {'finalize','all'}:finalize()


if __name__=='__main__':main()
