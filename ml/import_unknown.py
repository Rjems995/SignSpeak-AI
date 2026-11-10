"""Import visually reviewed real non-letter handshapes; never infer intent."""
import hashlib
import json
import os
import subprocess
from collections import Counter
from pathlib import Path
from .data_quality import ROOT, curate


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    output = ROOT / 'artifacts/rejection'
    output.mkdir(parents=True, exist_ok=True)
    # MediaPipe native logging otherwise overwhelms the useful import audit.
    log = (output / 'mediapipe.log').open('w')
    os.dup2(log.fileno(), 2)
    import cv2
    import mediapipe as mp
    import numpy as np
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    review = json.loads((ROOT / 'datasets/unknown-review.json').read_text())
    inputs, sources = [], {}
    directory = ROOT / 'datasets/public/hagrid'
    for key, name, target in [('hagrid_no_gesture','review-selection.json','no_gesture'),('hagrid_resting','resting-selection.json','four')]:
        selection = json.loads((directory / name).read_text())
        manifest_path = directory / f'{target}-download-manifest.json'
        if not manifest_path.exists() and target=='no_gesture':
            manifest_path = directory / 'download-manifest.json'
        manifest = json.loads(manifest_path.read_text())
        for file, spec in manifest.items():
            if digest(directory / file) != spec['sha256']:
                raise ValueError('Official annotation hash mismatch')
        sources.update(manifest)
        for category, indices in review[key].items():
            if not isinstance(indices,list):
                continue
            for index in indices:
                item = selection[index]
                path = directory / (item['id']+'.jpg')
                if digest(path) != item['image_sha256']:
                    raise ValueError('Reviewed source image hash mismatch')
                inputs.append((path, {'source':'HaGRIDv2','source_url':'https://github.com/hukenovs/hagrid',
                    'source_label':'no_gesture','license':'HaGRID custom attribution/share-alike public license; datasets/HAGRID-LICENSE.pdf',
                    'source_file':item['member'],'source_sha256':item['image_sha256'],
                    'source_annotation':f"annotations/{item['split']}/{target}.json#{item['id']}",
                    'participant':'hagrid_'+item['annotation']['user_id'], 'split':{'val':'validation'}.get(item['split'],item['split']),
                    'category':category, 'review_index':f'{key}:{index}',
                    'crop_bbox':item['annotation']['bboxes'][item.get('hand_index',0)] if key=='hagrid_resting' else None}))
    source = review['aslyset']
    from .download_aslyset import SHA256, SOURCE
    archive = ROOT / 'datasets/public/aslyset/ASLYset.rar'
    if digest(archive) != SHA256:
        raise ValueError('ASLYset archive hash mismatch')
    sources['ASLYset'] = {'url':SOURCE,'archive_sha256':SHA256}
    members = [f'ASLYset/images/User{user}/fn/User{user}_FN_{number}.png' for user in source['users'] for number in source['image_numbers']]
    if any(not (archive.parent/member).exists() for member in members):
        root = archive.parent.resolve()
        if any(not (root/member).resolve().is_relative_to(root) for member in members):
            raise ValueError('ASLYset extraction path leaves the verified cache')
        listing = subprocess.check_output(['tar','-tvf',str(archive)],text=True).splitlines()
        regular = {line[line.index('ASLYset/'):] for line in listing if line.startswith('-') and 'ASLYset/' in line}
        if not set(members).issubset(regular):
            raise ValueError('ASLYset controls must be regular archive entries')
        subprocess.run(['tar','-xf',str(archive),'-C',str(root),'--safe-writes',*members],check=True)
    for user in source['users']:
        for number in source['image_numbers']:
            member = f'ASLYset/images/User{user}/fn/User{user}_FN_{number}.png'
            path = archive.parent / member
            inputs.append((path, {'source':'ASLYset','source_url':SOURCE,'source_label':'FN','license':'CC BY 4.0',
                'source_file':member,'source_sha256':digest(path),'participant':f'aslyset_user{user}',
                'split':{1:'train',4:'train',3:'validation',2:'test'}[user], 'category':source['category'],'crop_bbox':None}))
    records, failures = [], []
    options = vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(ROOT / 'datasets/videos/hand_landmarker.task')),
        running_mode=vision.RunningMode.IMAGE,num_hands=2,min_hand_detection_confidence=.6,min_hand_presence_confidence=.6)
    with vision.HandLandmarker.create_from_options(options) as detector:
        for path, metadata in inputs:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot decode verified source image')
            if metadata['crop_bbox']:
                # Isolate the source-annotated resting hand from the gesture hand.
                # This is a documented crop, not a new independent performance.
                x,y,w,h = metadata['crop_bbox']; height,width = image.shape[:2]
                margin = .25*max(w,h)
                image = image[max(0,int((y-margin)*height)):min(height,int((y+h+margin)*height)),
                              max(0,int((x-margin)*width)):min(width,int((x+w+margin)*width))].copy()
            result = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB,data=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)))
            base = {**metadata,'letter':'_','session':None,'hand':'unknown','lighting':'unknown','angle':'unknown',
                    'label_review':review['reviewer'],'extraction':f'MediaPipe {mp.__version__} IMAGE; one hand; confidence .6'}
            if len(result.hand_landmarks)!=1:
                failures.append({**base,'reason':'requires_one_detected_hand'})
                continue
            points = [v for p in result.hand_landmarks[0] for v in (p.x,p.y)]
            records.append({**base,'points':points})
    clean, audit = curate(records)
    if audit['conflicts'] or audit['invalid'] or audit['cross_participant_duplicates']:
        raise ValueError('Resolve reviewed unknown data conflicts before training')
    known = json.loads((ROOT/'datasets/prepared.json').read_text())+json.loads((ROOT/'datasets/public-balanced.json').read_text())
    if {r['fingerprint'] for r in known}&{r['fingerprint'] for r in clean}:
        raise ValueError('Reviewed non-letter coordinates conflict with a known-letter sample')
    # Detect same source people/images crossing the frozen partitions.
    for key in ['participant','source_sha256','fingerprint']:
        rows = clean if key=='fingerprint' else clean+failures
        values = {split:{row[key] for row in rows if row['split']==split} for split in ['train','validation','test']}
        if any(values[a]&values[b] for a,b in [('train','validation'),('train','test'),('validation','test')]):
            raise ValueError(f'Unknown benchmark split leakage: {key}')
    result = {'samples':clean,'tracking_failures':failures}
    (ROOT / 'datasets/unknown-hands.json').write_text(json.dumps(result,separators=(',',':'),allow_nan=False),encoding='utf-8')
    report = {'sources':sources,'review_sha256':digest(ROOT/'datasets/unknown-review.json'),'selected_images':len(inputs),
              'tracked_unique':len(clean),'tracking_failures':len(failures),'curation':audit,
              'by_split':dict(Counter(r['split'] for r in clean)), 'by_category':dict(Counter(r['category'] for r in clean)),
              'selection':[{k:r[k] for k in ['source','source_file','source_sha256','split','category','participant']} for _,r in inputs]}
    (ROOT / 'datasets/unknown-import-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k not in {'sources','selection','curation'}},indent=2))


if __name__=='__main__':
    main()
