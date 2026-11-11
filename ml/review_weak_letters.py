"""Export validation disagreements for ASL review without changing source labels."""
import json
import argparse
import math
from pathlib import Path
import numpy as np
from .data_quality import ROOT
from .static_features import record_features
from .compare_public_models import predict, probabilities


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--images',action='store_true',help='Render source-verified local hand crops under ignored artifacts/')
    args = parser.parse_args()
    model_path = ROOT/'datasets/candidates/focus-kernel-v4/classifier.json'
    model = json.loads(model_path.read_text())
    rows = [r for r in json.loads((ROOT/'datasets/focus-depth.json').read_text())['samples'] if r['split']=='validation' and r['letter'] in 'EMRT']
    x = np.array([record_features(r,model) for r in rows])
    raw, gated = predict(model,x)
    probability = probabilities(model,x)
    samples = []
    for i,row in enumerate(rows):
        if gated[i]==row['letter']:
            continue
        samples.append({k:row[k] for k in ['letter','source','source_url','source_file','source_sha256','participant','fingerprint','split']} |
            {'raw_prediction':str(raw[i]),'accepted':None if gated[i]=='_' else str(gated[i]),
             'expected_letter_probability':float(probability[i,model['labels'].index(row['letter'])]),
             'status':'needs_ASL_expert_review','label_changed':False})
    result = {'scope':'Validation-only disagreements; no test samples selected for relabeling.',
        'warning':'Model disagreement is not evidence that a publisher label is wrong. Review the original handshape with a knowledgeable ASL signer; preserve labels until independently checked.',
        'samples':samples}
    path = ROOT/'datasets/weak-letter-review.json'
    path.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(f'Queued {len(samples)} validation examples for review; no source labels or training eligibility changed.')
    if args.images:
        import cv2
        from PIL import Image, ImageDraw
        from .import_static_depth import SourceImages
        lookup = {r['fingerprint']:r for r in rows}
        sheet = Image.new('RGB',(960,220*math.ceil(len(samples)/4)),color='white')
        draw = ImageDraw.Draw(sheet)
        sources = SourceImages(samples)
        try:
            for i,item in enumerate(samples):
                row = lookup[item['fingerprint']]
                pixels = cv2.imdecode(np.frombuffer(sources.read(item),dtype=np.uint8),cv2.IMREAD_COLOR)
                height,width = pixels.shape[:2]
                xy = np.array(row['points']).reshape(21,3)[:,:2]*[width,height]
                lo,hi = xy.min(axis=0),xy.max(axis=0)
                pad = .2*max(hi-lo)
                crop = pixels[max(0,int(lo[1]-pad)):min(height,math.ceil(hi[1]+pad)),max(0,int(lo[0]-pad)):min(width,math.ceil(hi[0]+pad))]
                image = Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB))
                image.thumbnail((232,172))
                left,top = (i%4)*240,(i//4)*220
                sheet.paste(image,(left+4,top+43))
                draw.text((left+4,top+3),f"{i+1}. Publisher {row['letter']} | model {item['accepted'] or 'rejected'}",fill='black')
                draw.text((left+4,top+20),row['participant'],fill='black')
        finally:
            sources.close()
        output = ROOT/'artifacts/weak-letter-review.jpg'
        output.parent.mkdir(parents=True,exist_ok=True)
        sheet.save(output,quality=95)
        print(f'Source-image review sheet stays local and ignored: {output}')


if __name__=='__main__':
    main()
