"""Create local review contact sheets; no labels inferred from model outputs."""
import json
import sys
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
directory = ROOT / 'datasets/public/hagrid'
resting = '--resting' in sys.argv
rows = json.loads((directory / ('resting-selection.json' if resting else 'review-selection.json')).read_text())
output = ROOT / 'artifacts/rejection'
for start in range(0, len(rows), 40):
    sheet = Image.new('RGB', (1000, 8*155), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows[start:start+40]):
        image = Image.open(directory / (row['id'] + '.jpg')).convert('RGB')
        x, y, w, h = row['annotation']['bboxes'][row.get('hand_index',0)]
        margin = .25*max(w,h)
        width, height = image.size
        crop = image.crop((max(0,int((x-margin)*width)),max(0,int((y-margin)*height)),min(width,int((x+w+margin)*width)),min(height,int((y+h+margin)*height))))
        crop.thumbnail((195,128))
        sheet.paste(crop, ((i%5)*200,(i//5)*155+22))
        draw.text(((i%5)*200+2,(i//5)*155+2),f"{start+i}: {row['split']}",fill='black')
    sheet.save(output / f'{"resting" if resting else "hagrid"}-{start//40}.jpg')
