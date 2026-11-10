"""Fetch bounded official HaGRID ZIP members for review, never extract paths."""
import hashlib
import io
import json
import struct
import urllib.request
import zipfile
import zlib
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "datasets/public/hagrid"
BASE = "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid_v2/"
LOCK_PATH = ROOT.parents[1] / 'unknown-source-lock.json'


def request(url, start=None, end=None):
    headers = {"Range": f"bytes={start}-{end}"} if start is not None else {"Range": "bytes=-65536"}
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        if response.status != 206:
            raise ValueError("Server must honor bounded range requests")
        data = response.read()
        total = int(response.headers["Content-Range"].split("/")[-1])
        return data, total


def index(url):
    tail, total = request(url)
    end = tail.rfind(b"PK\x05\x06")
    if end < 0:
        raise ValueError('ZIP end record missing')
    directory_bytes = zipfile._EndRecData(io.BytesIO(tail))[zipfile._ECD_SIZE]
    if directory_bytes > 10_000_000:
        raise ValueError('ZIP index exceeds the bounded review budget')
    # Include ZIP64 end/locator records as well as the ordinary end record.
    if directory_bytes + len(tail) - end + 128 > len(tail):
        tail, total = request(url, total - directory_bytes - (len(tail) - end) - 128, total - 1)
    # zipfile adjusts offsets assuming this tail starts at zero; restore them.
    archive = zipfile.ZipFile(io.BytesIO(tail))
    return {i.filename: (i, i.header_offset + total - len(tail)) for i in archive.infolist()}


def member(url, entry):
    info, offset = entry
    header, _ = request(url, offset, offset + 29)
    fields = struct.unpack("<4s5H3I2H", header)
    if fields[0] != b"PK\x03\x04" or info.file_size > 100_000_000 or info.compress_size > 20_000_000:
        raise ValueError("Unexpected or excessive ZIP member")
    start = offset + 30 + fields[-2] + fields[-1]
    data, _ = request(url, start, start + info.compress_size - 1)
    decoded = zlib.decompress(data, -15) if info.compress_type == zipfile.ZIP_DEFLATED else data
    if len(decoded) != info.file_size or zlib.crc32(decoded) != info.CRC:
        raise ValueError("ZIP size/CRC mismatch")
    return decoded


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resting', action='store_true')
    args = parser.parse_args()
    target = 'four' if args.resting else 'no_gesture'
    lock = json.loads(LOCK_PATH.read_text()) if LOCK_PATH.exists() else None
    ROOT.mkdir(parents=True, exist_ok=True)
    url = BASE + "annotations_with_landmarks/annotations.zip"
    entries = index(url)
    manifest = {}
    for split in ["train", "val", "test"]:
        name = f"annotations/{split}/{target}.json"
        path = ROOT / f"{split}-{target}.json"
        data = path.read_bytes() if path.exists() else member(url, entries[name])
        if lock and hashlib.sha256(data).hexdigest()!=lock['annotations'][path.name]['sha256']:
            raise ValueError('Pinned annotation hash mismatch; review upstream changes explicitly')
        path.write_bytes(data)
        manifest[path.name] = {"url": url, "member": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
        rows = json.loads(data)
        print(split, len(rows), flush=True)
    (ROOT / f"{target}-download-manifest.json").write_text(json.dumps(manifest, indent=2))
    url = (BASE + "hagrid_v2_zip/no_gesture.zip" if not args.resting else
           "https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/four.zip")
    entries = index(url)
    by_stem = {Path(name).stem: (name, entry) for name, entry in entries.items() if name.endswith('.jpg')}
    selection = []
    counts = [('train', 30), ('val', 15), ('test', 30)] if args.resting else [('train', 100), ('val', 40), ('test', 60)]
    for split, count in counts:
        rows = json.loads((ROOT / f'{split}-{target}.json').read_text())
        people = set()
        for key in sorted(rows, key=lambda key: hashlib.sha256(key.encode()).hexdigest()):
            row = rows[key]
            # Single-hand examples match the app. Deterministic selection, before inference.
            hand = row['labels'].index('no_gesture') if 'no_gesture' in row['labels'] else -1
            if hand < 0 or (not args.resting and len(row['labels']) != 1) or not row['hand_landmarks'][hand] or row['user_id'] in people:
                continue
            if key not in by_stem:
                raise ValueError('Missing official image')
            people.add(row['user_id'])
            name, entry = by_stem[key]
            selection.append({'id': key, 'split': split, 'member': name, 'hand_index': hand, 'annotation': row})
            if len(people) >= count:
                break
        print('selected', split, len(people), flush=True)
    def download(row):
        path = ROOT / (row['id'] + '.jpg')
        data = path.read_bytes() if path.exists() else member(url, by_stem[row['id']][1])
        if lock and (row['id'] not in lock['images'] or hashlib.sha256(data).hexdigest()!=lock['images'][row['id']]):
            raise ValueError('Pinned source image hash mismatch; review upstream changes explicitly')
        path.write_bytes(data)
        row['image_sha256'] = hashlib.sha256(data).hexdigest()
    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(download, selection))
    (ROOT / ('resting-selection.json' if args.resting else 'review-selection.json')).write_text(json.dumps(selection, separators=(',', ':')))
    print('downloaded', len(selection), 'official images for review', flush=True)


if __name__ == "__main__":
    main()
