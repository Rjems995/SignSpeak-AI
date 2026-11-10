"""Download a bounded, reproducible CC0 J/Z clip subset from Kaggle's public API."""
import argparse
import concurrent.futures
import hashlib
import json
import io
import zipfile
from pathlib import Path
from urllib.parse import quote
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
DATASET = "signnteam/asl-sign-language-alphabet-videos-j-z"
BASE = "https://www.kaggle.com/api/v1/datasets/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-letter", type=int, default=24)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/videos")
    parser.add_argument("--provenance", type=Path, default=ROOT / "datasets/motion-provenance.json")
    parser.add_argument("--include-baseline", action="store_true", help="Retain the frozen original clips in an expanded source manifest")
    args = parser.parse_args()
    previous = json.loads(args.provenance.read_text(encoding="utf-8")) if args.provenance.exists() else {"files": []}
    pinned = {row["upstream"]: row["sha256"] for row in previous["files"]}
    if args.per_letter < 5:
        parser.error("--per-letter must be at least 5 for separate training/validation/test clips")
    with urlopen(BASE + "list/" + DATASET + "?pageSize=1000", timeout=60) as response:
        listing = json.load(response)
    files = listing["datasetFiles"]
    # The API caps page size; follow its cursor rather than assuming all files fit.
    while listing.get("nextPageToken") or listing.get("nextPageTokenNullable"):
        token = listing.get("nextPageToken") or listing["nextPageTokenNullable"]
        with urlopen(BASE + "list/" + DATASET + "?pageSize=1000&pageToken=" + quote(token), timeout=60) as response:
            listing = json.load(response)
        files.extend(listing["datasetFiles"])
    selected = []
    for letter in ["J", "Z"]:
        rows = sorted((r for r in files if f"/{letter}/" in r["name"] and r["name"].endswith(".avi")), key=lambda r: int(Path(r["name"]).stem))
        if len(rows) < args.per_letter:
            raise ValueError("Not enough public clips for the requested subset")
        indices = [round(i * (len(rows)-1)/(args.per_letter-1)) for i in range(args.per_letter)]
        selected.extend(rows[i] for i in indices)
    if args.include_baseline:
        original = json.loads((ROOT / "datasets/motion-provenance.json").read_text())
        wanted = {r["upstream"] for r in original["files"]}
        selected = list({r["name"]: r for r in selected+[r for r in files if r["name"] in wanted]}.values())
    target = args.output_dir
    target.mkdir(parents=True, exist_ok=True)
    def download(row):
        name = row["name"]
        letter = Path(name).parent.name
        path = target / letter / Path(name).name
        path.parent.mkdir(exist_ok=True)
        # Reuse only the exact known baseline file, checking its saved digest.
        baseline = ROOT / "datasets/videos" / letter / Path(name).name
        if args.include_baseline and not path.exists() and baseline.exists():
            import shutil
            original_row = next((r for r in original["files"] if r["upstream"] == name), None)
            if original_row and hashlib.sha256(baseline.read_bytes()).hexdigest() == original_row["sha256"]:
                shutil.copyfile(baseline, path)
        if not path.exists() or path.stat().st_size != row["totalBytes"]:
            url = BASE + "download/" + DATASET + "/" + quote(name, safe="") + "?datasetVersionNumber=1"
            with urlopen(url, timeout=120) as response:
                data = response.read()
            if data[:2] == b"PK":
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    matches = [item for item in archive.namelist() if Path(item).name == path.name]
                    if len(matches) != 1:
                        raise ValueError(f"Expected exactly one matching clip in {name}")
                    data = archive.read(matches[0])
            if data[:4] not in {b"RIFF", b"\x1aE\xdf\xa3"}:
                raise ValueError(f"Expected a video file for {name}; got {data[:50]!r}")
            if len(data) != row["totalBytes"]:
                raise ValueError(f"Incomplete source video: {name}")
            path.write_bytes(data)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        if name in pinned and checksum != pinned[name]:
            raise ValueError(f"Cached video differs from the pinned source manifest: {name}")
        return {"file": f"{letter}/{path.name}", "upstream": name, "sha256": checksum, "bytes": path.stat().st_size}
    print(f"Downloading {len(selected)} clips ({sum(r['totalBytes'] for r in selected)/1e6:.1f} MB)", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = []
        for row in pool.map(download, selected):
            records.append(row)
            if len(records) % 20 == 0:
                print(f"Verified {len(records)}/{len(selected)} clips", flush=True)
    provenance = {"source": "https://www.kaggle.com/datasets/" + DATASET, "version": 1,
                  "license": "CC0: Public Domain (Kaggle dataset metadata)",
                  "participants": "Not provided. Clip IDs are not participant IDs.", "files": records}
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Downloaded and hashed {len(records)} real motion clips.")


if __name__ == "__main__":
    main()
