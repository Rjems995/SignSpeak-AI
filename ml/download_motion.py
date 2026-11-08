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
    args = parser.parse_args()
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
    target = ROOT / "datasets" / "videos"
    target.mkdir(parents=True, exist_ok=True)
    def download(row):
        name = row["name"]
        letter = Path(name).parent.name
        path = target / letter / Path(name).name
        path.parent.mkdir(exist_ok=True)
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
            path.write_bytes(data)
        checksum = hashlib.sha256(path.read_bytes()).hexdigest()
        return {"file": f"{letter}/{path.name}", "upstream": name, "sha256": checksum, "bytes": path.stat().st_size}
    print(f"Downloading {len(selected)} clips ({sum(r['totalBytes'] for r in selected)/1e6:.1f} MB)", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        records = list(pool.map(download, selected))
    provenance = {"source": "https://www.kaggle.com/datasets/" + DATASET, "version": 1,
                  "license": "CC0: Public Domain (Kaggle dataset metadata)",
                  "participants": "Not provided. Clip IDs are not participant IDs.", "files": records}
    (ROOT / "datasets" / "motion-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Downloaded and hashed {len(records)} real motion clips.")


if __name__ == "__main__":
    main()
