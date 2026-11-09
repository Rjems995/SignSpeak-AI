"""Fetch and hash the CC BY 4.0 ASLYset archive; raw images stay out of Git."""
import hashlib
import json
import subprocess
from urllib.error import HTTPError
from pathlib import Path
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]
SOURCE = "https://data.mendeley.com/datasets/xs6mvhx6rh/1"
DOWNLOAD = "https://data.mendeley.com/public-files/datasets/xs6mvhx6rh/files/8c504c17-1765-4935-9f4f-4a2aa68cb9f0/file_downloaded?download=1"
SHA256 = "e50202c30dfff4795b88762b765bbeb357e184c13fe22b8f6e0d836d02235c80"
SIZE = 872115989


def main():
    directory = ROOT / "datasets" / "public" / "aslyset"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "ASLYset.rar"
    if not path.exists():
        partial = path.with_suffix(".part")
        print(f"Downloading ASLYset ({SIZE/1e6:.0f} MB)", flush=True)
        try:
            with urlopen(DOWNLOAD, timeout=180) as response, partial.open("wb") as output:
                while block := response.read(1024*1024):
                    output.write(block)
            partial.replace(path)
        except HTTPError as error:
            if error.code != 403:
                raise
            subprocess.run(["node", str(ROOT / "scripts" / "download-aslyset.mjs")], check=True)
    with path.open("rb") as archive:
        digest = hashlib.file_digest(archive, "sha256").hexdigest()
    if path.stat().st_size != SIZE or digest != SHA256:
        raise ValueError("ASLYset archive does not match its published size/hash. Do not import it.")
    provenance = {"dataset": "ASLYset", "author": "Miguel Rivera", "doi": "10.17632/xs6mvhx6rh.1",
                  "version": 1, "source": SOURCE, "license": "CC BY 4.0",
                  "license_url": "https://creativecommons.org/licenses/by/4.0/", "sha256": digest,
                  "bytes": SIZE, "published_participants": 4, "published_hand": "right",
                  "conditions": "Varying distance and position reported. Lighting/angle labels are not supplied; do not invent them.",
                  "attribution": "Rivera, Miguel (2019), ASLYset, Mendeley Data, V1, doi:10.17632/xs6mvhx6rh.1. Changes: extracted hand landmarks and curated source records."}
    (ROOT / "datasets" / "aslyset-provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print("ASLYset verified against the publisher's SHA-256.", flush=True)


if __name__ == "__main__":
    main()
