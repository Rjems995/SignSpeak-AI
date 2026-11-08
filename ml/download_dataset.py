"""Fetch the pinned, public SigNN coordinate dataset and its upstream license."""
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

REVISION = "40bd651899e09a0099767457194062f77f774156"
ROOT = Path(__file__).resolve().parents[1]
BASE = f"https://raw.githubusercontent.com/AriAlavi/SigNN/{REVISION}/"


def main():
    target = ROOT / "datasets"
    target.mkdir(exist_ok=True)
    for remote, local in [("scripts/mediapipe_data_creation/training_data.json", "signn.json"), ("LICENSE", "SIGNN-LICENSE.txt")]:
        with urlopen(BASE + remote, timeout=60) as response:
            data = response.read()
        (target / local).write_bytes(data)
        print(f"Downloaded {local}: {len(data):,} bytes; sha256={hashlib.sha256(data).hexdigest()}")
    (target / "provenance.json").write_text(json.dumps({
        "source": "https://github.com/AriAlavi/SigNN", "revision": REVISION,
        "license": "Apache-2.0 (upstream repository license; retained alongside dataset)",
        "format": "24 static ASL labels; lists of 21 image-space (x,y) landmark pairs",
        "participants": "Not supplied by upstream. No signer-independent accuracy claim is possible.",
        "limitations": "Small historical MediaPipe dataset; duplicates; unknown signer/session grouping; landmark-extractor domain shift possible.",
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

