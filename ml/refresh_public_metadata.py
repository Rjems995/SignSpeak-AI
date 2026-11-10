"""Join corrected publisher metadata to existing verified extraction; never infer fields."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path
from .import_public import ROOT, SOURCES, source_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", choices=SOURCES)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source = SOURCES[args.source]
    archive = ROOT / "datasets/public" / args.source / source["archive"]
    with archive.open("rb") as stream:
        if hashlib.file_digest(stream, "sha256").hexdigest() != source["sha256"]:
            raise ValueError("Archive hash mismatch")
    data = json.loads(args.input.read_text(encoding="utf-8"))
    refreshed = []
    with zipfile.ZipFile(archive) as zipped:
        for record in data["samples"]:
            metadata = source_metadata(args.source, record["source_file"])
            if not metadata or hashlib.sha256(zipped.read(record["source_file"])).hexdigest() != record["source_sha256"]:
                raise ValueError("Record does not match its original source image")
            refreshed.append({**record, **metadata})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({**data, "samples": refreshed}, separators=(",", ":")), encoding="utf-8")
    report_path = ROOT / f"datasets/{args.source}-import-report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.update(source=source, metadata_refresh={"coordinates_changed": False, "source_images_reverified": len(refreshed),
                  "evidence": "Original author paper; per-image filename illumination labels; capture protocol kept separate from camera angle"})
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Refreshed source annotations for {len(refreshed)} hash-verified originals; coordinates unchanged.")


if __name__ == "__main__":
    main()
