"""Prepare/extract a small licensed-for-research benchmark outside training/redistribution."""
import argparse
import hashlib
import json
import re
import zipfile
import xml.etree.ElementTree as ET
from collections import Counter
from .data_quality import ROOT
from .import_motion_expansion import extract_video
from .motion_audit import ranges

DIRECTORY = ROOT / "datasets/public/asllvd"
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def prepare():
    source = json.loads((ROOT / "datasets/motion-sources.json").read_text())["asllvd"]
    xlsx = DIRECTORY / source["archive"]
    if hashlib.sha256(xlsx.read_bytes()).hexdigest() != source["sha256"]:
        raise ValueError("ASLLVD annotation hash changed")
    selected = []
    with zipfile.ZipFile(xlsx) as zipped:
        strings = ["".join(row.itertext()) for row in ET.fromstring(zipped.read("xl/sharedStrings.xml"))]
        sheet = ET.fromstring(zipped.read("xl/worksheets/sheet1.xml"))
        for row in sheet.findall(".//s:sheetData/s:row", NS):
            values, formulas = {}, {}
            for cell in row:
                column = re.sub(r"[0-9]+", "", cell.get("r"))
                value = cell.find("s:v", NS)
                text = value.text if value is not None else ""
                values[column] = strings[int(text)] if cell.get("t") == "s" and text else text
                formula = cell.find("s:f", NS)
                if formula is not None:
                    formulas[column] = formula.text
            if values.get("D") not in {"J", "Z", "I", "D", "L"} or "L" not in formulas:
                continue
            match = re.fullmatch(r'HYPERLINK\("http://csr\.bu\.edu/ftp/asl/asllvd/demos/verify_start_end_handshape_annotations//test_auto_move//signs_mov_separ_signers/([A-Za-z]+_[0-9]+\.mov)", "MOV"\)', formulas["L"])
            if not match:
                raise ValueError("Unexpected source movie URL")
            filename = match[1]
            selected.append({"source_label": values["D"], "letter": values["D"] if values["D"] in {"J", "Z"} else "_",
                             "source_consultant": values["C"], "source_row": row.get("r"), "file": filename,
                             "url": "https://csr.bu.edu/ftp/asl/asllvd/demos/verify_start_end_handshape_annotations/test_auto_move/signs_mov_separ_signers/"+filename,
                             "session": values["M"], "source_start_frame": int(values["O"]), "source_end_frame": int(values["P"])})
    people = sorted({r["source_consultant"] for r in selected})
    for row in selected:
        row["participant"] = f"asllvd_{people.index(row['source_consultant'])+1:02d}"
    (DIRECTORY / "selection.json").write_text(json.dumps({"source": source, "files": selected}, indent=2), encoding="utf-8")
    print(json.dumps({"selected": len(selected), "participants": len(people), "classes": dict(Counter(r['source_label'] for r in selected))}, indent=2))


def extract():
    manifest = json.loads((DIRECTORY / "selection.json").read_text())
    records, failures = [], []
    for row in manifest["files"]:
        path = DIRECTORY / row["file"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Research clip hash changed")
        # The MOV has a 50-frame source buffer. Positive benchmarks use the
        # annotated sign only, excluding preparation/retraction. Controls retain
        # 10 actual context frames each side to test non-J/Z transitions too.
        bounds = (50, 50+row["source_end_frame"]-row["source_start_frame"]) if row["letter"] != "_" else (40, 60+row["source_end_frame"]-row["source_start_frame"])
        try:
            metadata = {"letter": row["letter"], "source_label": row["source_label"], "clip": "asllvd/"+row["file"],
                        "participant": row["participant"], "session": row["session"], "hand": "right", "hand_source": "publisher_recording_protocol",
                        "source": "ASLLVD research benchmark", "source_sha256": row["sha256"], "training_eligible": False,
                        "source_url": manifest["source"]["url"], "license": manifest["source"]["license"], "source_row": row["source_row"],
                        "extraction": "frontal panel; raised hand; 60-fps capture time; annotated J/Z bounds, 10-frame context only for controls"}
            records.append(extract_video(path, metadata, fps_override=60, bounds=bounds, frontal_panel=True, keep_stationary_control=True))
        except ValueError as error:
            failures.append({"clip": row["file"], "letter": row["letter"], "reason": str(error)})
    out = ROOT / "artifacts/motion-research"
    out.mkdir(parents=True, exist_ok=True)
    (out / "asllvd.json").write_text(json.dumps({"samples": records, "failures": failures}, separators=(",", ":")), encoding="utf-8")
    print(json.dumps({"accepted": len(records), "failures": failures, "classes": dict(Counter(r["letter"] for r in records)), "ranges": ranges(records)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "extract"])
    args = parser.parse_args()
    prepare() if args.action == "prepare" else extract()
