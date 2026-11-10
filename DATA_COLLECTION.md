# Better ASL training data

The `/capture` workshop collects real coordinate samples with consistent participant/session IDs and explicit actual-hand, lighting and viewing-angle labels. It starts with Q and prioritizes E/S, M/N and U/V, while suggesting underrepresented cells so the rest of the alphabet stays covered.

## Latest public-data expansion

The training-eligible public pool now combines ASLYset with the original raw images from [ASL-HG, Mendeley Data V1](https://data.mendeley.com/datasets/j4y5w2c8w9/1), CC BY 4.0. The ASL-HG publisher describes ten volunteers and varied indoor/outdoor acquisition. Original filenames retain P1–P10 signer IDs. They do **not** annotate each image's physical hand, lighting intensity, camera angle, or recording session. Those fields remain unknown, even when MediaPipe predicts handedness.

| Stage | Samples |
| --- | ---: |
| ASL-HG raw static images selected deterministically | 5,800 |
| ASL-HG valid single-hand detections | 5,788 |
| ASL-HG usable unique records after curation | 5,746 |
| ASL-HG Q examples | 300 |
| Eligible ASLYset + ASL-HG pool | 10,516 |
| Q examples in the combined pool | 499 |
| Balanced candidate: 14 source signer IDs × 24 static letters × 17 originals | 5,712 |
| Each static letter in that candidate, including Q | 238 |

Selection adds extra original Q, E, S, M, N, T, U, V, R and P images to the pool. Curation removes 14 within-signer duplicates and quarantines all 28 ASL-HG rows in 14 identical-coordinate groups assigned to different signer IDs. Some of these are byte-identical source images under different IDs; those IDs are publisher labels, not independently verified identities. ASLYset contributes another 18 within-signer duplicate removals. No conflicting labels or invalid geometry remain in this import. The balanced candidate uses the smallest available signer/letter cell (17), capped at 20; it never repeats images to fill a quota. This is **signer/letter balance**, not hand or condition balance.

Added files:

- `datasets/public-sources.json`: pinned archive sizes/hashes, original attributions, licenses, metadata limits and training eligibility.
- `datasets/asl_hg-landmarks.json`, `asl_hg-import-report.json`, `asl_hg-quality-audit.json`: actual coordinates, source image hashes, signer IDs, extraction failures and curation audit.
- `datasets/public-balanced.json`, `public-quality-audit.json`: the balanced candidate input and coverage gaps.
- `datasets/public-candidate-evaluation.json`: frozen holdout comparison, including per-letter, per-source and per-signer metrics.
- `signspeak/static/data/source-coverage.json`: measured import counts shown at `/capture`, with review-only sources and unknown hands explicit.

Two additional sources were downloaded, checksum-verified and inspected, but **excluded from training**:

| Source | Valid unique review samples | Reason |
| --- | ---: | --- |
| [SignAlphaSet V3](https://data.mendeley.com/datasets/8fmvr9m98w/3) | 1,495 | The related paper reports three participants and both physical hands, but individual files identify neither signer nor hand. It describes augmentation without per-file original/augmentation flags. No synthetic variation is counted as new physical-hand or environmental evidence. |
| [Massey MU HandImages ASL](https://mro.massey.ac.nz/server/api/core/bitstreams/09187662-5ebe-4563-8515-3d7e5e1d2a33/content), pinned JPEG mirror | 895 | Source filename codes identify bottom/top/left/right/diffuse **light direction**, not hand or camera angle. The original paper's section 7 now resolves redistribution terms; coordinates are bundled as a frozen lighting benchmark, outside training. Actual hand/per-image camera angles remain unknown. Detection failed for 780 of 1,675 selected images. |

SignAlphaSet's 41 detection failures and Massey's 780 failures are recorded in their import reports. SignAlphaSet static coordinates remain under ignored `artifacts/public-review/`; the licensed Massey benchmark is now at `datasets/massey-landmarks.json`. Both original archives remain ignored under `datasets/public/`. These sources do not increase the frozen training candidate's signer count. MediaPipe `observed_hand` and confidence remain estimates: unknown mirroring and detector errors prevent treating them as physical-hand ground truth. [CONDITION_DATA.md](CONDITION_DATA.md) records the latest evidence review, paired camera diagnostic and remaining left-hand/joint-annotation gaps.

## New candidate evaluation

The candidate uses 8 source signers for training (3,264 originals), 3 for validation (1,224), and 3 for testing (1,224). Splits are participant-disjoint and stratified by source; both sources occur in every split. The test signer IDs are `asl_hg_p10`, `asl_hg_p8`, and `aslyset_user2`. Parameters and rejection thresholds were not tuned on test results. Original/augmented copies cannot cross these splits; augmentation occurs only after selecting training originals.

| Model on the same frozen test | Raw accuracy | Correct after 0.85 confidence/distance rejection |
| --- | ---: | ---: |
| Current published static model | 57.60% | 45.51% |
| New public-data candidate | 89.13% | 80.23% |

The candidate accepts 83.82% of test samples; 95.71% of accepted predictions are correct. Rejected letters count as failures in the gated accuracy above. Q recall is 100% raw and 90.20% gated on **51 test images**, versus 0% for the current model on these same Q images. S raw recall improves to 94.12%. E raw recall regresses from 98.04% to 78.43%, and gated M recall regresses from 47.06% to 29.41%. T raw recall remains 35.29%; R and M gated recall are each 29.41%. These weaknesses prevent interpreting the overall score as reliable recognition of every letter.

Python export verification and the app's TensorFlow.js implementation agree on all 1,224 held-out examples, including normalization and rejection decisions (maximum probability difference below 0.000005). The static candidate is saved under ignored `artifacts/public-candidate/`. **The live static model remains unchanged.** This static import obtained no verified physical-left-hand test cohort, per-image lighting/camera-angle groups, independent sessions, expert-reviewed new labels, unsupported-pose false-acceptance benchmark, or live webcam benchmark. Public data improves the static training pool without completing those collection targets. The separate J/Z model has since been expanded and installed; see [MOTION_DATA.md](MOTION_DATA.md).

## Reproduce the expansion

Install `requirements-motion.txt` and obtain the MediaPipe `datasets/videos/hand_landmarker.task` model as described in README.md. The scripts download public archives into the ignored cache, verify complete SHA-256 digests, read only validated image members, and never execute downloaded source code. Sampling uses deterministic filename hashes, not classifier confidence. The first two ASLYset commands below may be skipped if that verified import already exists.

```powershell
.\.venv\Scripts\python.exe -m ml.download_aslyset
.\.venv\Scripts\python.exe -m ml.import_aslyset
node scripts/download-public-source.mjs asl_hg
.\.venv\Scripts\python.exe -m ml.import_public asl_hg
.\.venv\Scripts\python.exe -m ml.audit_dataset datasets/asl_hg-landmarks.json --kind static --output datasets/asl_hg-quality-audit.json
.\.venv\Scripts\python.exe -m ml.build_public_candidate
.\.venv\Scripts\python.exe -m ml.train --dataset datasets/public-balanced.json --require-participant-split --output artifacts/public-candidate
.\.venv\Scripts\python.exe -m ml.evaluate --model-directory artifacts/public-candidate
.\.venv\Scripts\python.exe -m ml.compare_public_models
node scripts/verify-candidate.mjs artifacts/public-candidate
```

To reproduce the separate local reviews and the four-source UI audit:

```powershell
node scripts/download-public-source.mjs signalphaset
.\.venv\Scripts\python.exe -m ml.import_public signalphaset --per-cell 64 --priority-extra 0
node scripts/download-public-source.mjs massey
.\.venv\Scripts\python.exe -m ml.import_public massey
.\.venv\Scripts\python.exe -m ml.report_public_coverage
```

## Initial ASLYset import

ASLYset, by Miguel Rivera (2019), is available under CC BY 4.0 at [doi:10.17632/xs6mvhx6rh.1](https://data.mendeley.com/datasets/xs6mvhx6rh/1). Its source folders identify four signers. The publisher describes right-hand images at varied distances/positions. It does not provide per-image lighting, angle, or session labels. Those fields remain `unknown`/null in this project.

| Stage | Real sample count |
| --- | ---: |
| Source static-letter images (24 letters, 4 signers, 50 per cell) | 4,800 |
| Exactly one hand detected with valid geometry | 4,788 |
| Unique usable coordinate samples after curation | 4,770 |
| Q examples among those unique samples | 199 |
| Balanced subset (40 per signer per letter, right hand only) | 3,840 |
| Each static letter in the balanced subset, including Q | 160 |

Files:

- `datasets/aslyset-landmarks.json`: extracted real coordinates, original signer IDs, source image names/hashes and honest metadata.
- `datasets/aslyset-balanced.json`: 3,840 selected original records with equal signer/letter/hand counts and prepared features. No images were copied or augmented to fill a quota.
- `datasets/aslyset-provenance.json`: source, attribution, CC BY 4.0 license and the publisher's archive hash.
- `datasets/aslyset-import-report.json`: actual extraction counts and all 12 rejected image records.
- `datasets/aslyset-quality-audit.json`: deduplication and remaining diversity gaps.
- `datasets/signn-quality-audit.json`: seven conflicting-coordinate label groups found in the legacy source. All records in conflicting groups are quarantined by curation; labels are never resolved by keeping whichever letter occurs first.

Raw source images and the 872 MB archive remain under ignored `datasets/public/`. No new camera footage from a human signer was accessed by this development task. ASLYset contains no J/Z motion clips and no documented left-hand examples.

## Collect the missing diversity

1. Open `/capture`, then enable the camera yourself. Ask the signer to opt in to coordinate collection and local export. No footage is recorded or uploaded by the collector.
2. Assign a pseudonymous participant ID. Reuse it for that same person on every visit; never give each frame or session a new participant ID.
3. Assign a session ID for the visit. The form generates one, and you may replace it with a meaningful pseudonym.
4. Select the actual hand used, lighting and viewing angle. The preview is mirrored, but that does not change which physical hand the person is using. Keep thumb placement and meaningful letter orientation correct while varying the view slightly.
5. Check the intended letter against the ASL reference. Collector confirmation is not expert certification; a knowledgeable ASL signer should review labels. The model's prediction does not determine the recorded label.
6. Capture individual static examples, or use **Capture 5 with pauses**. Every accepted sample must come from a fresh tracked hand. Identical normalized coordinates are rejected. Change the real viewpoint/pose slightly without changing the letter.
7. For J/Z, choose the letter and record a complete movement. Signer/condition labels are frozen at the start of the recording. Stopped or interrupted recordings do not create partial examples.
8. Use **Suggest next letter** to cover missing signer/hand/letter cells. Use **Undo last sample** if a label was wrong.
9. Download static coordinates and J/Z sequences separately. Import previous version 3 exports to see aggregate coverage; original signer IDs and conditions are retained, and duplicates are not counted again. Samples are not automatically saved in browser storage.
10. Withdrawing consent stops the camera, cancels collection and clears every in-memory sample. Remove downloaded copies separately if the signer requests withdrawal after export.

The starting plan is ten consenting signers, twenty distinct examples per letter per physical hand, at least two independent sessions, and coverage of bright, soft and dim lighting plus frontal/slightly left/slightly right views. These are collection targets, not proof of accuracy. Add extra real Q and confused-letter examples after establishing balanced baseline coverage. Comfortable signing takes precedence; do not force a hand position or make a person use a hand they cannot use comfortably.

Counts based on collector-provided IDs/conditions do not verify a person's identity or physical conditions. Public-source counts are shown separately from the private collection in the current tab. Mirrored coordinates, duplicated frames and synthetic jitter are never counted as new people, new physical hands, or new environmental evidence.

## Audit and prepare your exports

Keep private exports under ignored `datasets/private/`. Metadata survives preparation and balancing.

```powershell
.\.venv\Scripts\python.exe -m ml.audit_dataset datasets/private/signer01-static.json datasets/private/signer02-static.json --kind static --output datasets/private/coverage.json
.\.venv\Scripts\python.exe -m ml.prepare datasets/private/signer01-static.json datasets/private/signer02-static.json --output datasets/private/prepared.json
.\.venv\Scripts\python.exe -m ml.balance_dataset datasets/private/prepared.json --hands left right --per-cell 20 --output datasets/private/balanced.json
```

The balancer stops if any required signer/letter/hand cell lacks enough real examples. It selects existing records only. The audit counts unknown metadata as unknown. Add `--require-targets` to the audit command when you want it to fail until the collection plan is met.

If the audit finds conflicting labels, review the source labels. To exclude every conflicting group rather than silently selecting a label, run `ml.prepare` with `--drop-conflicts`. Invalid geometry must be removed or recollected separately.

Motion exports use `ml.prepare_motion`, followed by `ml.train_motion`. Keep static samples and movements in separate datasets. Future signer-independent training can explicitly refuse unidentified sample splits:

```powershell
.\.venv\Scripts\python.exe -m ml.train --dataset datasets/private/balanced.json --require-participant-split --output artifacts/private-candidate
.\.venv\Scripts\python.exe -m ml.evaluate --model-directory artifacts/private-candidate
```

Do not merge unidentified legacy rows into a claimed participant-independent evaluation. More than three signers are preferable, and every split must contain the required classes.

## Reproduce the public import and candidate

Install `requirements-motion.txt` for MediaPipe and image decoding. Windows `tar.exe` supports this archive. The Node fallback handles the publisher's download endpoint; hashes are checked before any import. The importer only extracts validated regular image paths inside `datasets/public/aslyset/`.

```powershell
.\.venv\Scripts\python.exe -m ml.download_aslyset
.\.venv\Scripts\python.exe -m ml.import_aslyset
.\.venv\Scripts\python.exe -m ml.audit_dataset datasets/aslyset-landmarks.json --kind static --output datasets/aslyset-quality-audit.json
.\.venv\Scripts\python.exe -m ml.balance_dataset datasets/aslyset-landmarks.json --hands right --per-cell 40 --output datasets/aslyset-balanced.json
.\.venv\Scripts\python.exe -m ml.train --dataset datasets/aslyset-balanced.json --require-participant-split --output artifacts/aslyset-candidate
.\.venv\Scripts\python.exe -m ml.evaluate --model-directory artifacts/aslyset-candidate
```

A measured review summary is retained at `datasets/aslyset-candidate-evaluation.json`. A review candidate was trained on two signers (1,920 samples), validated on a third (960), and tested on the fourth (960). Its raw static accuracy is **87.8125%**, with **78.75%** correct after the default rejection gate. This is a four-signer, right-hand-only source evaluation. It is not directly comparable with the old model's random-sample test, and it does not establish left-hand, webcam, lighting or angle robustness. S recall is 0% and D recall is 10% on this particular held-out signer, so the overall score does not mean every letter is reliable. Prioritize review and additional real examples for those letters too. No unsupported-pose test set was added, so false acceptance remains unmeasured.

The candidate is under ignored `artifacts/aslyset-candidate/`; the app's published model artifacts and their reported accuracy were not replaced. Review new consented webcam data and remaining coverage gaps before choosing a model for the live app.
