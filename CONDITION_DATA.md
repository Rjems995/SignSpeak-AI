# Source-verified hand, lighting and camera annotations

**The requested gaps are only partially addressed.** The reviewed public sources do not supply a usable ASL alphabet subset jointly annotated with actual left/right hand, physical illumination and camera view. The project now preserves the available source annotations and reports the missing ones explicitly. No live model weights or training splits changed.

## Available annotation evidence

| Source | Real samples/views audited | Hand used | Lighting | Camera |
| --- | ---: | --- | --- | --- |
| MU HandImages ASL / Massey | 895 usable static coordinate samples from 1,675 source images | Unknown | Five source-coded light directions per image | Fixed tripod/perpendicular setup documented for the dataset; per-image angle unknown |
| ASLLVD research subset | 10 views of five existing J/Z performances | Unknown per token; source describes right-hand-dominant signers | Physical illumination unknown; delivered video brightness is normalized | Five frontal and five signer-right profile views |
| Existing ASLYset pool | 4,770 unique samples | Publisher explicitly states right-hand images | Unknown per image | Unknown per image |
| Existing ASL-HG pool | 5,746 unique samples | Unknown | Unknown per image | Unknown per image |

**New verified left-hand samples: 0. Samples jointly labeled with actual hand, physical lighting and camera view: 0.** Lighting direction does not establish bright/soft/dim intensity or a lux value. Camera position/setup does not establish an exact per-image viewing angle. No numeric camera angle was invented. The schema distinguishes participant dominance from the hand actually used: right-hand dominance alone cannot populate a per-token physical-hand annotation. The previous local ASLLVD metadata shortcut has been corrected to keep actual hand unknown; coordinate data and recognition metrics are unchanged.

## Massey: resolved source terms and lighting labels

The [original author paper](https://mro.massey.ac.nz/server/api/core/bitstreams/09187662-5ebe-4563-8515-3d7e5e1d2a33/content), Barczak et al. (2011), documents the mounted lamps and filename convention in sections 3/4. The mirror's `dif` code corresponds to diffuse illumination; `left` and `right` refer to the light, not the physical hand or camera. The paper also describes a perpendicular capture setup while allowing slight hand tilt. `camera_setup` records that protocol; `angle` stays unknown.

Section 7 grants attributed use/distribution, requires its notice to accompany redistributed copies, and prohibits selling the images. The source is not assigned an invented Creative Commons license. The verbatim author-required notice is retained in `datasets/MASSEY-TERMS.txt`, with attribution in NOTICE.md. Original PDF/text hashes are pinned in `datasets/condition-documents.json`.

The previous unclear-rights decision is superseded by this primary evidence. `datasets/massey-landmarks.json` now bundles the existing **895 original coordinate records**, with corrected license metadata and per-field evidence links/locators. Every source-image hash was rechecked against the pinned archive; coordinates did not change. The images remain in the ignored cache. This source is reserved for a lighting benchmark, not added to the training pool. `ml.train` rejects benchmark/review-only records to preserve that separation.

| Light direction | Source images | Valid tracked coordinates | Detection failures |
| --- | ---: | ---: | ---: |
| Bottom | 480 | 260 | 220 |
| Diffuse | 475 | 285 | 190 |
| Left | 240 | 116 | 124 |
| Right | 240 | 83 | 157 |
| Top | 240 | 151 | 89 |

Detection fails for **780/1,675** selected source images. Evaluating only the 895 successfully tracked images would hide those failures. The new benchmark counts them as recognition failures and reports class support per condition. End-to-end gated recognition is 36.3% for the installed static model and 44.9% for the existing frozen static candidate. Unequal class/signer composition, segmented backgrounds and hand tilt confound comparisons between light directions; these numbers do not isolate a causal lighting effect. Neither model was fitted or tuned on this benchmark.

## ASLLVD: paired camera views

The [publisher's recording documentation](https://www.bu.edu/asllrp/av/dai-asllvd.html) identifies frontal and signer-right profile cameras. Processed videos stack those synchronized views. The same five source-labeled J/Z performances used in the earlier research check are extracted once per camera panel, following the original annotation boundaries and correcting the playback clock to capture time.

There are **five performances, three source signer IDs, ten camera views**. The extra view does not create another person or another recording. Both panels remain grouped by the original performance ID and clip hash. Physical hand stays unknown per token even though the recording protocol describes right-hand-dominant consultants. Brightness normalization is recorded as processing, not physical lighting. All ten views pass the same tracking/movement checks.

| Fixed motion model | Frontal correct after rejection | Profile correct after rejection |
| --- | ---: | ---: |
| Preserved original model | 0/5 | 0/5 |
| Installed expanded model | 3/5 | 1/5 |

The small profile result exposes a remaining limitation. This is a paired diagnostic using an already-reviewed source, not a new untouched test corpus. No model fitting, gate tuning or promotion follows these results. Source terms restrict redistribution/commercial use; videos, source annotation subset, signer mapping and coordinates remain in ignored local storage. Only aggregate results are bundled.

## Reviewed alternatives

`datasets/condition-source-review.json` records thirteen source decisions with primary links, available evidence, missing fields and actions. It includes:

- SignAlphaSet: reported both-hand percentages do not identify individual files. Detector output and static augmentation cannot fill that annotation gap.
- ASL-HG: signer/letter IDs are available; described environmental variety is not a per-image condition table.
- NVIDIA ASL 1000: the AWS registry explicitly identifies controlled access. No access controls were bypassed or uninspected metadata claimed.
- Google fingerspelling: participants may switch hands; MediaPipe left/right blocks and participant dominance are not verified actual-hand labels. Phrase text is not a character-boundary annotation.
- ChicagoFSWild: human transcriptions and signing/non-signing hand boxes do not supply the requested joint capture annotations.
- Pugeault/Bowden: the harder lighting set is depth-only, so it cannot establish RGB webcam lighting coverage.
- OpenSign and ASLA: full data/access agreements require author contact; ASLA is an EMG corpus. No messages or forms were submitted.
- LSM/Sign4all: documented metadata belongs to different sign languages; Latin letter names do not establish ASL equivalence.

This review is not a claim that no qualifying dataset can ever exist. Completing the target requires a public release or annotation sidecar that identifies actual hand and capture conditions for each original ASL sample. A future source should include original sample/subject IDs, image/video hashes, actual hand used, illumination description, camera view/angle definition, transformation status and usable licensing terms.

## Reproduce and inspect

The source documents can be verified without downloading more images:

```powershell
node scripts/download-condition-docs.mjs
```

On an existing checkout with the earlier Massey review extraction, refresh metadata without changing coordinates:

```powershell
.\.venv\Scripts\python.exe -m ml.refresh_public_metadata massey --input artifacts/public-review/massey-landmarks.json --output datasets/massey-landmarks.json
```

On a fresh checkout, the bundled coordinate file is ready for audit. To repeat extraction from the pinned images:

```powershell
node scripts/download-public-source.mjs massey
.\.venv\Scripts\python.exe -m ml.import_public massey
```

The separate research camera-view benchmark requires the local ASLLVD source preparation described in MOTION_DATA.md and compliance with its terms:

```powershell
.\.venv\Scripts\python.exe -m ml.condition_benchmark extract-views
.\.venv\Scripts\python.exe -m ml.condition_benchmark report
.\.venv\Scripts\python.exe -m ml.report_public_coverage
```

`datasets/condition-benchmark-report.json` contains measured condition scores, sample/view counts, source limits and unresolved gaps. `signspeak/static/data/condition-coverage.json` supplies the partial-coverage notice at `/capture`. `ml.condition_evidence` counts only matching, source-linked annotations from recognized publisher evidence, rejects detector/augmentation claims, and keeps camera setup, signer dominance and postprocessing separate from capture truth. Tests enforce these distinctions and prevent benchmark data entering training.
