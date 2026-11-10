# Unknown alphabet handshapes

The installed static classifier now vetoes a letter when its normalized landmarks
are at least as close to a reviewed non-letter training example as to that
letter's training support. The 24-letter network, its probabilities, positive
support and distance thresholds are unchanged. The extra support contains **76
real training images**, with a validation-selected distance ratio of **1.0**.
The veto remains active when the user changes the confidence threshold. The
published benchmark uses the default confidence threshold of **0.85**.

Tracked rejected hands display **“Sign not recognized.”**, an empty letter and no
confidence percentage. They reset the hold progress and cannot commit a letter.
No detected hand has a separate prompt. Degenerate coordinates reject without
stopping the camera. Rejected or invalid J/Z recordings clear stale accepted
results; the J/Z classifier itself is unchanged.

## Frozen test results

| Test subset | Tracked samples | Previous false letters | Installed false letters |
|---|---:|---:|---:|
| Relaxed hands | 11 | 5 | 0 |
| Partial finger flexion | 2 | 1 | 1 |
| Unrelated gestures | 35 | 10 | 1 |
| Total | 48 | 16 (33.33%) | 2 (4.17%) |

Three more selected test images failed one-hand tracking, so end-to-end false
acceptance is 16/51 versus 2/51. Keeping those failures in the denominator does
not imply their handshape was correctly classified. The tracked denominator is
the primary rejection measure. Resting hands are source-bounding-box crops from
images containing another gesture hand; they do not demonstrate rejection of an
uncropped two-hand scene.

Correct accepted letters on the older SigNN test fall from **86/125 (68.8%) to
85/125 (68.0%)**: one additional false rejection. Raw classification stays 74.4%.
The separate public known-letter test remains **557/1,224 (45.51%)** correctly
accepted; seven incorrect acceptances are additionally rejected. These results
do not fix the letter classifier's existing accuracy limitations.

`datasets/unknown-evaluation.json` retains both complete comparisons, per-source
and per-category counts, every validation trial, split fingerprints, model/data
hashes and all 48 held-out feature vectors. The current static evaluation report
also reflects the new 68.0% gated result. Original classifier/evaluation/metadata
artifacts are preserved under `datasets/baselines/static-v1/`.

## Data and review

Reviewed **275 deterministically selected official HaGRID images**, one source
participant per selection cohort, plus **20 ASLYset FN controls**. Selected
indices and decisions are in `datasets/unknown-review.json`; image and annotation
hashes are pinned in `datasets/unknown-source-lock.json`. Images are not bundled.

* HaGRID's dedicated `no_gesture` subset provides face-touching gestures. Its
  source label does **not** establish that a handshape is outside ASL. Object
  grips, hidden thumbs, closed parallel fingers and letter-compatible or
  ambiguous shapes were excluded conservatively.
* The `four` annotation files identify a secondary `no_gesture` hand. Only the
  reviewed resting-hand crop is extracted, with 25% bounding-box padding.
* ASLYset FN controls show a five-finger spread palm, outside this application's
  alphabet vocabulary. Source control label FN remains recorded. This shape is
  not an assertion of invalid ASL: numbers and other signs are outside scope.
  SP controls were excluded because their shapes can overlap alphabet letters.
* Another 48 frames from SignAlphaSet movement clips were reviewed but excluded:
  apparent incomplete signs often already matched another valid letter, or were
  blurred/occluded. No video transition was assigned a fabricated negative label.

Review was **Codex visual inspection, not ASL expert certification**. It selected
162 real images before model inference: 151 unique one-hand extractions and 11
tracking failures. Tracked train/validation/test counts are **76/27/48**. Cropping
does not create independent performances. All imported fingerprints, original
image hashes and identified source participants stay within one partition;
ASLYset partitions also match the existing public known-letter split. IDs are
publisher identities, not independently reconciled identities across datasets.

Coordinates were re-extracted with Python MediaPipe **0.10.35**, IMAGE mode,
up to two hands, detection/presence confidence 0.6, retaining exactly one hand.
The browser uses **0.10.21**, VIDEO mode. This version/mode difference and still
images limit conclusions about live tracking. Physical hand, lighting and camera
angle metadata remain unknown in this negative subset.

Only **three** partial-flexion images tracked across all splits (one training,
two test; validation example failed tracking). These are visible bent/spread
finger poses, a **proxy**, without annotated intended letter or incomplete-sign
ground truth. Half of their tracked test examples still produce a false letter.
An incomplete intended sign that exactly matches a valid letter cannot be
rejected from this handshape representation alone. More expert-reviewed real
transitions and live clips are still needed; do not claim general reliability.

## Validation and installation

Threshold candidates were fixed before inspecting test predictions:
0.25, 0.5, 0.75, 1, 1.25, 1.5, 2, 3, 4. Minimize unknown validation acceptances;
break ties toward the smallest ratio. A candidate must preserve all previous
correct legacy validation acceptances, at least 99% of public correct validation
acceptances overall and 95% per letter, and reduce unknown validation acceptances.
Ratio 1 meets those criteria: unknown acceptances fall **8/27 to 0/27**, with
all **79 legacy and 695 public** correct validation acceptances retained. Larger
ratios fail the retention policy. Test results do not choose the ratio or support.

Re-evaluate or install from bundled coordinates (no image downloads required):

```powershell
.venv\Scripts\python.exe -m ml.unknown_rejection
.venv\Scripts\python.exe -m ml.unknown_rejection --install
.venv\Scripts\python.exe -m pytest -q
npm.cmd test
# Against a running isolated local QA server:
$env:TEST_BASE_URL='http://127.0.0.1:5015'
npm.cmd run test:rejection-browser
```

To reconstruct the coordinates, first obtain the hash-verified ASLYset archive
and Google's hand-landmarker task as described in README.md, then:

```powershell
.venv\Scripts\python.exe scripts/fetch-hagrid-review.py
.venv\Scripts\python.exe scripts/fetch-hagrid-review.py --resting
.venv\Scripts\python.exe scripts/review-unknown-images.py
.venv\Scripts\python.exe scripts/review-unknown-images.py --resting
.venv\Scripts\python.exe -m ml.import_unknown
```

The downloader uses bounded HTTP ZIP member ranges, validates CRC/size and pinned
SHA-256 hashes, and stores originals under ignored `datasets/public/`. It fetches
only the selected images and six annotation members, not the large full image
archives. MediaPipe extraction may vary across versions; retain the committed
coordinates and hashes for the frozen benchmark.

## Attribution and terms

**HaGRID / HaGRIDv2:** Alexander Kapitanov, Andrey Makhlyarchuk, Karina Kvanchiani,
Aleksandr Nagaev, Roman Kraynov and Anton Nuzhdin; see the
[official project, credits and source annotations](https://github.com/hukenovs/hagrid).
Its README describes a variant of attribution/share-alike terms. The actual
[English license](https://github.com/hukenovs/hagrid/blob/master/license/en_us.pdf)
explicitly says it is **not a Creative Commons license**. Its complete supplied
text is retained in `datasets/HAGRID-LICENSE.pdf`, with its hash in the source lock.
HaGRID-derived coordinates and their rejection-support component are provided
under that same custom public license, including its attribution, share-alike,
no-additional-restrictions and warranty-disclaimer provisions. Do not relabel
them MIT, Apache-2.0 or unmodified CC BY-SA. Changes: deterministic selection,
visual review, resting-hand cropping, MediaPipe extraction, normalization,
deduplication and grouping selected non-alphabet poses as unsupported.

**Miguel Rivera (2019), ASLYset, Mendeley Data V1, doi:10.17632/xs6mvhx6rh.1**,
[original dataset](https://data.mendeley.com/datasets/xs6mvhx6rh/1),
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Changes: selection of
20 original FN images, coordinate extraction, normalization and negative-support
export. Their original source labels and image hashes remain in the import audit.
Original SigNN classifier components retain their existing Apache-2.0 attribution.
No source creator endorsement is implied.
