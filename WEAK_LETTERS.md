# E, M, R and T: measured progress in an unreleased candidate

**The installed static-v2 model and J/Z model remain unchanged.** The latest
candidate improves E/M/R/T against the earlier public candidate on a reused
development benchmark, but its **78.21% overall correct acceptance fails the
unchanged 80% release minimum**. It is retained for review at
`datasets/candidates/focus-topology-v5/` and is **not deployed**. R still produces
wrong U labels, and these results do not establish reliable live recognition.

## Same-image comparison

The earlier public-data candidate was also never deployed. On the **same 840
successfully tracked images**, correct accepted letters are:

| Letter | Earlier public candidate | Latest topology candidate |
| ------ | -----------------------: | ------------------------: |
| E      |            40/60 (66.7%) |              60/60 (100%) |
| M      |            19/60 (31.7%) |               39/60 (65%) |
| R      |            23/60 (38.3%) |               39/60 (65%) |
| T      |            20/60 (33.3%) |               30/60 (50%) |

All-letter correct acceptance changes **632/840 (75.24%) → 657/840 (78.21%)**.
The latest candidate meets the declared E/M/R/T count minima, accepted-R precision
minimum and unknown false-acceptance maximum. It fails the overall 80% minimum;
the criterion was not relaxed. Its accepted R precision is 100%, but the 60 valid
R images yield **39 correct R labels, 19 wrong U labels and 2 rejections**.
Precision among accepted R labels does not describe mistakes on true R attempts.

These images were initially reserved before evaluating the preceding kernel
candidate, then **reused for this development regression**. Topology coefficients
and its threshold use training/validation only; the final score is still not an
independent holdout result, a new-signer estimate or a live-webcam benchmark.

For context, the installed v2 model scores 384/840 (45.71%) on these images;
its E/M/R/T correct counts are 59/27/10/8. Compared with **that installed model**,
the candidate reduces false letters on the historical unknown benchmark from
**2/48 to 0/48** tracked poses. Three additional unknown images fail tracking and
remain in the selected denominator of 51. The earlier public candidate falsely
accepts 11/48 on this unknown set. These small reused controls are not a new
independent unknown-handshape benchmark.

`datasets/weak-letter-topology-evaluation.json` preserves the latest comparison,
failure and denominators. The preceding kernel candidate is retained at
`datasets/candidates/focus-kernel-v4/` with `datasets/weak-letter-evaluation.json`:
it scored **630/840 (75%)**, including R **12/60**, 27 wrong U labels and 21
rejections. It failed both R and overall criteria. The initial 273-feature MLP
also failed its original test release; its decision, policy and results remain
in `datasets/weak-letter-first-*.json`. That original test had already been
inspected before the kernel experiment.

## What was added

The final pool has **8,980 real training images and 1,224 frozen validation images**.
Only the eight existing training participant IDs contribute new training data.
Selection adds 3,202 original ASL-HG training images for E/M/R/T and their S/N/U/V
confusions; 14 fail extraction and 36 formerly selected training rows are removed:
**5,828 + 3,202 − 14 − 36 = 8,980**. Source-byte checks quarantine 100 cross-person
V-image duplicate groups. Publisher participant IDs do not independently verify
identity, and copying an image under another ID cannot add signer diversity.

The initially reserved check has 20 images per target letter/person and 10 per other static
letter/person: **840 tracked images from `asl_hg_p8`, `asl_hg_p10`, and
`aslyset_user2`**. Original balanced-test files/hashes and duplicates are excluded.
These are the **same previously assessed people**, not new unseen signers or
independent recording sessions. Six failed fresh reserve extractions are audited
separately and are not part of 840; the accuracy above is conditional on tracking.
Source order is deterministic and never uses classifier confidence.

## Candidate and validation

The 315 features retain 42 oriented image-XY values, add 21 relative image-Z values,
210 normalized world-landmark pair distances and 42 palm-aligned XY values.
Original XY orientation remains available for G/Q, H/U and K/P distinctions.
MediaPipe XYZ/world values are **estimated geometry, not verified measured depth**.
Python IMAGE extraction uses 0.10.35; the browser VIDEO tracker uses 0.10.21.

Training fits StandardScaler on originals, then a 640-prototype Nyström RBF basis
(`gamma=1/315`, seed 17) and multinomial logistic regression (`C=1000`). No synthetic
training images are added. Nyström normalization is folded into the output weights.
Up to eight real representatives per signer/letter yield **1,520 support vectors**
after duplicate quarantine; the vectors are rounded to six decimals. Positive gates
use validation-only 95th-percentile distances with a .12 minimum. The 76 reviewed
negative training poses retain a distance-ratio veto of 1; confidence remains .85.
E/R agreement uses the fully gated preserved v2 model plus primary probability
at least .01. The older expert has unidentified training signers.

Before opening the fresh holdout, validation scores **1,127/1,224 (92.08%)** correct
acceptances: E **51/51**, M **47/51**, R **38/51**, T **47/51**, unknown **0/27**.
The per-person view already exposes R's weakness: only **4/17** for ASLYset user3.
Its first fresh-image test exposed a larger R regression.

The latest candidate retains those primary weights and adds a **trained binary R
classifier** over 12 descriptors of the first 42 canonical XY values. These measure
finger crossing/order, fingertip/PIP/MCP lateral gaps, four finger extensions, two
thumb-to-MCP distances, fingertip distance and fingertip vertical gap. The feature
math does not directly assign a letter. StandardScaler and balanced logistic
regression fit 8,980 known training hands plus 76 unknown hands, including 675 R
examples. Feature design used validation images; no extra source images enter this
step, and no still-image pixels are read during recognition.

Three fits (`C=.1, 1, 10`) at four thresholds (`.85, .9, .95, .99`) produce 12
validation trials. The selected **C=1, threshold=.99** candidate raises validation
R from **38/51 to 45/51**, including **4/17 to 11/17** for ASLYset user3, with no
additional wrong R labels, unchanged E/M/T counts and **0/27** unknown acceptances.
Overall validation correct acceptance is **1,134/1,224 (92.65%)**. The added
classifier can supply R only when the primary raw label is R/U/V/D and the gated
result is one of those letters or rejected. It does not veto existing R labels;
the effective threshold also honors a higher user confidence setting. The frozen
validation audit is `datasets/weak-letter-topology-validation.json`.

Four additional binary-R image-texture trials use 64px grayscale HOG: three linear
models and one 256-prototype RBF model, each at four validation thresholds.
**None of 16 trials improves R without additional wrong R acceptances.** The safe
RBF .99 threshold makes no recall gain and does not qualify. No texture expert or
raw photographs are installed/bundled. `ml/texture_features.py` and the guarded
`ml/train_texture_expert.py` retain reproducible research math and failed-trial audit.

## Reproduce

Obtain the hash-verified source archives and Google hand-landmarker task described
in README.md; install `requirements-motion.txt`. Existing bundled coordinates can
be used directly for training. Reconstruction and review commands are:

```powershell
.venv\Scripts\python.exe -m ml.import_static_depth --split train validation test --output datasets/static-depth.json
.venv\Scripts\python.exe -m ml.import_static_depth --unknown --split train validation test --output datasets/unknown-depth.json
.venv\Scripts\python.exe -m ml.expand_focus_depth --phase prepare
.venv\Scripts\python.exe -m ml.expand_focus_depth --phase extract
.venv\Scripts\python.exe -m ml.expand_focus_depth --phase finalize
.venv\Scripts\python.exe -m ml.train_focus_kernel
.venv\Scripts\python.exe -m ml.train_topology_expert
.venv\Scripts\python.exe -m ml.review_weak_letters --images
.venv\Scripts\python.exe -m ml.evaluate_static_depth --candidate datasets/candidates/focus-kernel-v4 --dataset datasets/focus-depth.json --test-split fresh_test
.venv\Scripts\python.exe -m ml.evaluate_static_depth --candidate datasets/candidates/focus-topology-v5 --dataset datasets/focus-depth.json --test-split fresh_test --policy datasets/weak-letter-topology-release-policy.json --validation datasets/weak-letter-topology-validation.json --report datasets/weak-letter-topology-evaluation.json
```

Training writes `artifacts/focus-kernel-reproduced/` and
`artifacts/topology-expert-reproduced/`, checks selection/source hashes and split
leakage, and never performs holdout inference or installation. The topology trainer
requires a fresh output directory; use `--output` to choose another for reruns.
Evaluation checks sealed model/validation hashes, both dataset hashes, test split,
complete cohort identities/counts and unknown tracking-failure denominators before
inference. Adding `--install` **refuses both failed candidates**. The 21-example
image review queue retains publisher labels pending ASL expert review; its contact
sheet stays in ignored local artifacts. Further release evidence needs another
clearly identified independent cohort and expert-reviewed R examples across signers.

## Sources and limits

[ASLYset V1, Miguel Rivera (2019)](https://data.mendeley.com/datasets/xs6mvhx6rh/1)
and [ASL-HG V1, Pranto et al. (2025)](https://data.mendeley.com/datasets/j4y5w2c8w9/1)
are CC BY 4.0. Changes include original-image selection, cropping, estimated XYZ/
world extraction, deduplication, geometry/topology/HOG derivation and candidate training.
Unknown controls also use [HaGRID](https://github.com/hukenovs/hagrid), under its
**custom attribution/share-alike public license**, not an unmodified Creative
Commons license; its full text is in `datasets/HAGRID-LICENSE.pdf`. Full credits
and component terms remain in NOTICE.md and UNKNOWN_HANDS.md. No endorsement is
implied. Physical hand, lighting, camera-angle and session gaps remain; detector
depth/handedness estimates do not supply those missing annotations. No live webcam
or broad fairness/robustness claim follows from these source-image experiments.
