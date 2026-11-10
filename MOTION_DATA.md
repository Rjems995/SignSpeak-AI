# J/Z movement expansion

The live motion model now uses **188 real J/Z movements**, up from 46, plus **10 real non-J/Z controls**. The static classifier is unchanged. New recordings sample more frequently, accept complete movements as short as 250 ms when at least eight valid frames exist, and allow up to 4.5 seconds for slower signing. Select **Finish movement** to finish early. Camera stop or tab hiding cancels the recording without saving a partial sample.

## Original recordings and provenance

| Source | Accepted J | Accepted Z | Accepted non-J/Z controls |
| --- | ---: | ---: | ---: |
| [SigNN J/Z video dataset V1](https://www.kaggle.com/datasets/signnteam/asl-sign-language-alphabet-videos-j-z), CC0 | 89 | 80 | 0 |
| [SignAlphaSet dynamic V3](https://data.mendeley.com/datasets/8fmvr9m98w/3), CC BY 4.0 | 10 | 9 | 10 |
| Total training/evaluation pool | 99 | 89 | 10 |

A deterministic 72-per-letter SigNN selection is combined with the original 48 files, yielding 178 original videos. The verified SignAlphaSet archive contributes 20 J/Z AVIs and 24 candidate controls from A/D/I/L/R/S. Still images extracted from those AVIs are never counted as separate movements. Of 222 selected videos, 198 pass tracking and movement validation; 24 rejection records are retained. No accepted video hashes or canonical trajectory fingerprints are repeated. Source labels, clip hashes, licenses and source URLs survive preparation.

The original 46 accepted clips retain their frozen coordinate extraction and train/validation/test assignment. New videos are sampled at approximately 30 fps, improving temporal detail over the previous approximately 12 fps extraction. Training augmentation happens only after selecting original clips; augmented vectors never increase real-clip counts or diversity coverage.

SignAlphaSet's paper says its three static-dataset participants also recorded dynamic signs. Neither source provides **per-clip signer IDs**, speed, motion-size, actual-hand or condition annotations. Participant IDs stay null; filenames are never used as invented people. The expanded split is **clip-disjoint, not signer-independent**. Adding an independent source gives additional recording variation without establishing a verified training-signer count.

The static SignAlphaSet import remains review-only. This motion import specifically uses the source's original AVIs, with hashes and movement checks; it does not use the ambiguously augmented static images. The new controls retain their actual source labels and are trained as `_`, meaning non-J/Z. The browser never emits `_` as a letter.

## Observed speed and motion size

These are measurements from detected landmarks, not source capture labels. Active-interval detection and MediaPipe errors affect them. Duration is not interchangeable with travel speed; trajectory extent is expressed in median palm lengths.

| Positive recordings | Active-duration range | Trajectory-extent range |
| --- | ---: | ---: |
| J | 450–3,666 ms | 1.23–4.91 palm lengths |
| Z | 750–3,967 ms | 0.80–6.97 palm lengths |

`datasets/motion-expansion-audit.json` also records travel-speed proxies and medians. The comparison divides held-out positives into lower/middle/upper thirds using **training-only** quantile boundaries for duration, extent and travel speed. Correct gated recognition improves in all nine reported groups. The largest-extent group remains weak: 66.7% on 15 test clips, versus 53.3% for the old model. The longest-duration group reaches only 62.5% on eight clips. These small bins do not establish universal speed or motion-size robustness. No mirrored, resized or retimed copy is counted as a new recording or signer.

## Evaluation and installed model

The new model uses **119 training clips, 39 validation clips and 40 test clips**. Six training clips, two validation clips and two test clips are non-J/Z controls. Every original baseline clip remains in its original partition. Exact video hashes and normalized trajectory fingerprints cannot cross partitions. Gates use validation data; support points and animated references use training data. No test-based parameter or threshold tuning was performed.

| Same frozen evaluation | Old model | Installed expanded model |
| --- | ---: | ---: |
| Raw J/Z accuracy, 38 positive test clips | 100% | 97.37% |
| Correct J/Z recognition after rejection, same 38 clips | 71.05% | 84.21% |
| False acceptance, two held-out non-J/Z controls | 0/2 | 0/2 |
| Correct after rejection, original ten test clips | 7/10 | 9/10 |

The expanded model has one additional raw Z miss, classified as unsupported. It accepts more correct movements after rejection rather than weakening rejection based on test results. Including the two controls, raw accuracy is 97.5% and gated accuracy is 85% on all 40 clips. The model metadata shown in the app reports the **positive J/Z** figure (37/38), so controls do not inflate that displayed score.

The architecture remains a 160-feature sequence input, scaler, 48-unit ReLU layer and 24-unit ReLU layer, now with three softmax outputs: J, Z, and non-J/Z. The feature math is unchanged; only the minimum accepted recording duration changes from 450 to 250 ms. Shorter, stationary, non-finite or severely interrupted sequences still fail validation. Live recording targets approximately 30 fps during motion, while static processing keeps its previous cadence. Actual camera throughput depends on the device. Early finish clears the automatic timer, preventing duplicate results; cancellation clears the buffered sequence.

## Independent annotated research benchmark

A separate [ASLLVD](https://www.bu.edu/asllrp/av/dai-asllvd.html) benchmark contains five annotated J/Z performances from **three source-identified native ASL signers**, plus eleven D/I/L controls. It is excluded from training and validation. The source terms allow research/education and restrict redistribution/commercial use; its videos, annotation subset, signer mapping and extracted coordinates remain in ignored local caches. Only aggregate evaluation results are bundled.

The source's processed MOVs stack synchronized frontal/profile views and play captured 60-fps frames at 15 fps. Extraction uses only the frontal panel, corrects timestamps to capture time, selects the raised signing hand, and follows the published sign start/end frames for positives. The profile view is never counted as another performance. Actual context frames are retained for controls. In particular, a fluent J performance lasting about 333 ms exposed the previous overly restrictive minimum-duration check.

| Independent research test | Old model | Installed expanded model |
| --- | ---: | ---: |
| Correct J/Z recognition after rejection | 0/5 | 3/5 |
| False acceptance of D/I/L controls | 0/11 | 0/11 |

Both models use the same updated preprocessing for this comparison. Five positive clips and eleven controls are a **tiny benchmark**, not a reliable estimate of accuracy on all new signers. Raw native-signer J/Z classification drops from 5/5 to 4/5, while accepted correct predictions improve. Native J remains 2/3 accepted; native Z is only 1/2 accepted. No real live-webcam benchmark was collected.

## Files and reproduction

- `datasets/motion-expanded.json`: 198 original coordinate sequences with provenance, original source labels and measured observations.
- `datasets/motion-expanded-provenance.json`: the 178 pinned SigNN video names, sizes and SHA-256 hashes.
- `datasets/motion-sources.json`: SignAlphaSet archive and ASLLVD annotation hashes, attributions, metadata limits and terms.
- `datasets/motion-expansion-audit.json`: counts, all rejections and measured variation.
- `datasets/motion-candidate-evaluation.json`: frozen baseline comparison, speed/size groups and aggregate research benchmark.
- `datasets/baselines/motion-v1/`: all four original motion artifacts for reproducibility and rollback.
- `signspeak/static/models/motion-*.json`: installed expanded model, metadata, training references and held-out report.

Install `requirements-motion.txt` and obtain `datasets/videos/hand_landmarker.task` as described in README.md. Raw downloads are roughly 1.04 GB for selected SigNN files plus a 1.25 GB SignAlphaSet archive. Existing baseline files are reused after hash verification. All raw videos and review data stay ignored.

```powershell
.\.venv\Scripts\python.exe -m ml.download_motion --per-letter 72 --include-baseline --output-dir datasets/videos/expanded/signn --provenance datasets/motion-expanded-provenance.json
node scripts/download-public-source.mjs signalpha_motion --motion
.\.venv\Scripts\python.exe -m ml.import_motion_expansion
.\.venv\Scripts\python.exe -m ml.train_motion --dataset datasets/motion-expanded.json --freeze-baseline-splits datasets/baselines/motion-v1/motion-evaluation.json --output artifacts/motion-candidate
.\.venv\Scripts\python.exe -m ml.evaluate --model-directory artifacts/motion-candidate
node scripts/verify-motion-candidate.mjs artifacts/motion-candidate
```

The separate research benchmark can be reproduced locally, within the source's terms:

```powershell
node scripts/download-public-source.mjs asllvd --motion
.\.venv\Scripts\python.exe -m ml.asllvd_review prepare
node scripts/download-asllvd-review.mjs
.\.venv\Scripts\python.exe -m ml.asllvd_review extract
.\.venv\Scripts\python.exe -m ml.compare_motion_models
```

`ml.compare_motion_models` compares the preserved baseline against `artifacts/motion-candidate/`; it never silently compares an installed model against itself. Python and TensorFlow.js agree on all 40 held-out clips, including feature extraction, probabilities and rejection decisions. Separate browser checks cover faster sampling, early completion, the automatic limit and cancellation without partial samples.
