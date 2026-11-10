# SignSpeak AI

A Flask application for learning and recognizing the **full American Sign Language alphabet, A-Z**. MediaPipe tracks one hand. A trained TensorFlow.js network recognizes 24 static handshapes; a separate trained temporal network recognizes J/Z movements from a short recording. Recognition builds editable text, and browser speech synthesis reads it aloud.

This is an experimental alphabet MVP, not full ASL translation. It does not interpret facial expressions, grammar, or context. ASL and FSL are separate languages; this application supports ASL only.

## What works

- Responsive landing page, dashboard, light/dark themes, and mobile navigation.
- Permission-based webcam access; real 21-point MediaPipe tracking; optional landmark overlay.
- Static classifier: 42 -> 96 -> 64 -> 24. Movement classifier: 160 temporal features -> 48 -> 24 -> 3 (J, Z, unsupported). Both use learned weights, not hardcoded letter predictions.
- Wrist/scale/mirror normalization that preserves meaningful hand direction, confidence/distance gates, reviewed negative-handshape support, temporal smoothing, and release-before-repeat for static letters. Rejected hands display “Sign not recognized.”
- Editable text, space/delete/clear/copy, and explicit saving to private history.
- English text-to-speech, available voice selection, speed, pause/resume, and stop.
- All 26 ASL lesson cards, alphabetical navigation, recorded static references, animated J/Z movement references, and links to ASL University visual demonstrations.
- Five-letter camera practice sessions, explicit model feedback, attempt counts, session summaries, and saved results.
- Registration, login/logout, scrypt password hashes, CSRF protection, authentication rate limits, account deletion, and user-scoped records.
- SQLAlchemy tables for users, profiles, settings, translations, practice sessions/results, and learning progress. SQLite for easy local setup; PostgreSQL on Render.
- Local, consent-based landmark sample collection at `/capture`; data preparation, training, image extraction, and evaluation scripts.

There are no simulated recognition results, seeded history records, or invented progress metrics. Hand illustrations and lesson reference drawings are labeled reference material. Dashboard accuracy comes from submitted practice attempts, not a claim about model performance. Lesson “explored” status records reading a lesson, not proficiency. Practice records are client-reported learning aids, not verified assessments.

## Windows + VS Code setup

Install Python **3.12** and open this folder in VS Code. Select **Python: Select Interpreter** and choose the project's `.venv` interpreter after creating it.

From PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python.exe -m flask --app run init-db
.\.venv\Scripts\python.exe run.py
```

Open **http://127.0.0.1:5000**. Activation is optional; using the interpreter directly avoids PowerShell execution-policy changes. If `py` is unavailable, use the installed Python 3.12 executable to create the environment. If using the Python Install Manager, install 3.12 first with `py install 3.12`.

Replace `SECRET_KEY` in `.env` with a random value. Generate one with:

```powershell
.\.venv\Scripts\python.exe -c "import secrets; print(secrets.token_hex(32))"
```

For a local production-style server on Windows:

```powershell
.\.venv\Scripts\python.exe -m waitress --listen=127.0.0.1:5000 run:app
```

Camera access requires localhost or HTTPS. Desktop Chrome/Edge are the main validation targets. Other modern browsers may differ in camera, WebAssembly, clipboard, and speech support. First camera activation downloads MediaPipe/TensorFlow.js and landmark assets; an internet connection is required. No paid AI APIs are used.

The checked-in classifier runs without installing machine learning dependencies or training anything.

## Model results and practical limits

The static model uses [SigNN's pinned public coordinate file](https://github.com/AriAlavi/SigNN/blob/40bd651899e09a0099767457194062f77f774156/scripts/mediapipe_data_creation/training_data.json), distributed under the upstream Apache 2.0 license. Its 622 unique canonical samples are split into 373 training, 124 validation, and 125 test samples. Direction is preserved so orientation-dependent letters remain distinguishable.

The installed J/Z model uses **188 real movements and 10 non-J/Z controls** from SigNN's CC0 videos and SignAlphaSet's CC BY 4.0 original AVIs. It splits 198 clips into 119 training, 39 validation and 40 test clips, preserving every original baseline split before augmentation. Per-clip signer IDs are unavailable. Full provenance, speed/size measurements, limitations and reproduction commands are in [MOTION_DATA.md](MOTION_DATA.md).

| Model | Raw held-out accuracy | Accuracy after the default rejection gate |
| --- | --- | --- |
| 24 static letters | 74.4% on 125 samples | 68.0% on 125 samples (previously 68.8%) |
| J/Z movements | 97.37% on 38 positive test clips | 84.21% on those 38 clips |

Gated accuracy counts rejected valid examples as errors; it is not precision among accepted outputs. The motion model rejects 2/2 held-out non-J/Z controls. An independent, research-only ASLLVD benchmark yields 3/5 accepted correct J/Z predictions and 0/11 falsely accepted controls, versus 0/5 and 0/11 for the old model. These tiny control sets do not establish reliable unknown-gesture rejection. The public motion split has no per-clip signer IDs and does not estimate unseen-signer accuracy. The static rejection update reduces false letters from 16/48 to 2/48 reviewed held-out unknown poses; see UNKNOWN_HANDS.md for the still-image limits and one additional known-letter false rejection. In the static class network, Q still has only two unique samples: one training, one test, no validation. Its gate uses a fixed conservative fallback, and the single Q test sample was not correctly classified.

Confidence is an uncalibrated model probability. Unsupported gestures can still receive high scores. Inspect/correct the generated text and collect more diverse, consented, participant-labeled data before relying on recognition. The new metrics replace the previous six-letter model's results.

See `signspeak/static/models/evaluation.json`, `motion-evaluation.json`, their metadata files, and [MODEL_CARD.md](MODEL_CARD.md). Reports contain per-class precision/recall/F1, confusion matrices, split IDs, and held-out probabilities.

## Using every letter

- A-I and K-Y: hold one clear handshape steady until the hold indicator fills.
- J and Z: enable the camera, select **Record J / Z**, trace once, and select **Finish movement**. Recording ends automatically after 4.5 seconds. J uses the little fingertip; Z uses the index fingertip. Coordinates are held temporarily in browser memory; no video is recorded or uploaded.
- In practice, J/Z prompts show a moving reference. Record the movement, then select **Check my sign** within eight seconds of an accepted result.
- Lower/release your hand before repeating a static letter. Use a new recording for another J/Z. Camera stop, tab hiding, and navigation cancel an unfinished recording.

## Reproduce training and evaluation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
# Optional: re-download the pinned public source and license.
.\.venv\Scripts\python.exe -m ml.download_dataset
.\.venv\Scripts\python.exe -m ml.prepare datasets/signn.json --drop-conflicts --output datasets/private/clean-signn.json
.\.venv\Scripts\python.exe -m ml.train
.\.venv\Scripts\python.exe -m ml.evaluate
```

The classifiers are trained with scikit-learn MLPs. Their learned kernels, biases, and scalers are exported to JSON and evaluated with TensorFlow.js in the browser. No Python inference or video upload is involved. Tests compare Python and JavaScript feature extraction and held-out probabilities for both models.

`ml.train` writes a new classifier, model metadata, measured test results, reference samples, and confusion-matrix image. The legacy source audit now finds seven conflicting-coordinate label groups; preparation refuses to choose a label arbitrarily. Use `--drop-conflicts` to quarantine those groups. The retained `datasets/prepared.json` and `datasets/baselines/static-v1/` are the frozen historical baseline; generating cleaned data does not reproduce its original training set. The installed model adds the separately evaluated rejection support described in UNKNOWN_HANDS.md; retraining a classifier replaces that export, so evaluate rejection again before installing new weights. Restart or refresh the app afterward. Keep the scripts' random seeds and use the locked development versions in `requirements-lock.txt` to reproduce the checked-in artifacts. Retraining with future library versions can change the results.

If Matplotlib cannot write its user cache, set a workspace-local directory first:

```powershell
$env:MPLCONFIGDIR = "$PWD\artifacts\matplotlib"
```

### Collect a better dataset

The workshop and public data are documented in [DATA_COLLECTION.md](DATA_COLLECTION.md). The eligible ASLYset + ASL-HG pool contains **10,516 usable unique samples**, including **499 Q samples**, across **14 source signer IDs**. `datasets/public-balanced.json` selects 17 originals per signer/letter: **5,712 samples, 238 per static letter**. The workshop tracks signer/session, actual hand, lighting, angle and duplicates. Unknown public metadata stays unknown. SignAlphaSet static images remain review-only. The Massey source is now a licensed, bundled lighting benchmark reserved outside training.

[CONDITION_DATA.md](CONDITION_DATA.md) documents **895 source-labeled lighting samples** and **10 camera views of five existing ASL performances**. The audit keeps light direction, camera setup, actual camera view and signer dominance separate. Verified physical left-hand samples and jointly annotated hand/light/view examples are still missing. `/capture` displays this partial-coverage status; no model weights were changed by the condition audit.

A separate static candidate scores **89.13% raw / 80.23% with rejection** on 1,224 images from three held-out source signers; the live static model scores **57.60% / 45.51% on that same test**. E and gated M performance regress, so that class network was not promoted. The historical comparison is `datasets/public-candidate-evaluation.json`; reproduction commands and per-letter limits are in DATA_COLLECTION.md. The separate J/Z model has been updated as described in MOTION_DATA.md.

[UNKNOWN_HANDS.md](UNKNOWN_HANDS.md) documents the installed static rejection update: **16/48 → 2/48** false letters on reviewed held-out unknown hands, with real relaxed hands, partial-flexion proxies and unrelated gestures. Correct accepted letters change **86/125 → 85/125** on the legacy test and remain **557/1,224** on the public test. Partial/incomplete shapes remain weak; this small still-image benchmark does not establish live-webcam reliability. Coordinates, split isolation, source terms, validation policy and remaining failures are retained for review.

1. Visit **http://127.0.0.1:5000/capture**. Give each consenting signer a pseudonymous participant ID. The form's consent checkbox specifically covers collecting hand coordinates. Use the same signer ID across sessions and label the actual hand, lighting and viewing angle.
2. Enable the camera. Choose a letter, hold the correct shape, and capture individual samples while varying distance, hand position, lighting, and sessions. Check the labels with a knowledgeable ASL signer.
3. Collect unsupported shapes with the **Other / unsupported handshape** option. Include near-confusable signs, relaxed hands, and partially formed letters. Aim for balanced supported classes and diverse negative examples.
4. Download each JSON file into `datasets/private/`. This folder is ignored by Git. Only coordinates are downloaded; no video is recorded.
5. Collect every class from enough independent participants that training, validation, and test sets each include all classes. Start with at least 10 participants; more diversity is preferable.
6. Train **only on your identified-participant dataset** to get a participant-disjoint evaluation:

```powershell
.\.venv\Scripts\python.exe -m ml.prepare datasets/private/signer01.json datasets/private/signer02.json datasets/private/signer03.json --output datasets/private/prepared.json
.\.venv\Scripts\python.exe -m ml.train --dataset datasets/private/prepared.json
```

Both training scripts use participant-separated groups when all samples have IDs and there are at least three participants. Every training and test class must be represented. The static script allows a class with only two samples to have no validation sample, explicitly reports it, and uses a fixed conservative distance gate. With missing participant IDs, reports state sample-based or source-clip-based splits; mixing unidentified public data with private captures does not produce a signer-independent evaluation.

To extract landmarks from consented still images, organize files as `INPUT/participant/LETTER/*.jpg`. Install the optional extractor:

```powershell
.\.venv\Scripts\python.exe -m pip install mediapipe
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile 'datasets/private/hand_landmarker.task'
.\.venv\Scripts\python.exe -m ml.extract_images datasets/private/photos --hand-model datasets/private/hand_landmarker.task --output datasets/private/images.json
.\.venv\Scripts\python.exe -m ml.prepare datasets/private/images.json --output datasets/private/prepared.json
```

Create `datasets/private/` before downloading if it does not exist. The extractor keeps images with exactly one detected hand. Retain participant grouping and permission records separately; do not commit private images.

### Reproduce J/Z movement training

The trained artifacts are bundled; downloads are needed only to import new recordings. The current reproducible input is `datasets/motion-expanded.json`; raw source videos remain ignored. The original 46-clip input at `datasets/motion.json` and all four original model artifacts under `datasets/baselines/motion-v1/` are retained for comparison and rollback. See [MOTION_DATA.md](MOTION_DATA.md) for the expanded import and separate research benchmark.

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-motion.txt
.\.venv\Scripts\python.exe -m ml.train_motion --dataset datasets/motion-expanded.json --freeze-baseline-splits datasets/baselines/motion-v1/motion-evaluation.json --output artifacts/motion-candidate
.\.venv\Scripts\python.exe -m ml.evaluate --model-directory artifacts/motion-candidate
node scripts/verify-motion-candidate.mjs artifacts/motion-candidate
```

The extractor uses Google's actual hand-landmarker model on real frames and logs rejected clips. Motion features resample tracked wrist/thumb/index/middle/little landmarks to 16 time steps, preserve movement relative to the initial wrist, and reject a stationary hand. Models, scalers, reference sequences, and gates use training/validation clips only.

At `/capture`, select J or Z, enter a participant ID, opt in, and select **Record movement sample**. Use the separate Download static and Download J/Z buttons; each export preserves signer, session, actual-hand and condition metadata. Combine your downloaded J/Z exports and train with participant grouping:

```powershell
.\.venv\Scripts\python.exe -m ml.prepare_motion datasets/private/signer01-motion.json datasets/private/signer02-motion.json --output datasets/private/motion.json
.\.venv\Scripts\python.exe -m ml.train_motion --dataset datasets/private/motion.json
```

Collect many participants, each contributing both letters across varied conditions. Never use source clip IDs as participant IDs. Do not mix static samples into a motion dataset. The motion trainer stops when either class or a required split lacks enough data.

## PostgreSQL and Render

Create an empty PostgreSQL database and configure:

```dotenv
DATABASE_URL=postgresql+psycopg://user:password@localhost:5432/signspeak
```

Then run `python -m flask --app run init-db`. The application also accepts Render's `postgresql://` connection format and selects psycopg 3 automatically. `init-db` creates missing tables; it is not a migration system. Introduce reviewed schema migrations before altering an existing production database. Switching database URLs does not migrate local SQLite records.

The included `render.yaml` defines a Flask web service and PostgreSQL database. Push the project to your Git repository, create a Render Blueprint using that repository, review the current plan availability/cost, and deploy. The application has **not** been deployed by this workspace task.

- Build: `pip install -r requirements.txt`
- Start: `flask --app run init-db && gunicorn run:app --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120`
- Health check: `/healthz`
- Render generates `SECRET_KEY`, injects `DATABASE_URL`, and enables secure cookies.

`TRUST_PROXY=1` trusts one reverse proxy for the client IP and scheme. Use it only behind infrastructure that supplies/replaces these headers. The default limiter storage is process-local; the included deployment uses one worker. For multiple workers/instances use a shared limiter store, e.g. install `redis`, set `RATELIMIT_STORAGE_URI=redis://...`, and configure a managed Redis service. Database setup should become a dedicated release step as the service grows.

See [Render's Flask guide](https://render.com/docs/deploy-flask). Python/Flask and PostgreSQL need a Python-capable host; they cannot run as a static Sites deployment.

## Tests

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-ml.txt
.\.venv\Scripts\python.exe -m pytest -q
npm.cmd ci --no-audit --no-fund
npm.cmd test
```

For headless browser checks, start the local server in another terminal, then:

```powershell
npm.cmd run test:browser
```

Browser checks use the installed Windows Google Chrome at its standard location; set `CHROME_PATH` if yours differs. Set `TEST_BASE_URL` for another local port. They cover desktop/mobile page rendering and overflow, actual CDN model loading and inference with a **fake camera**, a real recorded image from Google's MediaPipe hand sample, text controls, preferences, lessons, and the account/history/deletion lifecycle. The recorded-image check validates detector-to-classifier integration; it is not an ASL accuracy benchmark. Tests create and delete a temporary test account. They do not access your real webcam. The test requires internet access for external models and the reference image. Screenshots go to ignored `artifacts/`.

Automated checks cannot establish live recognition quality or ASL teaching validity. Real signers, camera hardware, browser speech voices, PostgreSQL integration, and a Render deployment require their own acceptance testing.

## Privacy and project layout

Camera activation is explicit. Processing stays in the browser; the server has no video-upload endpoint. Camera tracks stop when the tab hides, you leave the page, or you press Stop. Saved translation text and practice results go to your authenticated account only after their respective actions. Disabling history saving blocks new text saves; existing records remain until deleted. Preferences and lesson review status may be stored in the current browser, separately by account ID. Account deletion clears those preferences in the current browser; storage on other devices can be cleared through those browsers.

MediaPipe/TensorFlow.js assets load from jsDelivr/Google Storage, and fonts load from Google Fonts; these hosts receive ordinary asset requests. No camera images are included. Browser/OS speech synthesis may use an online voice service. Private history API responses disable caching.

```text
signspeak/               Flask factory, database models, routes, ASL lesson content
  templates/            Landing, dashboard, translator, lessons, practice, accounts, settings
  static/css/           Responsive interface and dark theme
  static/js/            Camera, feature normalization, classifier, speech, page interactions
  static/models/        Trained weights, real evaluation report, reference data
ml/                     Download, prepare, train, evaluate, optional image extractor
datasets/               Pinned public samples, provenance, upstream license
tests/                  Backend privacy/auth tests, TF.js parity, browser checks
render.yaml             Render web service and PostgreSQL configuration
```

Technical references: [MediaPipe Hand Landmarker](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker/web_js), [TensorFlow.js models](https://www.tensorflow.org/js/guide/models_and_layers), [Flask-WTF CSRF](https://flask-wtf.readthedocs.io/en/1.2.x/csrf/). ASL learning reference: [ASL University / Lifeprint](https://www.lifeprint.com/asl101/fingerspelling/fingerspelling.htm). External reference materials are linked, not bundled as locally authored demonstrations.
