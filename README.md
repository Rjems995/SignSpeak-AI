# SignSpeak AI

A Flask application for learning and recognizing **six static American Sign Language alphabet letters: A, B, C, L, V, and Y**. MediaPipe tracks one hand, and a real trained neural classifier runs through TensorFlow.js in the browser. Recognition builds editable text; browser speech synthesis reads it aloud.

This is an experimental alphabet MVP. It does not translate ASL sentences, recognize motion letters J/Z, or interpret facial expressions, grammar, or context. ASL and FSL are separate languages; this application supports ASL only.

## What works

- Responsive landing page, dashboard, light/dark themes, and mobile navigation.
- Permission-based webcam access; real 21-point MediaPipe tracking; optional landmark overlay.
- A trained 42 → 64 → 32 → 7 MLP: six letters plus an unsupported-handshape class.
- Shared wrist/scale/rotation/mirror normalization, confidence and distance gates, temporal smoothing, hold duration, and release-before-repeat behavior.
- Editable text, space/delete/clear/copy, and explicit saving to private history.
- English text-to-speech, available voice selection, speed, pause/resume, and stop.
- Six ASL lesson cards with written steps, recorded landmark reference drawings, and links to ASL University's visual demonstrations.
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

The starter model is trained from [SigNN's public MediaPipe coordinate samples](https://github.com/AriAlavi/SigNN/blob/40bd651899e09a0099767457194062f77f774156/scripts/mediapipe_data_creation/training_data.json), pinned to revision `40bd651899e09a0099767457194062f77f774156`. The upstream Apache 2.0 license and source provenance are in `datasets/`.

After canonical-coordinate deduplication there are **622 unique samples**: 372 training, 125 validation, and 125 test. Splits precede augmentation; only training samples are augmented. The test set stays untouched by fitting and distance-gate tuning.

| Measure | Result |
| --- | --- |
| Raw classifier accuracy across six letters plus unsupported examples | 89.6% on 125 samples |
| Raw accuracy on the six supported letters alone | **76.5% on 34 samples** |
| Gated classification accuracy, default 85% threshold | 92.0% across all 125 samples |

The overall figures contain 91 unsupported test examples. They must not be presented as six-letter accuracy. The model especially struggles with V: raw test recall is 25% on four samples. The evaluation report includes per-class precision, recall, F1, raw and gated confusion matrices, split fingerprints, and all held-out predictions. See:

- `signspeak/static/models/evaluation.json`
- `signspeak/static/models/confusion-matrix.png`
- `signspeak/static/models/metadata.json`
- [Model card](MODEL_CARD.md)

**No signer IDs are supplied by this historical dataset. These sample-holdout results do not establish accuracy on unseen people.** Similar frames or samples from the same person may still cross the sample split even after exact/canonical deduplication. The historical landmark extractor may also differ from the current browser model. Collect new data and run a participant-disjoint evaluation before relying on recognition.

Confidence is the model's highest class probability, not a calibrated probability of correct signing. Unsupported handshapes are rejected using a trained background class and a validation-tuned distance gate, but rejection is incomplete. A closed-set model can still confidently misread unfamiliar gestures. Human review of generated text remains necessary.

## Reproduce training and evaluation

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-ml.txt
# Optional: re-download the pinned public source and license.
.\.venv\Scripts\python.exe -m ml.download_dataset
.\.venv\Scripts\python.exe -m ml.prepare datasets/signn.json
.\.venv\Scripts\python.exe -m ml.train
.\.venv\Scripts\python.exe -m ml.evaluate
```

The classifier is trained with scikit-learn's MLP implementation. Its learned dense kernels, biases, and scaler are exported to JSON and evaluated with TensorFlow.js matrix operations in the browser; no Python inference or video upload is involved. JavaScript tests verify that the exported TF.js network matches the original held-out probabilities. This avoids installing TensorFlow on the application server.

`ml.train` writes a new classifier, model metadata, measured test results, reference samples, and confusion-matrix image. Restart or refresh the app afterward. Keep the scripts' random seeds and use the locked development versions in `requirements-lock.txt` to reproduce the checked-in artifacts. Retraining with future library versions can change the results.

If Matplotlib cannot write its user cache, set a workspace-local directory first:

```powershell
$env:MPLCONFIGDIR = "$PWD\artifacts\matplotlib"
```

### Collect a better dataset

1. Visit **http://127.0.0.1:5000/capture**. Give each consenting signer a pseudonymous participant ID. The form's consent checkbox specifically covers collecting hand coordinates.
2. Enable the camera. Choose a letter, hold the correct shape, and capture individual samples while varying distance, hand position, lighting, and sessions. Check the labels with a knowledgeable ASL signer.
3. Collect unsupported shapes with the **Other / unsupported handshape** option. Include near-confusable signs, relaxed hands, and partially formed letters. Aim for balanced supported classes and diverse negative examples.
4. Download each JSON file into `datasets/private/`. This folder is ignored by Git. Only coordinates are downloaded; no video is recorded.
5. Collect every class from enough independent participants that training, validation, and test sets each include all classes. Start with at least 10 participants; more diversity is preferable.
6. Train **only on your identified-participant dataset** to get a participant-disjoint evaluation:

```powershell
.\.venv\Scripts\python.exe -m ml.prepare datasets/private/signer01.json datasets/private/signer02.json datasets/private/signer03.json --output datasets/private/prepared.json
.\.venv\Scripts\python.exe -m ml.train --dataset datasets/private/prepared.json
```

The training script uses participant-separated train/validation/test groups when every sample has an ID and there are at least three distinct participants. It stops if any split lacks a required class. With missing IDs it explicitly reports a sample-based split. Combining unidentified public samples with private captures therefore does **not** produce a signer-independent evaluation. Never identify each frame as a different participant.

To extract landmarks from consented still images, organize files as `INPUT/participant/LETTER/*.jpg`. Install the optional extractor:

```powershell
.\.venv\Scripts\python.exe -m pip install mediapipe
Invoke-WebRequest 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task' -OutFile 'datasets/private/hand_landmarker.task'
.\.venv\Scripts\python.exe -m ml.extract_images datasets/private/photos --hand-model datasets/private/hand_landmarker.task --output datasets/private/images.json
.\.venv\Scripts\python.exe -m ml.prepare datasets/private/images.json --output datasets/private/prepared.json
```

Create `datasets/private/` before downloading if it does not exist. The extractor keeps images with exactly one detected hand. Retain participant grouping and permission records separately; do not commit private images.

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
