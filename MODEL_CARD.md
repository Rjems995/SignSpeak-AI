# SignSpeak ASL alphabet models

**Coverage:** all 26 letters. A-I and K-Y use a static network. J/Z require a short recorded movement sequence. Models support experimental, human-reviewed fingerspelling and informal practice, not full language translation or proficiency assessment.

**Static architecture:** StandardScaler -> dense 96 ReLU -> dense 64 ReLU -> dense 24 softmax. Input is 21 image-space XY landmarks (42 values), wrist-centered, palm-length scaled, and mirrored by palm geometry. Image-plane direction is deliberately preserved: arbitrary rotation invariance erases distinctions such as K/P, G/Q, and H/U. Feature version: `wrist-scale-orientation-xy-v2`.

**Motion architecture:** StandardScaler -> dense 48 ReLU -> dense 24 ReLU -> two-class softmax (equivalent to the trained binary sigmoid). Real timestamped sequences are trimmed to their active interval and resampled at 16 time steps for five joints: wrist, thumb tip, index tip, middle tip, little tip. Coordinates remain relative to the initial wrist, with median palm scaling and mirror handling. Stationary, short, non-finite, and severely interrupted sequences are rejected. Feature version: `temporal-palm-scale-16x5xy-v1`.

**Static source/splits:** the pinned SigNN Apache-2.0 coordinate file in `datasets/`. Canonical deduplication precedes per-class sample splitting: 622 unique samples, 373 train, 124 validation, 125 test. Q has just two examples (one training, one test), no validation examples, and a fixed conservative gate. Other static gates use validation-only distances. Training augmentation happens after splitting.

**Motion source/splits:** a deterministic subset of SigNN's CC0 Kaggle J/Z video dataset. Forty-eight clips downloaded, 46 passed actual MediaPipe tracking/movement checks. Split by source clip into 27 train, 9 validation, 10 test. No frames or augmentations from a source clip cross splits. Each file's upstream name and SHA-256 are in `datasets/motion-provenance.json`; rejection reasons are in `datasets/motion.json`. Gates use validation clips. Reference animations use training clips.

**Measured performance:** static raw accuracy 74.4%, gated accuracy 68.8% over 125 held-out samples. Motion raw accuracy 100% over only ten held-out clips, gated accuracy 70% over those clips. Gated accuracy counts rejected valid signs as mistakes. These figures replace the earlier six-letter results. Do not describe the ten-clip result as universal or signer-independent accuracy. Q's single held-out sample was not correctly classified.

**Rejection:** confidence threshold (default .85), nearest training-support distance, one-hand requirement, static temporal consensus/hold/release, and an explicit movement gate. The expanded datasets have no held-out unsupported class, so unknown-pose/movement false-acceptance rates are unmeasured. The classifiers can still confidently accept unfamiliar shapes or motions. Confidence is not calibrated.

**Limits:** small uneven classes; very sparse Q, N, M and similar closed-hand shapes; missing signer/session IDs; possible correlated sample/clip leakage; older static versus current live landmark-extractor differences; no depth features. No independent tests establish behavior across skin tones, mobility differences, handedness, lighting, camera devices, or new signers. Mirror handling is mathematical preprocessing, not an accessibility/fairness evaluation.

**Privacy:** camera frames stay on the device. Live J/Z coordinates are temporarily buffered during explicit recording. Dataset capture requires separate consent; downloads contain coordinates and pseudonymous participant IDs, not footage.

**Reproduce:** see README.md. Python and TF.js parity tests cover both feature pipelines and every held-out probability. Split-isolation tests use static fingerprints and source-clip IDs. A complete next evaluation needs consented, participant-disjoint data, balanced letters, and diverse unsupported handshapes/movements reviewed with ASL signers and Deaf educators.
