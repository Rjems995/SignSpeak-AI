// Classic dedicated worker: the pinned WASM loader calls importScripts(),
// which is unavailable in a module worker. No camera frames leave this worker.
const MP_BASE = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const LANDMARK_MODEL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
let landmarker = null;
let initializing = null;
let detecting = false;

async function init() {
  if (!initializing)
    initializing = (async () => {
      const vision = await import(`${MP_BASE}/vision_bundle.mjs`);
      const fileset = await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
      landmarker = await vision.HandLandmarker.createFromOptions(fileset, {
        baseOptions: { modelAssetPath: LANDMARK_MODEL, delegate: 'CPU' },
        runningMode: 'VIDEO',
        numHands: 2,
        minHandDetectionConfidence: 0.6,
        minHandPresenceConfidence: 0.6,
        minTrackingConfidence: 0.6,
      });
    })();
  await initializing;
}

self.onmessage = async (event) => {
  const { type, id, frame, timestamp } = event.data || {};
  try {
    if (type === 'init') {
      await init();
      self.postMessage({ type: 'ready', id });
      return;
    }
    if (type !== 'detect') throw new Error('Unknown hand tracking request.');
    if (!landmarker) throw new Error('Hand tracking is not initialized.');
    if (detecting) throw new Error('A detection is already in flight.');
    if (
      !(frame instanceof ImageBitmap) ||
      !frame.width ||
      !frame.height ||
      !Number.isFinite(timestamp)
    )
      throw new Error('An open frame and finite timestamp are required.');
    detecting = true;
    const result = landmarker.detectForVideo(frame, timestamp);
    self.postMessage({
      type: 'result',
      id,
      result: {
        landmarks: result.landmarks,
        worldLandmarks: result.worldLandmarks,
        handednesses: result.handednesses ?? result.handedness,
      },
    });
  } catch (error) {
    self.postMessage({
      type: 'error',
      id,
      error: { name: error.name || 'Error', message: error.message || 'Hand tracking failed.' },
    });
  } finally {
    if (type === 'detect') {
      detecting = false;
      frame?.close?.();
    }
  }
};
