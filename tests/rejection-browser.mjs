// Use real reviewed public coordinates with a fake camera; no webcam recording.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:5015';
const evaluation = JSON.parse(await fs.readFile('datasets/unknown-evaluation.json', 'utf8'));
const rows = JSON.parse(await fs.readFile('datasets/unknown-hands.json', 'utf8')).samples;
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
});
try {
  const page = await browser.newPage({ viewport: { width: 1280, height: 900 } }),
    errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(base + '/translate', { waitUntil: 'networkidle' });
  await page.click('#camera-start');
  await page.waitForFunction(
    () => document.getElementById('camera-status').textContent.includes('Camera on'),
    undefined,
    { timeout: 90000 },
  );
  await page.click('#camera-stop');
  const examples = evaluation.test_examples
    .filter((r) => !r.accepted)
    .map((r) => rows.find((row) => row.fingerprint === r.fingerprint));
  const result = await page.evaluate(async (examples) => {
    const { CameraController } = await import('/static/js/camera.js');
    const { LandmarkClassifier, Stabilizer } = await import('/static/js/recognition.js');
    const data = await fetch('/static/models/classifier.json').then((r) => r.json());
    const camera = new CameraController({
      captureFrame: async () => ({ close() {} }),
      onCommit: (letter) => {
        document.getElementById('translation-text').value += letter;
      },
    });
    camera.classifier = new LandmarkClassifier(data, window.tf);
    camera.stabilizer = new Stabilizer(500);
    camera.running = true;
    camera.video = { readyState: 2, currentTime: 1 };
    let points = examples[0].points;
    camera.tracker = {
      detect: async () => ({
        landmarks: [
          Array.from({ length: 21 }, (_, i) => ({ x: points[i * 2], y: points[i * 2 + 1] })),
        ],
      }),
      close() {},
    };
    let checked = 0;
    const tick = async () => {
      camera.lastFrame = -Infinity;
      camera.video.currentTime++;
      await camera.frame();
      cancelAnimationFrame(camera.raf);
    };
    for (const row of examples) {
      points = row.points;
      camera.stabilizer.reset();
      await tick();
      if (camera.lastResult.label) throw Error('Rejected public coordinates became a letter');
      if (document.getElementById('prediction-label').textContent !== 'Sign not recognized.')
        throw Error('Unknown message missing');
      if (document.getElementById('confidence-value').textContent !== '—')
        throw Error('Rejected hand shows a confident percentage');
      checked++;
    }
    // Keep a real rejected hand visible beyond the hold duration.
    for (let i = 0; i < 15; i++) {
      await tick();
      await new Promise((resolve) => setTimeout(resolve, 90));
    }
    const text = document.getElementById('translation-text').value;
    // Invalid tracked geometry must reject without stopping the camera.
    points = Array(42).fill(0);
    await tick();
    const running = camera.running,
      invalid = camera.lastResult.label;
    // Rejected/invalid J/Z sequences must clear stale accepted results too.
    camera.motionClassifier = {
      predict: () => ({ label: null, confidence: 0.99, candidate: 'J' }),
    };
    camera.recording = true;
    camera.motionFrames = [];
    camera.lastMotion = { label: 'J' };
    camera.finishMotion();
    const motionLabel = document.getElementById('prediction-label').textContent;
    camera.motionClassifier = {
      predict: () => {
        throw Error('Movement too short.');
      },
    };
    camera.recording = true;
    camera.lastMotion = { label: 'Z' };
    camera.finishMotion();
    const cleared = camera.lastMotion === null && camera.lastResult.label === null;
    // No detected hand has its own prompt.
    camera.tracker = { detect: async () => ({ landmarks: [] }), close() {} };
    await tick();
    const absent = document.getElementById('prediction-label').textContent;
    camera.running = false;
    cancelAnimationFrame(camera.raf);
    return { checked, text, running, invalid, motionLabel, cleared, absent };
  }, examples);
  assert.equal(result.checked, 46);
  assert.equal(result.text, '');
  assert.equal(result.running, true);
  assert.equal(result.invalid, null);
  assert.equal(result.motionLabel, 'Sign not recognized.');
  assert.equal(result.cleared, true);
  assert.equal(result.absent, 'Looking for your hand');
  assert.deepEqual(errors, []);
  await page.setViewportSize({ width: 390, height: 844 });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  console.log(
    'Rejection browser checks passed: 46 real held-out poses, unknown message, hidden confidence, no text commits, malformed geometry, cleared J/Z results, absent-hand prompt and mobile layout.',
  );
} finally {
  await browser.close();
}
