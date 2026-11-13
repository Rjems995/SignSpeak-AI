// Real worker inference uses Google's public sample; responsiveness uses a
// bounded busy worker so the UI test does not depend on machine inference speed.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:5000';
const image = await fetch(
  'https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg',
);
assert.equal(image.status, 200);
const bytes = Buffer.from(await image.arrayBuffer());
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
});
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } }),
    errors = [],
    workers = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('worker', (worker) => workers.push(worker.url()));
  await page.route('**/__worker-hand.jpg', (route) =>
    route.fulfill({ body: bytes, contentType: 'image/jpeg' }),
  );
  await page.goto(base + '/translate', { waitUntil: 'networkidle' });
  const real = await page.evaluate(async () => {
    const { HandTracker } = await import('/static/js/hand-tracker.js');
    const tracker = new HandTracker(),
      started = performance.now();
    await tracker.init();
    const ready = performance.now();
    const bitmap = await createImageBitmap(
      await fetch('/__worker-hand.jpg').then((response) => response.blob()),
    );
    let ticks = 0;
    const timer = setInterval(() => ticks++, 10);
    try {
      const detectionStart = performance.now();
      const result = await tracker.detect(bitmap, detectionStart);
      return {
        hands: result.landmarks.length,
        points: result.landmarks.map((hand) => hand.length),
        world: result.worldLandmarks.map((hand) => hand.length),
        handedness: result.handednesses.length,
        initMs: ready - started,
        detectMs: performance.now() - detectionStart,
        uiTicks: ticks,
        transferredWidth: bitmap.width,
      };
    } finally {
      clearInterval(timer);
      tracker.close();
    }
  });
  assert.ok(workers.some((url) => url.endsWith('/static/js/hand-tracking-worker.js')));
  assert.ok(real.hands > 0);
  assert.ok(real.points.every((count) => count === 21));
  assert.ok(real.world.every((count) => count === 21));
  assert.equal(real.handedness, real.hands);
  assert.equal(real.transferredWidth, 0);

  await page.evaluate(async () => {
    const { HandTracker } = await import('/static/js/hand-tracker.js');
    const { CameraController } = await import('/static/js/camera.js');
    const source = `onmessage=event=>{
      const {type,id,frame}=event.data;
      if(type==='init'){postMessage({type:'ready',id});return;}
      postMessage({type:'busy'});
      const end=performance.now()+1000;while(performance.now()<end){}
      frame.close();postMessage({type:'result',id,result:{landmarks:[],worldLandmarks:[],handednesses:[]}});
    };`;
    const url = URL.createObjectURL(new Blob([source], { type: 'text/javascript' }));
    const probe = (window.workerProbe = {
      busy: false,
      workerCompleted: false,
      done: false,
      ticks: 0,
      url,
    });
    const tracker = new HandTracker({
      workerFactory: () => {
        const worker = new Worker(url);
        worker.addEventListener('message', (event) => {
          if (event.data.type === 'busy') probe.busy = true;
          if (event.data.type === 'result') {
            probe.workerCompleted = true;
            probe.completedAt = performance.now();
          }
        });
        return worker;
      },
    });
    await tracker.init();
    const canvas = document.createElement('canvas');
    canvas.width = 640;
    canvas.height = 480;
    const camera = new CameraController({ captureFrame: () => createImageBitmap(canvas) });
    camera.tracker = tracker;
    camera.running = true;
    camera.video = { readyState: 2, currentTime: 1, pause() {}, srcObject: null };
    camera.lastFrame = -Infinity;
    camera.lastVideoTime = -1;
    probe.camera = camera;
    probe.timer = setInterval(() => probe.ticks++, 10);
    document
      .getElementById('translation-text')
      .addEventListener('input', () => (probe.inputAt = performance.now()), { once: true });
    probe.pending = camera.frame().finally(() => (probe.done = true));
  });
  await page.waitForFunction(() => window.workerProbe.busy);
  await page.locator('#translation-text').fill('Editing while hand tracking is busy.');
  await page.click('#theme-toggle');
  const during = await page.evaluate(() => ({
    done: window.workerProbe.workerCompleted,
    theme: document.documentElement.dataset.theme,
  }));
  assert.equal(
    during.done,
    false,
    'editing and theme changes must complete while the worker is still busy',
  );
  assert.equal(during.theme, 'dark');
  await page.waitForFunction(() => window.workerProbe.done);
  const stress = await page.evaluate(async () => {
    const probe = window.workerProbe;
    await probe.pending;
    clearInterval(probe.timer);
    probe.camera.stop();
    URL.revokeObjectURL(probe.url);
    return {
      uiTicks: probe.ticks,
      inputAt: probe.inputAt,
      workerCompletedAt: probe.completedAt,
      text: document.getElementById('translation-text').value,
    };
  });
  assert.ok(stress.uiTicks >= 5);
  assert.ok(stress.inputAt < stress.workerCompletedAt);
  assert.equal(stress.text, 'Editing while hand tracking is busy.');
  assert.deepEqual(errors, []);
  await fs.mkdir('artifacts', { recursive: true });
  await fs.writeFile(
    'artifacts/worker-responsiveness.json',
    JSON.stringify(
      {
        real,
        stress,
        method:
          'Public still-image inference in the actual MediaPipe worker plus a one-second busy-worker UI isolation check; not a device-wide FPS benchmark.',
      },
      null,
      2,
    ),
  );
  console.log(
    'Worker browser checks passed: real image/world/handedness results, transferable frames, and editable responsive UI during a busy detection.',
  );
  console.log(JSON.stringify({ real, stress }));
} finally {
  await browser.close();
}
