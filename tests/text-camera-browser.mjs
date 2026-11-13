// Camera integration uses Chrome's fake video device only. Permission races use
// deferred fixture streams and never call the browser's camera API.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:5000';
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
});
const errors = [];
const phone = { width: 390, height: 844 };
const cameraTab = '.workspace-tabs [data-workspace-view="camera"]';
const textTab = '.workspace-tabs [data-workspace-view="text"]';

async function instrument(context, { deferred = false, theme = 'light' } = {}) {
  await context.addInitScript(
    ({ deferred, theme }) => {
      localStorage.setItem('signspeak:settings:guest', JSON.stringify({ theme }));
      const state = {
        streams: [],
        permissionRequests: 0,
        workersCreated: 0,
        workersTerminated: 0,
        lateTracksStopped: 0,
      };
      window.__textCameraTest = state;
      const NativeWorker = window.Worker;
      window.Worker = class extends NativeWorker {
        constructor(...args) {
          super(...args);
          state.workersCreated++;
        }
        terminate() {
          state.workersTerminated++;
          return super.terminate();
        }
      };
      const original = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
      navigator.mediaDevices.getUserMedia = deferred
        ? () => {
            state.permissionRequests++;
            return new Promise((resolve, reject) => {
              const track = {
                readyState: 'live',
                stop() {
                  this.readyState = 'ended';
                  state.lateTracksStopped++;
                },
              };
              state.resolvePermission = () => resolve({ getTracks: () => [track] });
              state.rejectPermission = () =>
                reject(new DOMException('Fixture permission denied', 'NotAllowedError'));
            });
          }
        : async (constraints) => {
            state.permissionRequests++;
            const stream = await original(constraints);
            state.streams.push(stream);
            return stream;
          };
    },
    { deferred, theme },
  );
}

async function openTranslator(context) {
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('https://fonts.googleapis.com/**', (route) => route.abort());
  await page.goto(base + '/translate', { waitUntil: 'networkidle' });
  await page.waitForFunction(() =>
    document.querySelector('.workspace-tabs [aria-selected="true"]'),
  );
  return page;
}

async function expectStatus(page, text, enabled, timeout = 10000) {
  await page.waitForFunction(
    ({ text, enabled }) => {
      const original = document.getElementById('camera-status');
      const mirror = document.getElementById('text-camera-status');
      return (
        original?.textContent.trim() === text &&
        mirror?.textContent.trim() === text &&
        document.getElementById('camera-stop').disabled === !enabled &&
        document.getElementById('text-camera-stop').disabled === !enabled
      );
    },
    { text, enabled },
    { timeout },
  );
  const dots = await page.evaluate(() =>
    ['camera-status', 'text-camera-status'].map((id) => {
      const dot = document.getElementById(id).querySelector('.status-dot');
      return { className: dot?.className, hiddenFromAT: dot?.getAttribute('aria-hidden') };
    }),
  );
  assert.deepEqual(dots[1], dots[0], `${text}: mirror the canonical status indicator`);
}

async function expectTextSelected(page) {
  assert.equal(await page.locator(textTab).getAttribute('aria-selected'), 'true');
  assert.equal(await page.locator('#text-workspace').isVisible(), true);
  assert.equal(await page.locator('#camera-workspace').isVisible(), false);
  assert.equal(await page.locator('#text-camera-controls').isVisible(), true);
}

async function controlGeometry(page) {
  return page.locator('#text-camera-stop').evaluate((button) => {
    const box = button.getBoundingClientRect();
    const row = document.getElementById('text-camera-controls').getBoundingClientRect();
    const hit = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
    return {
      button: {
        x: box.x,
        y: box.y,
        width: box.width,
        height: box.height,
        right: box.right,
        bottom: box.bottom,
      },
      row: { x: row.x, y: row.y, right: row.right, bottom: row.bottom },
      hit: hit === button || button.contains(hit),
      width: innerWidth,
      height: innerHeight,
      scrollY,
      documentWidth: document.documentElement.scrollWidth,
      documentHeight: document.documentElement.scrollHeight,
    };
  });
}

async function expectReachable(page, label) {
  const geometry = await controlGeometry(page);
  assert.ok(
    geometry.button.width >= 43.5 && geometry.button.height >= 43.5,
    `${label}: Stop needs a 44px touch target: ${JSON.stringify(geometry)}`,
  );
  assert.ok(
    geometry.button.x >= -1 &&
      geometry.button.y >= -1 &&
      geometry.button.right <= geometry.width + 1 &&
      geometry.button.bottom <= geometry.height + 1 &&
      geometry.row.x >= -1 &&
      geometry.row.y >= -1 &&
      geometry.row.right <= geometry.width + 1 &&
      geometry.row.bottom <= geometry.height + 1,
    `${label}: text camera controls must stay in the viewport: ${JSON.stringify(geometry)}`,
  );
  assert.equal(geometry.hit, true, `${label}: Stop must remain uncovered`);
  assert.ok(
    geometry.documentWidth <= geometry.width + 1 && geometry.documentHeight <= geometry.height + 1,
    `${label}: the document must fit the viewport: ${JSON.stringify(geometry)}`,
  );
  assert.equal(geometry.scrollY, 0, `${label}: document must not scroll`);
  return geometry;
}

async function expectFixedWhilePanelScrolls(page, label, mustScroll = false) {
  await page.locator('#text-workspace').evaluate((panel) => (panel.scrollTop = 0));
  const before = await expectReachable(page, `${label} at panel start`);
  const scroll = await page.locator('#text-workspace').evaluate((panel) => {
    panel.scrollTop = panel.scrollHeight;
    return { top: panel.scrollTop, height: panel.clientHeight, content: panel.scrollHeight };
  });
  if (mustScroll)
    assert.ok(
      scroll.top > 0,
      `${label}: exercise actual internal scrolling: ${JSON.stringify(scroll)}`,
    );
  const after = await expectReachable(page, `${label} at panel end`);
  assert.deepEqual(after.row, before.row, `${label}: panel scrolling must not move the camera row`);
  assert.deepEqual(
    after.button,
    before.button,
    `${label}: panel scrolling must not move the Stop button`,
  );
}

try {
  const context = await browser.newContext({ viewport: phone, isMobile: true, hasTouch: true });
  await instrument(context);
  const page = await openTranslator(context);
  assert.equal(await page.locator(cameraTab).getAttribute('aria-selected'), 'true');
  assert.equal(await page.locator('#text-camera-controls').isVisible(), false);
  await page.locator(textTab).click();
  await expectTextSelected(page);
  await expectStatus(page, 'Camera off', false);
  await expectReachable(page, 'initial camera off');

  await page.locator(cameraTab).click();
  await page.locator('#camera-start').click();
  await expectStatus(page, 'Camera on', true, 120000);
  await page.locator(textTab).click();
  await expectTextSelected(page);
  await expectStatus(page, 'Camera on', true);
  const message = 'Keep this message when I stop the camera.';
  await page.locator('#translation-text').fill(message);
  await expectFixedWhilePanelScrolls(page, '390x844 camera on');
  const active = await page.evaluate(() => ({
    workers: window.__textCameraTest.workersCreated,
    terminated: window.__textCameraTest.workersTerminated,
    streams: window.__textCameraTest.streams.length,
    tracks: window.__textCameraTest.streams.flatMap((stream) =>
      stream.getTracks().map((track) => track.readyState),
    ),
    running: document.getElementById('camera-viewport').classList.contains('running'),
  }));
  assert.ok(active.workers >= 1, 'the integration must start a real hand-tracking worker');
  assert.equal(active.streams, 1);
  assert.ok(active.tracks.length > 0 && active.tracks.every((state) => state === 'live'));
  assert.equal(active.running, true);
  await fs.mkdir('artifacts', { recursive: true });
  await page.screenshot({ path: 'artifacts/text-camera-mobile.png', fullPage: true });

  await page.locator('#text-camera-stop').click();
  await expectStatus(page, 'Camera off', false);
  const stopped = await page.evaluate(() => ({
    terminated: window.__textCameraTest.workersTerminated,
    tracks: window.__textCameraTest.streams.flatMap((stream) =>
      stream.getTracks().map((track) => track.readyState),
    ),
    running: document.getElementById('camera-viewport').classList.contains('running'),
    videoCleared: document.getElementById('webcam').srcObject === null,
  }));
  assert.ok(
    stopped.terminated > active.terminated,
    'Text Stop must terminate the inference worker',
  );
  assert.ok(
    stopped.tracks.every((state) => state === 'ended'),
    'Text Stop must release every track',
  );
  assert.equal(stopped.running, false);
  assert.equal(stopped.videoCleared, true);
  assert.equal(await page.locator('#translation-text').inputValue(), message);
  await expectTextSelected(page);

  await page.locator(cameraTab).click();
  await page.locator('#camera-start').click();
  await expectStatus(page, 'Camera on', true, 120000);
  await page.locator(textTab).click();
  await expectTextSelected(page);
  await expectStatus(page, 'Camera on', true);
  assert.equal(await page.locator('#translation-text').inputValue(), message);
  assert.equal(await page.evaluate(() => window.__textCameraTest.streams.length), 2);
  for (const viewport of [{ width: 320, height: 568 }, { width: 844, height: 390 }, phone]) {
    await page.setViewportSize(viewport);
    await expectTextSelected(page);
    await expectStatus(page, 'Camera on', true);
    await expectFixedWhilePanelScrolls(
      page,
      `${viewport.width}x${viewport.height} camera on`,
      viewport.height === 390,
    );
  }

  await page.locator('#text-camera-stop').focus();
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.waitForFunction(() => document.activeElement?.id === 'camera-stop');
  assert.equal(await page.locator('#text-camera-controls').isVisible(), false);
  assert.equal(await page.locator('#camera-workspace').isVisible(), true);
  assert.equal(await page.locator('#text-workspace').isVisible(), true);
  assert.equal(await page.locator('#camera-stop').isEnabled(), true);
  assert.equal(await page.locator('#translation-text').inputValue(), message);
  await page.locator('#camera-stop').click();
  await expectStatus(page, 'Camera off', false);
  await context.close();

  const deferredContext = await browser.newContext({
    viewport: phone,
    isMobile: true,
    hasTouch: true,
  });
  await instrument(deferredContext, { deferred: true, theme: 'dark' });
  const pending = await openTranslator(deferredContext);
  assert.equal(await pending.getAttribute('html', 'data-theme'), 'dark');
  await pending.locator('#camera-start').click();
  await expectStatus(pending, 'Requesting permission', true);
  await pending.locator(textTab).click();
  await expectTextSelected(pending);
  await expectReachable(pending, 'dark pending permission');
  await pending.locator('#translation-text').fill(message);
  await pending.locator('#text-camera-stop').click();
  await expectStatus(pending, 'Camera off', false);
  await pending.evaluate(() => window.__textCameraTest.resolvePermission());
  await pending.waitForFunction(() => window.__textCameraTest.lateTracksStopped === 1);
  await expectStatus(pending, 'Camera off', false);
  assert.deepEqual(
    await pending.evaluate(() => ({
      requests: window.__textCameraTest.permissionRequests,
      workers: window.__textCameraTest.workersCreated,
      running: document.getElementById('camera-viewport').classList.contains('running'),
      videoCleared: document.getElementById('webcam').srcObject === null,
    })),
    { requests: 1, workers: 0, running: false, videoCleared: true },
    'a permission result arriving after Text Stop must release its stream before starting tracking',
  );
  assert.equal(await pending.locator('#translation-text').inputValue(), message);
  await expectTextSelected(pending);

  await pending.locator(cameraTab).click();
  await pending.locator('#camera-start').click();
  await expectStatus(pending, 'Requesting permission', true);
  await pending.locator(textTab).click();
  await pending.evaluate(() => window.__textCameraTest.rejectPermission());
  await expectStatus(pending, 'Camera unavailable', false);
  assert.match(await pending.locator('#camera-error').textContent(), /permission was denied/i);
  assert.equal(await pending.locator('#camera-start').isEnabled(), true);
  assert.equal(await pending.evaluate(() => window.__textCameraTest.workersCreated), 0);
  assert.equal(await pending.locator('#translation-text').inputValue(), message);
  await expectTextSelected(pending);
  await expectReachable(pending, 'dark permission denied');
  await deferredContext.close();
  assert.deepEqual(errors, [], 'browser JavaScript errors');
  console.log(
    'Text camera controls passed: mirrored off/on/pending/error states, fake-camera worker and stream cleanup, restart, fixed mobile controls, desktop focus transfer, and late permission cancellation. No real camera was activated.',
  );
} finally {
  await browser.close();
}
