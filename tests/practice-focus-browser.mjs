// Sessions run through the real practice UI and classifier. A canvas media
// stream and a fixture inference worker keep camera input deterministic and local.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:5000';
const references = JSON.parse(
  await fs.readFile(new URL('../signspeak/static/models/references.json', import.meta.url), 'utf8'),
);
const tensorflow = await fs.readFile(
  new URL('../node_modules/@tensorflow/tfjs/dist/tf.min.js', import.meta.url),
  'utf8',
);
const points = Array.from({ length: 21 }, (_, index) => ({
  x: references.L[index * 2],
  y: references.L[index * 2 + 1],
  z: 0,
}));
const worker = `
  const points = ${JSON.stringify(points)};
  onmessage = ({ data }) => {
    if (data.type === 'init') postMessage({ type: 'ready', id: data.id });
    if (data.type === 'detect') {
      data.frame.close();
      postMessage({ type: 'result', id: data.id, result: {
        landmarks: [points], worldLandmarks: [],
        handednesses: [[{ categoryName: 'Right', score: 1 }]],
      }});
    }
  };
`;
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
});
const errors = [];
let currentPage;

async function openPractice(context, path = '/practice') {
  const page = await context.newPage();
  currentPage = page;
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('https://fonts.googleapis.com/**', (route) => route.abort());
  await page.route('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js', (route) =>
    route.fulfill({ body: tensorflow, contentType: 'text/javascript' }),
  );
  await page.route('**/static/js/hand-tracking-worker.js', (route) =>
    route.fulfill({ body: worker, contentType: 'text/javascript' }),
  );
  await page.goto(base + path, { waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.getElementById('practice-focus'));
  return page;
}

async function cameraOn(page) {
  await page.locator('#camera-start').click();
  await page.waitForFunction(() => {
    const status = document.getElementById('camera-status').textContent.trim();
    return status === 'Camera on' || !document.getElementById('camera-error').hidden;
  });
  assert.equal(
    (await page.locator('#camera-status').textContent()).trim(),
    'Camera on',
    await page.locator('#camera-error').textContent(),
  );
}

async function begin(page, focus) {
  await page.locator('#practice-focus').selectOption(focus);
  await cameraOn(page);
  assert.equal(await page.locator('#practice-start').isDisabled(), false);
  await page.locator('#practice-start').click();
  await page.waitForFunction(() => !document.getElementById('practice-check').hidden);
  assert.equal(await page.locator('#practice-focus').inputValue(), focus);
  assert.equal(await page.locator('#practice-focus').isDisabled(), true);
}

async function expectFocusControl(page) {
  const detail = await page.locator('#practice-focus').evaluate((control) => {
    const rect = control.getBoundingClientRect();
    return {
      labelled: Boolean(control.labels?.length),
      width: rect.width,
      height: rect.height,
      left: rect.left,
      right: rect.right,
      viewport: innerWidth,
      documentWidth: document.documentElement.scrollWidth,
    };
  });
  assert.equal(detail.labelled, true, 'the focus selector has an associated label');
  assert.ok(detail.width >= 100 && detail.height >= 43.5, JSON.stringify(detail));
  assert.ok(detail.left >= -1 && detail.right <= detail.viewport + 1, JSON.stringify(detail));
  assert.ok(detail.documentWidth <= detail.viewport + 1, 'practice must not overflow horizontally');
}

async function prompt(page, number, total, allowed) {
  assert.match(
    await page.locator('#practice-step').textContent(),
    new RegExp(`LETTER ${number} OF ${total}`),
  );
  const letter = (await page.locator('#practice-letter').textContent()).trim();
  assert.ok(allowed.includes(letter), `unexpected target ${letter}; expected ${allowed}`);
  assert.equal(await page.locator('#practice-focus').isDisabled(), true);
  assert.equal(await page.locator('#practice-skip').isVisible(), true);
  assert.equal(await page.locator('#practice-check').textContent(), 'Check my sign');
  return letter;
}

async function skipAndNext(page) {
  await page.locator('#practice-skip').click();
  await page.locator('#practice-next').click();
}

async function expectFinished(page, attempts) {
  await page.waitForFunction(() => !document.getElementById('session-summary').hidden);
  assert.equal(await page.locator('#summary-attempts').textContent(), String(attempts));
  assert.equal(await page.locator('#practice-focus').isDisabled(), false);
  assert.equal(await page.locator('#practice-check').isVisible(), false);
  assert.equal((await page.locator('#camera-status').textContent()).trim(), 'Camera off');
  const tracks = await page.evaluate(() =>
    window.__practiceTest.streams.flatMap((stream) =>
      stream.getTracks().map((track) => track.readyState),
    ),
  );
  assert.ok(tracks.length > 0 && tracks.every((state) => state === 'ended'));
}

function expectBalanced(letters, pair) {
  assert.equal(letters.length, 6);
  for (const letter of pair) assert.equal(letters.filter((value) => value === letter).length, 3);
}

try {
  await fs.mkdir('artifacts', { recursive: true });
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript(() => {
    localStorage.setItem(
      'signspeak:settings:guest',
      JSON.stringify({ hold_ms: 300, threshold: 0.85, landmarks: false }),
    );
    window.__practiceTest = { streams: [], frames: 0 };
    navigator.mediaDevices.getUserMedia = async () => {
      const canvas = document.createElement('canvas');
      canvas.width = 640;
      canvas.height = 480;
      const drawing = canvas.getContext('2d');
      drawing.fillRect(0, 0, canvas.width, canvas.height);
      const stream = canvas.captureStream(30);
      const track = stream.getVideoTracks()[0];
      const draw = () => {
        if (track.readyState === 'ended') return;
        const frame = ++window.__practiceTest.frames;
        drawing.fillStyle = frame % 2 ? '#f0edf7' : '#ded8ed';
        drawing.fillRect(0, 0, canvas.width, canvas.height);
        requestAnimationFrame(draw);
      };
      draw();
      window.__practiceTest.streams.push(stream);
      return stream;
    };
  });

  const page = await openPractice(context);
  assert.equal(await page.locator('#practice-focus').inputValue(), 'mixed');
  await expectFocusControl(page);
  await begin(page, 'ru');
  assert.match(await page.locator('#practice-focus-status').textContent(), /R\s*\/\s*U/);
  await page.screenshot({
    path: 'artifacts/practice-focus-desktop.png',
    fullPage: true,
    animations: 'disabled',
  });
  const ru = [await prompt(page, 1, 6, ['R', 'U'])];

  // The fixture is a real, accepted L. An incorrect attempt must keep the current
  // focused target until the learner explicitly moves to the next prompt.
  await page.waitForFunction(
    () =>
      document.getElementById('prediction-letter').textContent === 'L' &&
      document.getElementById('prediction-hint').textContent.includes('Letter held'),
  );
  await page.locator('#practice-check').click();
  await page.waitForFunction(() => !document.getElementById('practice-next').hidden);
  assert.equal((await page.locator('#practice-letter').textContent()).trim(), ru[0]);
  assert.match(await page.locator('#practice-feedback').textContent(), /model read L/i);
  await page.locator('#practice-next').click();
  ru.push(await prompt(page, 2, 6, ['R', 'U']));

  // Camera interruptions preserve the focus, target and session progress.
  const beforePause = await page.locator('#practice-step').textContent();
  await page.locator('#camera-stop').click();
  assert.equal(await page.locator('#practice-focus').isDisabled(), true);
  assert.equal(await page.locator('#practice-check').isDisabled(), true);
  await cameraOn(page);
  assert.match(await page.locator('#practice-feedback').textContent(), /Hold your letter steady/);
  assert.equal(await page.locator('#practice-step').textContent(), beforePause);
  assert.equal((await page.locator('#practice-letter').textContent()).trim(), ru[1]);
  assert.equal(await page.locator('#practice-focus').inputValue(), 'ru');
  for (let number = 2; number <= 6; number++) {
    if (number > 2) ru.push(await prompt(page, number, 6, ['R', 'U']));
    await skipAndNext(page);
  }
  expectBalanced(ru, ['R', 'U']);
  await expectFinished(page, 6);
  assert.match(await page.locator('#summary-focus').textContent(), /R\s*\/\s*U/);

  await page.setViewportSize({ width: 390, height: 844 });
  await expectFocusControl(page);
  await begin(page, 'mn');
  assert.match(await page.locator('#practice-focus-status').textContent(), /M\s*\/\s*N/);
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({
    path: 'artifacts/practice-focus-mobile.png',
    fullPage: true,
    animations: 'disabled',
  });
  const mn = [];
  for (let number = 1; number <= 6; number++) {
    mn.push(await prompt(page, number, 6, ['M', 'N']));
    await skipAndNext(page);
  }
  expectBalanced(mn, ['M', 'N']);
  await expectFinished(page, 6);
  assert.match(await page.locator('#summary-focus').textContent(), /M\s*\/\s*N/);

  await page.locator('#practice-again').click();
  await page.waitForFunction(() => !document.getElementById('practice-check').hidden);
  assert.equal(await page.locator('#practice-focus').inputValue(), 'mn');
  await prompt(page, 1, 6, ['M', 'N']);
  await skipAndNext(page);
  await page.locator('#practice-finish').click();
  await expectFinished(page, 1);

  await begin(page, 'mixed');
  for (let number = 1; number <= 5; number++) {
    await prompt(page, number, 5, 'ABCDEFGHIJKLMNOPQRSTUVWXYZ');
    await skipAndNext(page);
  }
  await expectFinished(page, 5);

  const linked = await openPractice(context, '/practice?letter=R');
  await linked.setViewportSize({ width: 390, height: 844 });
  assert.equal(await linked.locator('#practice-focus').inputValue(), 'letter');
  await expectFocusControl(linked);
  await begin(linked, 'letter');
  for (let number = 1; number <= 5; number++) {
    await prompt(linked, number, 5, ['R']);
    await skipAndNext(linked);
  }
  await expectFinished(linked, 5);

  const repeated = await openPractice(context, '/practice?letter=L');
  await begin(repeated, 'letter');
  await prompt(repeated, 1, 5, ['L']);
  await repeated.waitForFunction(
    () =>
      document.getElementById('prediction-letter').textContent === 'L' &&
      document.getElementById('prediction-hint').textContent.includes('Letter held'),
  );
  await repeated.locator('#practice-check').click();
  await repeated.waitForFunction(() => !document.getElementById('practice-next').hidden);
  assert.match(await repeated.locator('#practice-feedback').textContent(), /model accepted L/i);
  const nextPrompt = await repeated.evaluate(() => {
    document.getElementById('practice-next').click();
    // Both clicks occur within one browser task, before any fresh camera result.
    document.getElementById('practice-check').click();
    return {
      step: document.getElementById('practice-step').textContent,
      nextHidden: document.getElementById('practice-next').hidden,
      checkHidden: document.getElementById('practice-check').hidden,
    };
  });
  assert.match(nextPrompt.step, /LETTER 2 OF 5/);
  assert.equal(nextPrompt.nextHidden, true, 'a previous hold cannot check the next prompt');
  assert.equal(nextPrompt.checkHidden, false, 'the next prompt still needs a fresh hold');
  await repeated.locator('#practice-finish').click();
  await expectFinished(repeated, 1);
  assert.equal(await repeated.locator('#summary-correct').textContent(), '1');
  assert.deepEqual(errors, [], 'browser JavaScript errors');
  await context.close();
  console.log(
    'Focused practice browser checks passed: balanced R/U and M/N sessions, locked focus through incorrect attempts and camera interruptions, repeat focus, single-letter links, mixed sessions, and mobile/desktop controls. Only generated canvas camera frames were used.',
  );
} catch (error) {
  if (currentPage && !currentPage.isClosed()) {
    console.error(
      'Practice browser failure:',
      await currentPage.evaluate(() => ({
        step: document.getElementById('practice-step')?.textContent,
        letter: document.getElementById('practice-letter')?.textContent,
        feedback: document.getElementById('practice-feedback')?.textContent,
        focus: document.getElementById('practice-focus')?.value,
        camera: document.getElementById('camera-status')?.textContent,
        cameraError: document.getElementById('camera-error')?.textContent,
        prediction: document.getElementById('prediction-hint')?.textContent,
        tracks: window.__practiceTest?.streams.flatMap((stream) =>
          stream.getTracks().map((track) => track.readyState),
        ),
      })),
    );
    await fs.mkdir('artifacts', { recursive: true });
    await currentPage.screenshot({ path: 'artifacts/practice-focus-failure.png', fullPage: true });
  }
  throw error;
} finally {
  await browser.close();
}
