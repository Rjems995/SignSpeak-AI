import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:5000';
const executable =
  process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const browser = await chromium.launch({
  headless: true,
  executablePath: executable,
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
});
const errors = [];
await fs.mkdir('artifacts', { recursive: true });
try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1100 } });
  const page = await context.newPage();
  page.on('pageerror', (e) => errors.push(e.stack || e.message));
  await page.goto(base, { waitUntil: 'networkidle' });
  assert.equal(await page.title(), 'A little gesture. A new connection. · SignSpeak AI');
  await page.screenshot({ path: 'artifacts/desktop.png', fullPage: true });
  for (const route of [
    '/dashboard',
    '/translate',
    '/learn',
    '/practice',
    '/settings',
    '/capture',
  ]) {
    const response = await page.goto(base + route, { waitUntil: 'networkidle' });
    assert.equal(response.status(), 200, route);
    assert.ok(
      await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      `overflow on desktop ${route}`,
    );
  }
  await page.goto(base + '/translate');
  await page.locator('#translation-text').fill('A little hello');
  await page.locator('#add-space').click();
  assert.equal(await page.locator('#translation-text').inputValue(), 'A little hello ');
  await page.locator('#delete-letter').click();
  assert.equal(await page.locator('#translation-text').inputValue(), 'A little hello');
  await page.locator('#clear-text').click();
  await page.locator('#confirm-dialog [value=confirm]').click();
  await page.waitForFunction(() => document.getElementById('translation-text').value === '');
  await page.locator('#camera-start').click();
  await page.waitForFunction(
    () => document.getElementById('camera-status').textContent.includes('Camera on'),
    undefined,
    { timeout: 90000 },
  );
  assert.equal(await page.locator('#camera-empty').isVisible(), false);
  assert.equal(await page.locator('#record-motion').isEnabled(), true);
  await page.locator('#record-motion').click();
  await page.waitForFunction(() =>
    document.getElementById('toast').textContent.includes('clearly visible hand'),
  );
  // Fake browser video has no ASL hand; it must not produce invented letters.
  await page.waitForTimeout(1200);
  assert.equal(await page.locator('#translation-text').inputValue(), '');
  await page.locator('#camera-stop').click();
  assert.ok((await page.locator('#camera-status').textContent()).includes('Camera off'));
  // Exercise the real detector and classifier on Google's public hand sample.
  // This is an integration check, not an independent ASL recognition benchmark.
  const source = await fetch(
    'https://storage.googleapis.com/mediapipe-tasks/hand_landmarker/woman_hands.jpg',
  );
  assert.equal(source.status, 200);
  const bytes = Buffer.from(await source.arrayBuffer());
  await page.route('**/__test-hand.jpg', (route) =>
    route.fulfill({ body: bytes, contentType: 'image/jpeg' }),
  );
  const realHand = await page.evaluate(async () => {
    const { FilesetResolver, HandLandmarker } =
      await import('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/vision_bundle.mjs');
    const files = await FilesetResolver.forVisionTasks(
      'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm',
    );
    const detector = await HandLandmarker.createFromOptions(files, {
      baseOptions: {
        modelAssetPath:
          'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      },
      runningMode: 'IMAGE',
      numHands: 1,
    });
    const img = new Image();
    img.src = '/__test-hand.jpg';
    await img.decode();
    const output = detector.detect(img);
    detector.close();
    if (!output.landmarks.length) return { hands: 0 };
    const { LandmarkClassifier } = await import('/static/js/recognition.js');
    const data = await fetch('/static/models/classifier.json').then((r) => r.json());
    const classifier = new LandmarkClassifier(data, window.tf);
    const prediction = classifier.predict(output.landmarks[0], 0.85);
    return { hands: output.landmarks.length, landmarks: output.landmarks[0].length, ...prediction };
  });
  assert.equal(realHand.hands, 1);
  assert.equal(realHand.landmarks, 21);
  assert.ok(Number.isFinite(realHand.confidence));
  console.log('Recorded-hand integration result:', JSON.stringify(realHand));
  await page.screenshot({ path: 'artifacts/translator.png', fullPage: true });
  await page.goto(base + '/settings');
  await page.selectOption('#theme', 'dark');
  await page.getByRole('button', { name: 'Save preferences' }).click();
  await page.waitForFunction(() => document.documentElement.dataset.theme === 'dark');
  await page.reload();
  assert.equal(await page.getAttribute('html', 'data-theme'), 'dark');
  await page.goto(base + '/learn', { waitUntil: 'networkidle' });
  assert.equal(await page.locator('.letter-card').count(), 26);
  assert.equal(await page.getAttribute('#learn-progress', 'max'), '26');
  await page.locator('.mark-learned').first().click();
  await page.waitForFunction(() => document.getElementById('learn-count').textContent === '1 / 26');
  // Complete registration, private saving/search/deletion, and account cleanup.
  const email = `browser-${Date.now()}@example.test`,
    password = 'a-long-browser-test-password';
  await page.goto(base + '/register');
  await page.fill('#name', 'Browser Test');
  await page.fill('#email', email);
  await page.fill('#password', password);
  await page.getByRole('button', { name: 'Create account', exact: true }).click();
  await page.waitForURL('**/dashboard');
  await page.goto(base + '/translate');
  await page.fill('#translation-text', 'ABC');
  await page.click('#save-text');
  await page.waitForFunction(
    () => document.getElementById('toast').textContent === 'Saved to your private history.',
  );
  await page.goto(base + '/history');
  await page.getByText('ABC', { exact: true }).waitFor();
  await page.click('[aria-label="Delete translation"]');
  await page.locator('#confirm-dialog [value=confirm]').click();
  await page.getByText('Your next conversation belongs here.').waitFor();
  await page.goto(base + '/profile');
  await page.fill('#delete-password', password);
  await page.getByRole('button', { name: 'Delete my account', exact: true }).click();
  await page.locator('#confirm-dialog [value=confirm]').click();
  await page.waitForURL(base + '/');
  const mobile = await browser.newContext({
    viewport: { width: 390, height: 844 },
    isMobile: true,
    hasTouch: true,
  });
  const mobilePage = await mobile.newPage();
  mobilePage.on('pageerror', (e) => errors.push(e.message));
  for (const route of [
    '/',
    '/dashboard',
    '/translate',
    '/learn',
    '/practice',
    '/settings',
    '/capture',
    '/login',
  ]) {
    await mobilePage.goto(base + route, { waitUntil: 'networkidle' });
    assert.ok(
      await mobilePage.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      `overflow on mobile ${route}`,
    );
  }
  await mobilePage.goto(base);
  await mobilePage.screenshot({ path: 'artifacts/mobile.png', fullPage: true });
  await mobilePage.click('#menu-toggle');
  assert.equal(await mobilePage.getAttribute('#menu-toggle', 'aria-expanded'), 'true');
  await mobilePage.getByRole('link', { name: 'Live translator', exact: true }).click();
  await mobilePage.waitForURL('**/translate');
  assert.deepEqual(errors, [], 'Browser JavaScript errors');
  console.log(
    'Browser checks passed: desktop/mobile pages, real MediaPipe loading with fake camera, text editor, settings, learning, account/history lifecycle.',
  );
} finally {
  await browser.close();
}
