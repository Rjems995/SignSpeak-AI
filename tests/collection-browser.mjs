// Isolated test fixtures exercise import/export and consent behavior; no new
// human samples are collected, and no real webcam is accessed by this test.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import { SampleCollection, CONSENT_VERSION } from '../signspeak/static/js/collection.js';
const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:5011';
const refs = JSON.parse(await fs.readFile('signspeak/static/models/references.json'));
const fixture = new SampleCollection();
const context = {
  participant: 'test_fixture_01',
  session: 'test_visit_01',
  letter: 'Q',
  hand: 'right',
  lighting: 'soft',
  angle: 'front',
  label_review: 'collector_confirmed',
  consent_version: CONSENT_VERSION,
};
fixture.capture(context, { points: refs.Q });
fixture.capture(
  { ...context, participant: 'test_fixture_02', letter: 'L', hand: 'left' },
  { points: refs.L },
);
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
  args: ['--use-fake-device-for-media-stream', '--use-fake-ui-for-media-stream'],
});
await fs.mkdir('artifacts', { recursive: true });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1100 } }),
    errors = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto(base + '/capture', { waitUntil: 'networkidle' });
  assert.equal(await page.locator('#coverage-rows tr').count(), 26);
  assert.equal(await page.locator('#capture-total').textContent(), '0');
  assert.ok((await page.locator('.source-coverage').textContent()).includes('4770 usable unique'));
  assert.ok((await page.locator('.source-coverage').textContent()).includes('5746 usable unique'));
  assert.equal(await page.locator('.source-audit').count(), 4);
  assert.ok((await page.locator('.source-coverage').textContent()).includes('hand unknown: 5746'));
  assert.ok(
    (await page.locator('.source-coverage').textContent()).includes(
      'Public annotation gaps remain',
    ),
  );
  assert.ok((await page.locator('.source-coverage').textContent()).includes('Lighting benchmark'));
  assert.equal(await page.locator('#capture-letter').inputValue(), 'Q');
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  await page.screenshot({
    path: 'artifacts/collection-desktop.png',
    fullPage: true,
    animations: 'disabled',
  });
  await page.fill('#participant', 'test_fixture_01');
  await page.selectOption('#capture-hand', 'right');
  await page.selectOption('#capture-lighting', 'soft');
  await page.selectOption('#capture-angle', 'front');
  await page.check('#capture-verified');
  await page.check('#capture-consent');
  await page.click('#camera-start');
  await page.waitForFunction(
    () => document.getElementById('camera-status').textContent.includes('Camera on'),
    undefined,
    { timeout: 90000 },
  );
  await page.click('#capture-sample');
  await page.waitForFunction(() =>
    document.getElementById('capture-status').textContent.includes('clearly visible hand'),
  );
  assert.equal(await page.locator('#capture-total').textContent(), '0');
  const upload = {
    name: 'automated-test-fixtures.json',
    mimeType: 'application/json',
    buffer: Buffer.from(JSON.stringify(fixture.export('static'))),
  };
  await page.locator('#import-samples').setInputFiles(upload);
  await page.waitForFunction(() => document.getElementById('capture-total').textContent === '2');
  assert.equal(await page.locator('#capture-people').textContent(), '2');
  assert.equal(await page.locator('#capture-hands').textContent(), '1 / 1');
  assert.equal(await page.locator('#capture-q').textContent(), '1');
  await page.locator('#import-samples').setInputFiles(upload);
  await page.waitForFunction(() =>
    document.getElementById('capture-status').textContent.includes('skipped 2 duplicates'),
  );
  assert.equal(await page.locator('#capture-total').textContent(), '2');
  const downloadPromise = page.waitForEvent('download');
  await page.click('#download-static');
  const download = await downloadPromise;
  const exported = JSON.parse(await fs.readFile(await download.path(), 'utf8'));
  assert.equal(exported.version, 3);
  assert.equal(exported.samples.length, 2);
  assert.equal(exported.samples[0].participant, 'test_fixture_01');
  assert.equal(exported.samples[0].lighting, 'soft');
  await page.click('#capture-next');
  assert.equal(await page.locator('#capture-letter').inputValue(), 'E');
  assert.equal(await page.locator('#capture-verified').isChecked(), false);
  await page.click('#capture-undo');
  assert.equal(await page.locator('#capture-total').textContent(), '1');
  await page.uncheck('#capture-consent');
  assert.equal(await page.locator('#capture-total').textContent(), '0');
  assert.ok((await page.locator('#camera-status').textContent()).includes('Camera off'));
  assert.equal(await page.locator('#download-static').isDisabled(), true);
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({
    path: 'artifacts/collection-mobile.png',
    fullPage: true,
    animations: 'disabled',
  });
  assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth));
  assert.deepEqual(errors, []);
  console.log(
    'Collection browser checks passed: real source counts, empty-camera rejection, metadata-preserving import/export, deduplication, priorities, withdrawal, desktop/mobile layout.',
  );
} finally {
  await browser.close();
}
