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
const viewports = [
  { width: 320, height: 568 },
  { width: 390, height: 844 },
  { width: 844, height: 390 },
  { width: 1024, height: 768 },
  { width: 1366, height: 768 },
  { width: 1440, height: 900 },
];
const errors = [];
await fs.mkdir('artifacts', { recursive: true });

async function expectFixedPage(page, label) {
  const size = await page.evaluate(() => ({
    height: innerHeight,
    width: innerWidth,
    y: scrollY,
    documentHeight: document.documentElement.scrollHeight,
    documentWidth: document.documentElement.scrollWidth,
    bodyHeight: document.body.getBoundingClientRect().height,
  }));
  assert.ok(
    size.documentHeight <= size.height + 1,
    `${label}: document height ${JSON.stringify(size)}`,
  );
  assert.ok(
    size.documentWidth <= size.width + 1,
    `${label}: document width ${JSON.stringify(size)}`,
  );
  assert.ok(size.bodyHeight <= size.height + 1, `${label}: body must fit the viewport`);
  assert.equal(size.y, 0, `${label}: the document must not scroll`);
}

async function expectReachable(page, selector, label) {
  const control = page.locator(selector);
  await control.scrollIntoViewIfNeeded();
  const result = await control.evaluate((element) => {
    const rect = element.getBoundingClientRect();
    let top = 0,
      left = 0,
      right = innerWidth,
      bottom = innerHeight;
    for (let ancestor = element.parentElement; ancestor; ancestor = ancestor.parentElement) {
      const style = getComputedStyle(ancestor),
        box = ancestor.getBoundingClientRect();
      if (/hidden|clip|auto|scroll/.test(style.overflowX)) {
        left = Math.max(left, box.left);
        right = Math.min(right, box.right);
      }
      if (/hidden|clip|auto|scroll/.test(style.overflowY)) {
        top = Math.max(top, box.top);
        bottom = Math.min(bottom, box.bottom);
      }
    }
    const x = rect.left + rect.width / 2,
      y = rect.top + rect.height / 2;
    const hit = document.elementFromPoint(x, y);
    return {
      rect: {
        top: rect.top,
        left: rect.left,
        right: rect.right,
        bottom: rect.bottom,
        width: rect.width,
        height: rect.height,
      },
      clip: { top, left, right, bottom },
      hit: hit === element || element.contains(hit),
    };
  });
  assert.ok(
    result.rect.width >= 43.5 && result.rect.height >= 43.5,
    `${label}: ${selector} is too small ${JSON.stringify(result)}`,
  );
  assert.ok(
    result.rect.top >= result.clip.top - 1 &&
      result.rect.bottom <= result.clip.bottom + 1 &&
      result.rect.left >= result.clip.left - 1 &&
      result.rect.right <= result.clip.right + 1,
    `${label}: ${selector} is clipped ${JSON.stringify(result)}`,
  );
  assert.equal(
    result.hit,
    true,
    `${label}: ${selector} is covered or unreachable ${JSON.stringify(result)}`,
  );
  await expectFixedPage(page, `${label} after reaching ${selector}`);
}

async function expectCameraNotClipped(page, label) {
  const card = await page.locator('.camera-card').evaluate((element) => ({
    height: element.clientHeight,
    content: element.scrollHeight,
    scrollTop: element.scrollTop,
  }));
  assert.ok(
    card.content <= card.height + 1,
    `${label}: camera content is clipped inside its card ${JSON.stringify(card)}`,
  );
  assert.equal(
    card.scrollTop,
    0,
    `${label}: controls must not depend on programmatically scrolling a hidden overflow`,
  );
}

try {
  for (const theme of ['light', 'dark']) {
    const context = await browser.newContext();
    await context.addInitScript(
      (theme) => localStorage.setItem('signspeak:settings:guest', JSON.stringify({ theme })),
      theme,
    );
    const page = await context.newPage();
    page.on('pageerror', (error) => errors.push(error.message));
    for (const viewport of viewports) {
      const label = `${theme} ${viewport.width}x${viewport.height}`;
      await page.setViewportSize(viewport);
      await page.goto(base + '/translate', { waitUntil: 'networkidle' });
      const cameraTab = page.locator('.workspace-tabs [data-workspace-view="camera"]');
      const textTab = page.locator('.workspace-tabs [data-workspace-view="text"]');
      await page.waitForFunction(() =>
        document.querySelector('.workspace-tabs [data-workspace-view="camera"]'),
      );
      assert.equal(await page.getAttribute('html', 'data-theme'), theme);
      await expectFixedPage(page, label);
      const compact = viewport.width <= 980;
      if (compact) {
        assert.equal(await cameraTab.getAttribute('role'), 'tab');
        assert.equal(await cameraTab.getAttribute('aria-selected'), 'true');
        assert.equal(await cameraTab.getAttribute('tabindex'), '0');
        assert.equal(await textTab.getAttribute('tabindex'), '-1');
        assert.equal(await page.locator('#camera-workspace').isVisible(), true);
        assert.equal(await page.locator('#text-workspace').isVisible(), false);
      } else {
        assert.equal(await page.locator('#camera-workspace').isVisible(), true);
        assert.equal(await page.locator('#text-workspace').isVisible(), true);
      }

      for (const selector of ['#camera-start', '#record-motion', '#camera-stop'])
        await expectReachable(page, selector, label);
      await expectCameraNotClipped(page, label);
      await page.locator('#camera-error').evaluate((element) => {
        element.textContent =
          'Camera permission was denied. Allow camera access in your browser settings, then select Enable camera again.';
        element.hidden = false;
      });
      for (const selector of ['#camera-start', '#record-motion', '#camera-stop'])
        await expectReachable(page, selector, label + ' with camera error');
      await expectCameraNotClipped(page, label + ' with camera error');
      await page.locator('#camera-error').evaluate((element) => {
        element.hidden = true;
        element.textContent = '';
      });

      if (compact) {
        await cameraTab.focus();
        await page.keyboard.press('ArrowRight');
        assert.equal(await textTab.getAttribute('aria-selected'), 'true');
        assert.equal(await textTab.getAttribute('tabindex'), '0');
        assert.equal(await cameraTab.getAttribute('tabindex'), '-1');
        assert.equal(await textTab.evaluate((element) => element === document.activeElement), true);
        assert.equal(await page.locator('#text-workspace').isVisible(), true);
        assert.equal(await page.locator('#camera-workspace').isVisible(), false);
        await page.keyboard.press('Home');
        assert.equal(await cameraTab.getAttribute('aria-selected'), 'true');
        await page.keyboard.press('End');
        assert.equal(await textTab.getAttribute('aria-selected'), 'true');
      }
      await page.locator('#translation-text').fill('A message ready to share.');
      for (const selector of ['#copy-text', '#speak-text', '#save-text'])
        await expectReachable(page, selector, label);
      await page.mouse.move(viewport.width - 25, viewport.height - 35);
      await page.mouse.wheel(0, 1200);
      await page.waitForTimeout(80);
      await expectFixedPage(page, label + ' after wheel');

      if (compact) {
        await page.setViewportSize({ width: 1440, height: 900 });
        await page.waitForFunction(
          () =>
            !document.getElementById('camera-workspace').hidden &&
            !document.getElementById('text-workspace').hidden,
        );
        assert.equal(await page.locator('#camera-workspace').isVisible(), true);
        assert.equal(await page.locator('#text-workspace').isVisible(), true);
        assert.equal(
          await page.locator('#translation-text').inputValue(),
          'A message ready to share.',
        );
        await expectFixedPage(page, label + ' resized wide');
        await page.setViewportSize(viewport);
        await cameraTab.click();
      }

      const helpOpen = page.locator('#translator-help-open'),
        help = page.locator('#translator-help');
      await helpOpen.click();
      assert.equal(await help.evaluate((element) => element.open), true);
      const bounds = await help.boundingBox();
      assert.ok(
        bounds &&
          bounds.x >= -1 &&
          bounds.y >= -1 &&
          bounds.x + bounds.width <= viewport.width + 1 &&
          bounds.y + bounds.height <= viewport.height + 1,
        `${label}: help dialog exceeds the viewport ${JSON.stringify(bounds)}`,
      );
      await page.keyboard.press('Escape');
      assert.equal(await help.evaluate((element) => element.open), false);
      assert.equal(await helpOpen.evaluate((element) => element === document.activeElement), true);
      await helpOpen.click();
      await page.locator('#translator-help-close').click();
      assert.equal(await help.evaluate((element) => element.open), false);
      await expectFixedPage(page, label + ' after help');

      if (theme === 'light' && viewport.width === 1366)
        await page.screenshot({
          path: 'artifacts/translator-viewport-desktop.png',
          fullPage: true,
        });
      if (theme === 'light' && viewport.width === 390)
        await page.screenshot({ path: 'artifacts/translator-viewport-mobile.png', fullPage: true });
      console.log(`Translator viewport checks passed: ${label}.`);
    }
    await context.close();
  }
  const editing = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await editing.goto(base + '/translate', { waitUntil: 'networkidle' });
  await editing.locator('#translation-text').fill('Keep my editing workspace.');
  await editing.setViewportSize({ width: 390, height: 844 });
  await editing.waitForFunction(() => !document.getElementById('text-workspace').hidden);
  assert.equal(
    await editing.locator('[data-workspace-view="text"]').getAttribute('aria-selected'),
    'true',
  );
  assert.equal(
    await editing.locator('#translation-text').inputValue(),
    'Keep my editing workspace.',
  );
  await expectFixedPage(editing, 'focused editor resized to phone');
  const other = await browser.newPage({ viewport: { width: 390, height: 568 } });
  other.on('pageerror', (error) => errors.push(error.message));
  await other.goto(base + '/dashboard', { waitUntil: 'networkidle' });
  assert.ok(
    await other.evaluate(() => document.documentElement.scrollHeight > innerHeight),
    'the viewport lock must be confined to the translator',
  );
  await other.mouse.wheel(0, 700);
  await other.waitForFunction(() => scrollY > 0);
  assert.deepEqual(errors, []);
  console.log(
    'Translator page stays fixed while camera/text controls, compact tabs and Help remain reachable. Other pages retain normal scrolling. No real camera was activated.',
  );
} finally {
  await browser.close();
}
