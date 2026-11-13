import { chromium } from 'playwright';
import assert from 'node:assert/strict';

const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:5000';
const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH || 'C:/Program Files/Google/Chrome/Application/chrome.exe',
});
const errors = [];
try {
  const page = await browser.newPage({ viewport: { width: 390, height: 844 } });
  page.on('pageerror', (error) => errors.push(error.message));
  // Navigation needs no remote model, font or camera access.
  await page.route('https://fonts.googleapis.com/**', (route) => route.abort());
  await page.goto(base + '/translate', { waitUntil: 'networkidle' });
  const toggle = page.locator('#menu-toggle'),
    sidebar = page.locator('#sidebar');
  const close = page.locator('#menu-close'),
    backdrop = page.locator('#nav-backdrop');
  const shell = page.locator('.app-shell');

  async function expectClosed() {
    assert.equal(await toggle.getAttribute('aria-expanded'), 'false');
    assert.equal(await sidebar.evaluate((element) => element.inert), true);
    assert.equal(await sidebar.getAttribute('aria-hidden'), 'true');
    assert.equal(await shell.evaluate((element) => element.inert), false);
    assert.equal(await backdrop.isVisible(), false);
    assert.equal(
      await page.locator('body').evaluate((element) => element.classList.contains('nav-open')),
      false,
    );
  }

  await expectClosed();
  assert.equal(await page.locator('.breadcrumb>strong').isVisible(), true);
  assert.equal((await page.locator('.breadcrumb>strong').innerText()).trim(), 'Live translator');
  await toggle.focus();
  await page.keyboard.press('Tab');
  assert.equal(
    await sidebar.evaluate((element) => element.contains(document.activeElement)),
    false,
    'closed navigation links must stay out of the tab order',
  );

  await toggle.click();
  assert.equal(await toggle.getAttribute('aria-expanded'), 'true');
  assert.equal(await sidebar.evaluate((element) => element.inert), false);
  assert.equal(await sidebar.getAttribute('role'), 'dialog');
  assert.equal(await sidebar.getAttribute('aria-modal'), 'true');
  assert.equal(await shell.evaluate((element) => element.inert), true);
  assert.equal(await backdrop.isVisible(), true);
  assert.equal(
    await page.locator('body').evaluate((element) => getComputedStyle(element).overflowY),
    'hidden',
  );
  assert.equal(await close.evaluate((element) => element === document.activeElement), true);
  await page.keyboard.press('Shift+Tab');
  assert.equal(
    await sidebar.locator('.user-card').evaluate((element) => element === document.activeElement),
    true,
    'backward tabbing from the first control must wrap inside the drawer',
  );
  await page.keyboard.press('Tab');
  assert.equal(
    await close.evaluate((element) => element === document.activeElement),
    true,
    'forward tabbing from the last control must wrap inside the drawer',
  );
  await page.keyboard.press('Escape');
  await expectClosed();
  assert.equal(await toggle.evaluate((element) => element === document.activeElement), true);

  await toggle.click();
  await close.click();
  await expectClosed();
  assert.equal(await toggle.evaluate((element) => element === document.activeElement), true);

  await toggle.click();
  await page.mouse.click(page.viewportSize().width - 10, 400);
  await expectClosed();
  assert.equal(await toggle.evaluate((element) => element === document.activeElement), true);

  await toggle.click();
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.waitForFunction(() => !document.querySelector('.app-shell').inert);
  assert.equal(await sidebar.evaluate((element) => element.inert), false);
  assert.equal(await sidebar.getAttribute('aria-hidden'), null);
  assert.equal(await sidebar.getAttribute('aria-modal'), null);
  assert.equal(await sidebar.getAttribute('role'), null);
  assert.equal(await backdrop.isVisible(), false);
  assert.equal(await toggle.isVisible(), false);
  assert.equal(
    await page.locator('body').evaluate((element) => element.classList.contains('nav-open')),
    false,
  );
  assert.equal(await sidebar.locator('[aria-current="page"]').isVisible(), true);
  assert.equal(
    await page.evaluate(() => document.activeElement.id === 'menu-close'),
    false,
    'resizing must not leave keyboard focus on a hidden close control',
  );

  await sidebar.locator('[aria-current="page"]').focus();
  await page.setViewportSize({ width: 390, height: 844 });
  await page.waitForFunction(() => document.getElementById('sidebar').inert);
  await expectClosed();
  assert.equal(
    await toggle.evaluate((element) => element === document.activeElement),
    true,
    'focused desktop navigation must transfer focus before becoming unavailable',
  );

  const skip = page.locator('.skip-link');
  await skip.focus();
  await page.keyboard.press('Enter');
  await page.waitForFunction(() => document.activeElement.id === 'main');
  assert.equal(await page.locator('#main').getAttribute('tabindex'), '-1');
  assert.deepEqual(errors, []);
  console.log(
    'Navigation browser checks passed: closed-drawer tab exclusion, modal focus trap, Escape/close/backdrop focus return, responsive reset, mobile page name and skip link.',
  );
} finally {
  await browser.close();
}
