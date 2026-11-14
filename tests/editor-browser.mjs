// The browser camera is Chrome's fake device. A fixture inference worker supplies
// public reference coordinates to the real classifier and translator callbacks.
// Editor events below exercise the application listeners, without test-only app APIs.
import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';

const base = process.env.TEST_BASE_URL || 'http://127.0.0.1:5000';
const references = JSON.parse(
  await fs.readFile(new URL('../signspeak/static/models/references.json', import.meta.url), 'utf8'),
);
const points = Array.from({ length: 21 }, (_, index) => ({
  x: references.L[index * 2],
  y: references.L[index * 2 + 1],
  z: 0,
}));
const workerSource = `
  let letter = 'L';
  const points = ${JSON.stringify(points)};
  onmessage = ({ data }) => {
    if (data.type === 'fixture') {
      letter = data.letter;
      postMessage({ type: 'fixture-ready', serial: data.serial });
      return;
    }
    if (data.type === 'init') {
      postMessage({ type: 'ready', id: data.id });
      return;
    }
    if (data.type === 'detect') {
      data.frame.close();
      postMessage({ type: 'result', id: data.id, result: {
        landmarks: letter ? [points] : [],
        worldLandmarks: [],
        handednesses: letter ? [[{ categoryName: 'Right', score: 1 }]] : [],
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
const textSelector = '#translation-text';
const textTab = '.workspace-tabs [data-workspace-view="text"]';
const cameraTab = '.workspace-tabs [data-workspace-view="camera"]';

async function openTranslator(context) {
  const page = await context.newPage();
  page.on('pageerror', (error) => errors.push(error.message));
  await page.route('https://fonts.googleapis.com/**', (route) => route.abort());
  await page.route('**/static/js/hand-tracking-worker.js', (route) =>
    route.fulfill({ body: workerSource, contentType: 'text/javascript' }),
  );
  await page.goto(base + '/translate', { waitUntil: 'networkidle' });
  await page.waitForFunction(() => document.getElementById('undo-text'));
  await page.waitForFunction(() => document.getElementById('character-count').textContent);
  return page;
}

async function valueIs(page, value, label = 'editor value') {
  await page.waitForFunction(
    (value) => document.getElementById('translation-text').value === value,
    value,
  );
  assert.equal(await page.locator(textSelector).inputValue(), value, label);
  assert.equal(
    (await page.locator('#character-count').textContent()).trim(),
    `${value.length} / 5,000 characters`,
    `${label}: character count`,
  );
}

async function expectPaused(page, paused = true) {
  assert.equal(await page.locator('#resume-insertion').isVisible(), paused);
  if (paused) assert.equal(await page.locator('#editor-paused').isVisible(), true);
  else if (await page.locator('#editor-paused').isVisible())
    assert.equal(
      (await page.locator('#editor-pause-message').textContent()).trim(),
      'Release your hand to continue',
      'resumed insertion may keep the release instruction visible until a neutral observation',
    );
}

async function selection(page, start, end = start, direction = 'none') {
  await page
    .locator(textSelector)
    .evaluate((text, { start, end, direction }) => text.setSelectionRange(start, end, direction), {
      start,
      end,
      direction,
    });
}

async function snapshot(page) {
  return page.locator(textSelector).evaluate((text) => ({
    value: text.value,
    start: text.selectionStart,
    end: text.selectionEnd,
    direction: text.selectionDirection,
  }));
}

async function expectSelection(page, start, end = start, direction) {
  const current = await snapshot(page);
  assert.equal(current.start, start, 'restore selection start');
  assert.equal(current.end, end, 'restore selection end');
  if (direction) assert.equal(current.direction, direction, 'restore selection direction');
}

// Synthetic beforeinput/input pairs let the same UI test cover mobile input
// histories and IME composition deterministically on a desktop Chrome runner.
async function edit(
  page,
  value,
  { inputType = 'insertText', data = value, caret = value.length } = {},
) {
  await page.locator(textSelector).focus();
  await page.locator(textSelector).evaluate(
    (text, { value, inputType, data, caret }) => {
      const before = new InputEvent('beforeinput', {
        bubbles: true,
        cancelable: true,
        inputType,
        data,
      });
      if (!text.dispatchEvent(before)) throw new Error('Ordinary editor input was cancelled.');
      text.value = value;
      text.setSelectionRange(caret, caret);
      text.dispatchEvent(new InputEvent('input', { bubbles: true, inputType, data }));
    },
    { value, inputType, data, caret },
  );
  await valueIs(page, value);
}

async function fixture(page, letter) {
  const serial = await page.evaluate((letter) => {
    const state = window.__editorTest;
    const serial = ++state.serial;
    state.worker.postMessage({ type: 'fixture', letter, serial });
    return serial;
  }, letter);
  await page.waitForFunction((serial) => window.__editorTest.ack === serial, serial);
}

async function cameraState(page) {
  return page.evaluate(() => ({
    running: document.getElementById('camera-viewport').classList.contains('running'),
    status: document.getElementById('camera-status').textContent.trim(),
    workers: window.__editorTest.workers,
    terminated: window.__editorTest.terminated,
    stops: window.__editorTest.stops,
    tracks: window.__editorTest.streams.flatMap((stream) =>
      stream.getTracks().map((track) => track.readyState),
    ),
  }));
}

async function confirmClear(page, confirm) {
  await page.locator('#clear-text').click();
  assert.equal(await page.locator('#confirm-dialog').isVisible(), true);
  await page.locator(`#confirm-dialog button[value="${confirm ? 'confirm' : 'cancel'}"]`).click();
  await page.waitForFunction(() => !document.getElementById('confirm-dialog').open);
}

async function expectMobileControls(page, label) {
  for (const selector of ['#resume-insertion', '#undo-text']) {
    await page.locator(selector).scrollIntoViewIfNeeded();
    const geometry = await page.locator(selector).evaluate((button) => {
      const box = button.getBoundingClientRect();
      const hit = document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2);
      return {
        x: box.x,
        y: box.y,
        right: box.right,
        bottom: box.bottom,
        width: box.width,
        height: box.height,
        reachable: hit === button || button.contains(hit),
        viewportWidth: innerWidth,
        viewportHeight: innerHeight,
        documentWidth: document.documentElement.scrollWidth,
        documentHeight: document.documentElement.scrollHeight,
        scrollY,
      };
    });
    assert.ok(
      geometry.x >= -1 &&
        geometry.y >= -1 &&
        geometry.right <= geometry.viewportWidth + 1 &&
        geometry.bottom <= geometry.viewportHeight + 1 &&
        geometry.reachable,
      `${label}: ${selector} must be reachable: ${JSON.stringify(geometry)}`,
    );
    assert.ok(
      geometry.width >= 43.5 && geometry.height >= 43.5,
      `${label}: ${selector} needs a 44px touch target: ${JSON.stringify(geometry)}`,
    );
    assert.ok(
      geometry.documentWidth <= geometry.viewportWidth + 1 &&
        geometry.documentHeight <= geometry.viewportHeight + 1 &&
        geometry.scrollY === 0,
      `${label}: keep document fixed while its text panel scrolls: ${JSON.stringify(geometry)}`,
    );
  }
}

try {
  const context = await browser.newContext({ viewport: { width: 1440, height: 1000 } });
  await context.addInitScript(() => {
    localStorage.setItem(
      'signspeak:settings:guest',
      JSON.stringify({ hold_ms: 300, threshold: 0.85, landmarks: false }),
    );
    const state = (window.__editorTest = {
      streams: [],
      workers: 0,
      terminated: 0,
      serial: 0,
      ack: 0,
      stops: [],
    });
    const nativeStop = MediaStreamTrack.prototype.stop;
    MediaStreamTrack.prototype.stop = function () {
      state.stops.push(new Error('track stop').stack);
      return nativeStop.call(this);
    };
    const NativeWorker = window.Worker;
    window.Worker = class extends NativeWorker {
      constructor(...args) {
        super(...args);
        if (!String(args[0]).includes('hand-tracking-worker.js')) return;
        state.worker = this;
        state.workers++;
        this.addEventListener('message', ({ data }) => {
          if (data.type === 'fixture-ready') state.ack = data.serial;
        });
      }
      terminate() {
        state.terminated++;
        return super.terminate();
      }
    };
    const getUserMedia = navigator.mediaDevices.getUserMedia.bind(navigator.mediaDevices);
    navigator.mediaDevices.getUserMedia = async (constraints) => {
      const stream = await getUserMedia(constraints);
      state.streams.push(stream);
      return stream;
    };
  });

  const cameraPage = await openTranslator(context);
  await expectPaused(cameraPage, false);
  assert.equal(await cameraPage.locator('#undo-text').isDisabled(), true);
  assert.equal(await cameraPage.locator('#redo-text').isDisabled(), true);
  assert.equal(
    await cameraPage.locator('#editor-announcement').getAttribute('aria-live'),
    'polite',
  );
  await cameraPage.locator('#camera-start').click();
  await cameraPage.waitForFunction(
    () => document.getElementById('camera-status').textContent.trim() === 'Camera on',
    undefined,
    { timeout: 120000 },
  );
  await valueIs(cameraPage, 'L', 'real classifier automatically appends through app onCommit');
  await expectPaused(cameraPage, false);
  assert.equal(
    await cameraPage.locator(textSelector).evaluate((text) => text === document.activeElement),
    false,
  );
  const running = await cameraState(cameraPage);
  assert.ok(running.workers >= 1);
  assert.ok(
    running.tracks.length > 0 && running.tracks.every((state) => state === 'live'),
    JSON.stringify(running),
  );

  await cameraPage.locator('#undo-text').click();
  await valueIs(cameraPage, '', 'Undo removes a recognized letter');
  await expectPaused(cameraPage);
  await cameraPage.waitForTimeout(1200);
  await valueIs(cameraPage, '', 'holding a sign must not restore an undone letter while paused');
  assert.deepEqual(
    await cameraState(cameraPage),
    running,
    'Undo leaves camera and inference worker active',
  );
  await cameraPage.locator('#redo-text').click();
  await valueIs(cameraPage, 'L', 'Redo restores recognized text');
  await cameraPage.locator('#undo-text').click();
  await valueIs(cameraPage, '');
  await cameraPage.locator('#resume-insertion').click();
  await expectPaused(cameraPage, false);
  await cameraPage.waitForTimeout(1200);
  await valueIs(cameraPage, '', 'Resume requires a release before accepting the still-held letter');
  await fixture(cameraPage, null);
  await cameraPage.waitForFunction(() =>
    document.getElementById('prediction-label').textContent.includes('Looking for your hand'),
  );
  await cameraPage.waitForTimeout(450);
  await fixture(cameraPage, 'L');
  await valueIs(cameraPage, 'L', 'explicit Resume, release and fresh hold enable recognition');

  await cameraPage.locator('#record-motion').click();
  assert.equal(await cameraPage.locator('#record-motion').textContent(), 'Finish movement');
  await cameraPage.locator(textSelector).focus();
  await expectPaused(cameraPage);
  assert.equal(
    await cameraPage.locator('#record-motion').textContent(),
    'Record J / Z',
    'entering corrections cancels an in-progress J/Z recording',
  );
  assert.match(await cameraPage.locator('#motion-status').textContent(), /cancelled/i);
  await selection(cameraPage, 1);
  await cameraPage.keyboard.insertText(' corrections');
  await valueIs(cameraPage, 'L corrections');
  await cameraPage.locator('#text-heading').click();
  await expectPaused(cameraPage);
  await fixture(cameraPage, null);
  await cameraPage.waitForTimeout(450);
  await fixture(cameraPage, 'L');
  await cameraPage.waitForTimeout(1200);
  await valueIs(cameraPage, 'L corrections', 'blur and fresh signs do not end correction mode');
  assert.deepEqual(await cameraState(cameraPage), running, 'manual correction keeps camera active');
  await cameraPage.locator('#camera-stop').click();
  await cameraPage.close();

  const page = await openTranslator(context);
  await edit(page, 'HELLO');
  await expectPaused(page);
  await selection(page, 1, 4, 'forward');
  await page.keyboard.insertText('i');
  await valueIs(page, 'HiO');
  await expectSelection(page, 2);
  await page.locator('#undo-text').click();
  await valueIs(page, 'HELLO');
  await expectSelection(page, 1, 4, 'forward');
  await page.locator('#redo-text').click();
  await valueIs(page, 'HiO');
  await expectSelection(page, 2);
  await page.locator('#undo-text').click();
  await page.keyboard.insertText('ey');
  await valueIs(page, 'HeyO');
  assert.equal(await page.locator('#redo-text').isDisabled(), true, 'a new edit invalidates Redo');

  await selection(page, 2);
  await page.locator('#add-space').click();
  await valueIs(page, 'He yO');
  await expectSelection(page, 3);
  await page.locator('#undo-text').click();
  await valueIs(page, 'HeyO');
  await expectSelection(page, 2);
  await page.locator('#redo-text').click();
  await valueIs(page, 'He yO');
  await selection(page, 1, 4, 'backward');
  await page.locator('#delete-letter').click();
  await valueIs(page, 'HO');
  await expectSelection(page, 1);
  await page.locator('#undo-text').click();
  await valueIs(page, 'He yO');
  await expectSelection(page, 1, 4, 'backward');
  await page.locator('#redo-text').click();
  await valueIs(page, 'HO');
  await expectSelection(page, 1);
  await edit(page, 'H😀O', { caret: 3 });
  await page.locator('#delete-letter').click();
  await valueIs(page, 'HO', 'Delete removes a full Unicode character');
  await page.locator('#undo-text').click();
  await valueIs(page, 'H😀O');
  await expectSelection(page, 3);

  const beforeCancel = await snapshot(page);
  const beforeButtons = await page
    .locator('#undo-text, #redo-text')
    .evaluateAll((buttons) => buttons.map((button) => button.disabled));
  await confirmClear(page, false);
  assert.deepEqual(
    await snapshot(page),
    beforeCancel,
    'Clear cancellation preserves text and selection',
  );
  assert.deepEqual(
    await page
      .locator('#undo-text, #redo-text')
      .evaluateAll((buttons) => buttons.map((button) => button.disabled)),
    beforeButtons,
    'Clear cancellation preserves undo and redo availability',
  );
  await expectPaused(page);
  await confirmClear(page, true);
  await valueIs(page, '');
  await page.locator('#undo-text').click();
  await valueIs(page, 'H😀O', 'confirmed Clear is reversible');
  await expectSelection(page, 3);
  await page.locator('#redo-text').click();
  await valueIs(page, '');

  await edit(page, 'AB');
  for (const [undo, redo] of [
    ['Control+z', 'Control+Shift+z'],
    ['Control+z', 'Control+y'],
    ['Meta+z', 'Meta+Shift+z'],
  ]) {
    await page.locator(textSelector).press(undo);
    await valueIs(page, '', `${undo} uses shared editor history`);
    await page.locator(textSelector).press(redo);
    await valueIs(page, 'AB', `${redo} restores shared editor history`);
  }
  for (const [inputType, value] of [
    ['historyUndo', ''],
    ['historyRedo', 'AB'],
  ]) {
    const intercepted = await page.locator(textSelector).evaluate((text, inputType) => {
      const event = new InputEvent('beforeinput', { bubbles: true, cancelable: true, inputType });
      text.dispatchEvent(event);
      return event.defaultPrevented;
    }, inputType);
    assert.equal(intercepted, true, `${inputType} must suppress native history`);
    await valueIs(page, value, `mobile ${inputType} uses shared history`);
  }

  await selection(page, 2);
  await page.locator(textSelector).evaluate((text) => {
    text.dispatchEvent(new CompositionEvent('compositionstart', { bubbles: true, data: '' }));
    for (const suffix of ['n', 'に']) {
      text.dispatchEvent(
        new InputEvent('beforeinput', {
          bubbles: true,
          inputType: 'insertCompositionText',
          data: suffix,
          isComposing: true,
        }),
      );
      text.value = `AB${suffix}`;
      text.setSelectionRange(text.value.length, text.value.length);
      text.dispatchEvent(
        new InputEvent('input', {
          bubbles: true,
          inputType: 'insertCompositionText',
          data: suffix,
          isComposing: true,
        }),
      );
    }
  });
  await expectPaused(page);
  assert.equal(
    await page.locator('#resume-insertion').isDisabled(),
    true,
    'IME composition disables Resume',
  );
  await page.locator(textSelector).evaluate((text) => {
    text.dispatchEvent(new CompositionEvent('compositionend', { bubbles: true, data: 'に' }));
    text.dispatchEvent(
      new InputEvent('input', { bubbles: true, inputType: 'insertFromComposition', data: 'に' }),
    );
  });
  await valueIs(page, 'ABに');
  assert.equal(await page.locator('#resume-insertion').isEnabled(), true);
  await page.locator('#undo-text').click();
  await valueIs(page, 'AB', 'composition intermediate values are one undo step');
  await page.locator('#redo-text').click();
  await valueIs(page, 'ABに');

  await edit(page, 'A message that remains available during corrections.\n'.repeat(35));
  for (const viewport of [
    { width: 390, height: 844 },
    { width: 320, height: 568 },
  ]) {
    await page.setViewportSize(viewport);
    await page.locator(textTab).click();
    await expectPaused(page);
    assert.equal(await page.locator('#text-workspace').isVisible(), true);
    assert.equal(await page.locator('#camera-workspace').isVisible(), false);
    await expectMobileControls(page, `${viewport.width}x${viewport.height} message mode`);
    await page.locator('#undo-text').click();
    assert.equal(await page.locator(textSelector).inputValue(), 'ABに');
    const editorVisible = await page.locator(textSelector).evaluate((text) => {
      const box = text.getBoundingClientRect();
      return document.elementFromPoint(box.x + box.width / 2, box.y + box.height / 2) === text;
    });
    assert.equal(
      editorVisible,
      true,
      'Undo must bring the restored editor into view inside its panel',
    );
    await page.locator('#redo-text').click();
    await valueIs(page, 'A message that remains available during corrections.\n'.repeat(35));
    await page.locator(cameraTab).click();
    await expectPaused(page);
    await page.locator('#resume-insertion').click();
    await expectPaused(page, false);
    await page.locator(textTab).click();
    await page.locator(textSelector).focus();
    await expectPaused(page);
  }
  await page.setViewportSize({ width: 390, height: 844 });
  await page.locator(textTab).click();
  await page.locator('#text-workspace').evaluate((panel) => (panel.scrollTop = 0));
  await fs.mkdir('artifacts', { recursive: true });
  await page.screenshot({ path: 'artifacts/editor-mobile.png', fullPage: true });
  await context.close();
  assert.deepEqual(errors, [], 'browser JavaScript errors');
  console.log(
    'Editor browser checks passed: real-classifier additions, persistent correction pause with live fake camera, explicit Resume and fresh release, shared recognition/manual history, selection restoration, Space/Delete/Clear, cancellation, shortcuts, mobile history input, IME composition, and fixed mobile controls. No real camera was activated.',
  );
} finally {
  await browser.close();
}
