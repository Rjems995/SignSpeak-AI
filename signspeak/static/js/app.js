export const config = JSON.parse(document.getElementById('app-config').textContent);
const storageKey = `signspeak:settings:${config.user_id ?? 'guest'}`;
export function readStorage(key, fallback) {
  try {
    return JSON.parse(localStorage.getItem(key)) ?? fallback;
  } catch {
    return fallback;
  }
}
export function writeStorage(key, value) {
  try {
    localStorage.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}
export const settings = { ...config.settings, ...readStorage(storageKey, {}) };
function applyTheme(theme) {
  const dark = theme === 'dark';
  document.documentElement.dataset.theme = dark ? 'dark' : 'light';
  const toggle = document.getElementById('theme-toggle');
  toggle.setAttribute('aria-label', dark ? 'Switch to light mode' : 'Switch to dark mode');
  toggle.setAttribute('aria-pressed', String(dark));
}
applyTheme(settings.theme);
export function persistSettings() {
  return writeStorage(storageKey, settings);
}
let toastTimer;
export function toast(message) {
  const el = document.getElementById('toast');
  el.textContent = message;
  el.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (el.hidden = true), 5000);
}
export async function api(path, options = {}) {
  const response = await fetch(path, {
    ...options,
    headers: {
      'Content-Type': 'application/json',
      'X-CSRFToken': document.querySelector('meta[name=csrf-token]').content,
      ...options.headers,
    },
  });
  const data = await response.json().catch(() => ({
    error: 'The server could not complete this request. Please refresh and try again.',
  }));
  if (!response.ok) throw new Error(data.error || 'Something went wrong. Please try again.');
  return data;
}
export function makeIcon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.classList.add('icon');
  svg.setAttribute('aria-hidden', 'true');
  const use = document.createElementNS(svg.namespaceURI, 'use');
  use.setAttribute('href', `/static/icons.svg#${name}`);
  svg.append(use);
  return svg;
}
export function confirmAction(title, message, confirmLabel = 'Delete') {
  const dialog = document.getElementById('confirm-dialog');
  document.getElementById('confirm-title').textContent = title;
  document.getElementById('confirm-message').textContent = message;
  dialog.querySelector('[value="confirm"]').textContent = confirmLabel;
  return new Promise((resolve) => {
    dialog.addEventListener('close', () => resolve(dialog.returnValue === 'confirm'), {
      once: true,
    });
    dialog.showModal();
  });
}
export async function copyText(text) {
  if (!text.trim()) return toast('Add a few letters first.');
  try {
    await navigator.clipboard.writeText(text);
    toast('Copied. Your words are ready to share.');
  } catch {
    toast('Clipboard is unavailable. Select your text and use Ctrl+C or Copy.');
  }
}
document.getElementById('theme-toggle').addEventListener('click', () => {
  settings.theme = settings.theme === 'dark' ? 'light' : 'dark';
  applyTheme(settings.theme);
  persistSettings();
  const select = document.getElementById('theme');
  if (select) select.value = settings.theme;
});
import { initNavigation } from './navigation.js';
initNavigation();
document.querySelectorAll('[data-date]').forEach(
  (el) =>
    (el.textContent = new Date(el.dataset.date).toLocaleDateString(undefined, {
      month: 'short',
      day: 'numeric',
    })),
);

export class Speaker {
  constructor() {
    this.synth = window.speechSynthesis;
    this.utterance = null;
  }
  voices() {
    return this.synth?.getVoices().filter((v) => v.lang.toLowerCase().startsWith('en')) ?? [];
  }
  speak(text, opts = settings) {
    if (!this.synth) {
      toast('Read aloud is unavailable in this browser. You can copy your text.');
      return;
    }
    if (!text.trim()) {
      toast('Add a few letters first.');
      return;
    }
    this.stop();
    const utterance = new SpeechSynthesisUtterance(text);
    utterance.lang = 'en-US';
    utterance.rate = Number(opts.rate);
    const voice = this.voices().find((v) => v.voiceURI === opts.voice);
    if (voice) utterance.voice = voice;
    this.utterance = utterance;
    utterance.onend = () => this.update(false);
    utterance.onerror = (event) => {
      this.update(false);
      if (!['interrupted', 'canceled'].includes(event.error))
        toast('This voice could not read the text. Try another English voice in Settings.');
    };
    this.synth.speak(utterance);
    this.update(true);
  }
  toggle() {
    if (!this.synth) return;
    if (this.synth.paused) this.synth.resume();
    else this.synth.pause();
    const el = document.getElementById('pause-speech');
    if (el) {
      el.setAttribute('aria-label', this.synth.paused ? 'Resume speech' : 'Pause speech');
      el.setAttribute('aria-pressed', String(this.synth.paused));
    }
  }
  stop() {
    this.synth?.cancel();
    this.utterance = null;
    this.update(false);
  }
  update(active) {
    for (const id of ['pause-speech', 'stop-speech']) {
      const el = document.getElementById(id);
      if (el) el.disabled = !active;
    }
    const pause = document.getElementById('pause-speech');
    if (pause) {
      pause.setAttribute('aria-label', 'Pause speech');
      pause.setAttribute('aria-pressed', 'false');
    }
  }
}
export const speaker = new Speaker();
window.addEventListener('pagehide', () => speaker.stop());

async function initTranslator() {
  const { initTranslatorLayout } = await import('./translator-layout.js');
  initTranslatorLayout();
  const text = document.getElementById('translation-text');
  const count = document.getElementById('character-count');
  let camera = null;
  let saving = false;
  const update = () => {
    count.textContent = `${text.value.length} / 5,000 characters`;
    for (const id of ['speak-text', 'copy-text'])
      document.getElementById(id).disabled = !text.value.trim();
    document.getElementById('save-text').disabled =
      saving || !text.value.trim() || !settings.save_history || !config.authenticated;
  };
  const saveHint = document.getElementById('save-hint');
  if (config.authenticated && !settings.save_history) {
    const link = document.createElement('a');
    link.href = '/settings';
    link.textContent = 'Enable saving in Settings';
    saveHint.replaceChildren('Saving is turned off. ', link, '.');
  }
  const { TranslatorEditor } = await import('./translator-editor.js');
  const editor = new TranslatorEditor({
    onChange: update,
    resetRecognition: () => camera?.clearMotion(),
    heldLetter: () => (camera?.running ? camera.lastResult?.label : null),
    confirmClear: () =>
      confirmAction('Clear this text?', 'You can restore the cleared text with Undo.', 'Clear'),
    stopSpeech: () => speaker.stop(),
    onLimit: () =>
      toast('This translation has reached 5,000 characters. Save it and start another.'),
  });
  document.getElementById('copy-text').addEventListener('click', () => copyText(text.value));
  document.getElementById('speak-text').addEventListener('click', () => speaker.speak(text.value));
  document.getElementById('pause-speech').addEventListener('click', () => speaker.toggle());
  document.getElementById('stop-speech').addEventListener('click', () => speaker.stop());
  const save = document.getElementById('save-text');
  save.addEventListener('click', async () => {
    if (!config.authenticated || !settings.save_history || saving) return;
    saving = true;
    update();
    try {
      await api('/api/history', { method: 'POST', body: JSON.stringify({ text: text.value }) });
      toast('Saved to your private history.');
    } catch (error) {
      toast(error.message);
    } finally {
      saving = false;
      update();
    }
  });
  const report = document.getElementById('model-report');
  const m = config.model;
  report.textContent = m.ready
    ? `Static model: ${m.static_labels.length} letters, ${m.training_samples} unique training samples, ${(m.test_accuracy * 100).toFixed(1)}% raw accuracy on ${m.test_samples} held-out samples. Split: ${m.split}. ${m.motion?.ready ? `J/Z model: ${(m.motion.test_accuracy * 100).toFixed(1)}% raw accuracy on only ${m.motion.test_samples} held-out source clips; ${m.motion.split}.` : 'J/Z movement model is not installed.'} ${m.low_support_warning || ''} Experimental source-image and clip benchmarks do not establish live accuracy.`
    : 'No trained classifier is installed. Hand tracking and manual text editing remain available.';
  const { CameraController } = await import('./camera.js');
  camera = new CameraController({
    onCommit: (letter) => editor.appendRecognized(letter),
    onResult: (result) => editor.observePrediction(result),
  });
  camera.bind();
  update();
}

async function initLearn() {
  const markCurrentLetter = () =>
    document.querySelectorAll('.alphabet-nav a').forEach((link) => {
      if (link.hash === window.location.hash) link.setAttribute('aria-current', 'location');
      else link.removeAttribute('aria-current');
    });
  window.addEventListener('hashchange', markCurrentLetter);
  markCurrentLetter();
  const key = `signspeak:learning:${config.user_id ?? 'guest'}`;
  const completed = new Set(readStorage(key, []));
  document
    .querySelectorAll('.mark-learned:disabled')
    .forEach((el) => completed.add(el.dataset.letter));
  const render = () => {
    document.getElementById('learn-count').textContent =
      `${completed.size} / ${config.alphabet.length}`;
    document.getElementById('learn-progress').value = completed.size;
    document.querySelectorAll('.mark-learned').forEach((btn) => {
      if (completed.has(btn.dataset.letter)) {
        btn.disabled = true;
        btn.textContent = 'Explored ✓';
        document.querySelector(`[data-completed="${btn.dataset.letter}"]`).textContent =
          'Explored ✓';
      }
    });
  };
  render();
  document.querySelectorAll('.mark-learned').forEach((btn) =>
    btn.addEventListener('click', async () => {
      btn.disabled = true;
      try {
        if (config.authenticated)
          await api('/api/progress', {
            method: 'POST',
            body: JSON.stringify({ letter: btn.dataset.letter }),
          });
        completed.add(btn.dataset.letter);
        writeStorage(key, [...completed]);
        render();
        toast(
          config.authenticated
            ? 'One more letter explored. Progress saved.'
            : 'Letter explored. Progress saved in this browser.',
        );
      } catch (error) {
        btn.disabled = false;
        toast(error.message);
      }
    }),
  );
  try {
    const responses = await Promise.all([
      fetch('/static/models/references.json'),
      fetch('/static/models/motion-references.json'),
    ]);
    if (responses.some((r) => !r.ok)) throw new Error();
    const [refs, motionRefs] = await Promise.all(responses.map((r) => r.json()));
    const { drawHand } = await import('./recognition.js');
    const { drawMotionReference } = await import('./motion.js');
    document.querySelectorAll('.reference-hand').forEach((canvas) => {
      if (canvas.dataset.motion === 'true')
        drawMotionReference(canvas, motionRefs[canvas.dataset.letter], canvas.dataset.letter);
      else drawHand(canvas, refs[canvas.dataset.letter], { reference: true });
    });
  } catch {
    toast(
      'Reference drawings are unavailable. The written handshape instructions are ready to use.',
    );
  }
}

async function initHistory() {
  const list = document.getElementById('history-list'),
    search = document.getElementById('history-search'),
    count = document.getElementById('history-count');
  let requestNumber = 0;
  async function load() {
    const serial = ++requestNumber;
    try {
      const records = await api(`/api/history?q=${encodeURIComponent(search.value)}`);
      if (serial !== requestNumber) return;
      list.replaceChildren();
      count.textContent = `${records.length} saved translation${records.length === 1 ? '' : 's'}`;
      document.getElementById('delete-all-history').disabled =
        records.length === 0 && !search.value;
      if (!records.length) {
        const empty = document.createElement('div');
        empty.className = 'card empty-state';
        empty.append(makeIcon('message'));
        const title = document.createElement('strong');
        title.textContent = search.value
          ? 'No conversations match your search.'
          : 'Your next conversation belongs here.';
        const p = document.createElement('p');
        p.textContent = search.value
          ? 'Try another word or clear your search.'
          : 'Save text from the translator to find it here later.';
        empty.append(title, p);
        list.append(empty);
        return;
      }
      for (const record of records) {
        const row = document.createElement('article');
        row.className = 'history-record';
        const emblem = document.createElement('span');
        emblem.className = 'feature-icon lavender';
        emblem.append(makeIcon('message'));
        const content = document.createElement('div'),
          text = document.createElement('p'),
          time = document.createElement('time');
        text.textContent = record.text;
        time.dateTime = record.created_at;
        time.textContent = new Date(record.created_at).toLocaleString(undefined, {
          dateStyle: 'medium',
          timeStyle: 'short',
        });
        content.append(text, time);
        const actions = document.createElement('div');
        actions.className = 'button-row';
        for (const [glyph, label, action] of [
          ['copy', 'Copy text', () => copyText(record.text)],
          ['volume', 'Read aloud', () => speaker.speak(record.text)],
          [
            'trash',
            'Delete translation',
            async () => {
              if (
                await confirmAction(
                  'Delete this translation?',
                  'This permanently removes this saved text.',
                )
              ) {
                try {
                  await api(`/api/history/${record.id}`, { method: 'DELETE' });
                  await load();
                  toast('Translation deleted.');
                } catch (error) {
                  toast(error.message);
                }
              }
            },
          ],
        ]) {
          const button = document.createElement('button');
          button.className = 'icon-button';
          button.setAttribute('aria-label', label);
          button.append(makeIcon(glyph));
          button.addEventListener('click', action);
          actions.append(button);
        }
        row.append(emblem, content, actions);
        list.append(row);
      }
    } catch (error) {
      if (serial === requestNumber) {
        count.textContent = 'Unable to load history';
        list.replaceChildren();
        const retry = document.createElement('button');
        retry.className = 'button secondary';
        retry.textContent = 'Retry loading history';
        retry.addEventListener('click', load);
        list.append(retry);
        toast(error.message);
      }
    }
  }
  let timer;
  search.addEventListener('input', () => {
    clearTimeout(timer);
    timer = setTimeout(load, 250);
  });
  document.getElementById('delete-all-history').addEventListener('click', async () => {
    if (
      await confirmAction(
        'Delete all your history?',
        'This permanently deletes every saved translation in your account, including records outside this search.',
      )
    ) {
      try {
        await api('/api/history', { method: 'DELETE' });
        await load();
        toast('All saved translations deleted.');
      } catch (error) {
        toast(error.message);
      }
    }
  });
  await load();
}

function initSettings() {
  const form = document.getElementById('settings-form');
  for (const [key, value] of Object.entries(settings)) {
    const input = form.elements.namedItem(key);
    if (input) {
      if (input.type === 'checkbox') input.checked = value;
      else if (
        input.tagName !== 'SELECT' ||
        [...input.options].some((o) => o.value === String(value))
      )
        input.value = value;
    }
  }
  const ranges = () => {
    document.getElementById('threshold-output').textContent =
      `${Math.round(Number(form.threshold.value) * 100)}%`;
    document.getElementById('hold-output').textContent =
      `${(Number(form.hold_ms.value) / 1000).toFixed(1)} s`;
    document.getElementById('rate-output').textContent = `${Number(form.rate.value).toFixed(1)}×`;
  };
  form.addEventListener('input', ranges);
  ranges();
  const voices = () => {
    const select = form.voice;
    select.replaceChildren(new Option('Default English voice', ''));
    speaker.voices().forEach((v) => select.add(new Option(`${v.name} (${v.lang})`, v.voiceURI)));
    select.value = settings.voice;
  };
  voices();
  window.speechSynthesis?.addEventListener('voiceschanged', voices);
  const listCameras = async () => {
    const devices = await navigator.mediaDevices.enumerateDevices();
    const select = form.camera;
    select.replaceChildren(new Option('Default camera', ''));
    devices
      .filter((d) => d.kind === 'videoinput' && d.deviceId)
      .forEach((d, i) => select.add(new Option(d.label || `Camera ${i + 1}`, d.deviceId)));
    select.value = settings.camera;
  };
  document.getElementById('find-cameras').addEventListener('click', async (e) => {
    const button = e.currentTarget;
    button.disabled = true;
    let stream;
    try {
      if (!navigator.mediaDevices?.getUserMedia)
        throw new Error('Camera access requires HTTPS or localhost and a supported browser.');
      stream = await navigator.mediaDevices.getUserMedia({ video: true, audio: false });
      await listCameras();
      toast('Available cameras updated. The camera has been stopped.');
    } catch (error) {
      toast(
        error.name === 'NotAllowedError'
          ? 'Camera permission was denied. You can allow it in your browser settings.'
          : error.message,
      );
    } finally {
      stream?.getTracks().forEach((t) => t.stop());
      button.disabled = false;
    }
  });
  if (navigator.mediaDevices?.enumerateDevices) listCameras().catch(() => {});
  document.getElementById('test-voice').addEventListener('click', () =>
    speaker.speak('A little gesture. A new connection.', {
      rate: Number(form.rate.value),
      voice: form.voice.value,
    }),
  );
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    const button = form.querySelector('[type=submit]');
    button.disabled = true;
    const next = { ...settings };
    for (const key of Object.keys(config.settings)) {
      const input = form.elements.namedItem(key);
      if (input)
        next[key] =
          input.type === 'checkbox'
            ? input.checked
            : input.type === 'range'
              ? Number(input.value)
              : input.value;
    }
    try {
      if (config.authenticated)
        await api('/api/settings', { method: 'POST', body: JSON.stringify(next) });
      Object.assign(settings, next);
      const stored = persistSettings();
      applyTheme(settings.theme);
      toast(
        config.authenticated
          ? 'Your preferences are saved.'
          : stored
            ? 'Preferences saved in this browser.'
            : 'This browser blocks local storage. Preferences apply until you leave this page.',
      );
    } catch (error) {
      toast(error.message);
    } finally {
      button.disabled = false;
    }
  });
}

function initProfile() {
  document.getElementById('profile-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    try {
      await api('/api/profile', {
        method: 'POST',
        body: JSON.stringify({ name: document.getElementById('profile-name').value }),
      });
      toast('Your profile is updated.');
    } catch (error) {
      toast(error.message);
    }
  });
  document.getElementById('delete-account-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    if (
      !(await confirmAction(
        'Permanently delete your account?',
        'Your profile, translations, practice records, and learning progress will all be deleted.',
      ))
    )
      return;
    try {
      await api('/api/profile/delete', {
        method: 'POST',
        body: JSON.stringify({ password: document.getElementById('delete-password').value }),
      });
      try {
        localStorage.removeItem(storageKey);
        localStorage.removeItem(`signspeak:learning:${config.user_id}`);
      } catch {}
      window.location.assign('/');
    } catch (error) {
      toast(error.message);
    }
  });
}

const initializers = {
  translate: initTranslator,
  learn: initLearn,
  history: initHistory,
  settings: initSettings,
  profile: initProfile,
  practice: async () => {
    const module = await import('./practice.js');
    await module.initPractice();
  },
  capture: async () => {
    const module = await import('./capture.js');
    await module.initCapture();
  },
};
if (initializers[config.page])
  Promise.resolve()
    .then(() => initializers[config.page]())
    .catch((error) => {
      console.error(error);
      toast('This page could not finish loading. Refresh to try again.');
    });
