import { config, api, toast } from './app.js';
import { CameraController } from './camera.js';
import { drawHand } from './recognition.js';
import { drawMotionReference } from './motion.js';
import { PRACTICE_FOCUSES, buildPracticeTargets } from './practice-targets.js';

export async function initPractice() {
  const get = (id) => document.getElementById(id),
    labels = config.model.labels ?? [];
  const focus = get('practice-focus');
  const initial = document.querySelector('.practice-prompt').dataset.target;
  const modes = [
    ...PRACTICE_FOCUSES,
    ...(initial
      ? [{ id: 'letter', label: `Letter ${initial}`, letters: [initial], rounds: 5 }]
      : []),
  ];
  const selectedMode = () => modes.find((mode) => mode.id === focus.value);
  const pausedFeedback = 'Camera paused. Enable it again to continue this session.';
  const available = (mode) =>
    mode &&
    (mode.letters ? mode.letters.every((letter) => labels.includes(letter)) : labels.length);
  for (const option of focus.options)
    option.disabled = !available(modes.find((mode) => mode.id === option.value));
  if (!available(selectedMode())) focus.value = 'mixed';
  let targets = [],
    index = 0,
    results = [],
    active = false,
    checked = false,
    refs = {},
    motionRefs = {},
    saved = false,
    saving = false,
    sessionFocus = null,
    promptStartedAt = 0;
  try {
    const response = await fetch('/static/models/references.json');
    if (response.ok) refs = await response.json();
  } catch {}
  try {
    const response = await fetch('/static/models/motion-references.json');
    if (response.ok) motionRefs = await response.json();
  } catch {}
  const camera = new CameraController({
    onState: (on) => {
      updateAvailability();
      get('practice-check').disabled = !on;
      if (active)
        get('practice-feedback').textContent = !on
          ? pausedFeedback
          : checked
            ? 'Camera ready. Your progress is kept; continue when you are ready.'
            : promptInstruction();
    },
  });
  camera.bind();
  const target = () => targets[index];
  function promptInstruction() {
    return config.model.motion_labels.includes(target())
      ? 'Select Record J / Z, trace the letter once, then select “Check my sign”.'
      : 'Hold your letter steady, then select “Check my sign”.';
  }
  function updateAvailability() {
    get('practice-start').disabled =
      active || saving || !camera.running || !camera.classifier || !available(selectedMode());
    get('practice-again').disabled = saving || !available(selectedMode());
    focus.disabled = active || saving;
  }
  function describeFocus() {
    const mode = active ? sessionFocus : selectedMode();
    const description =
      mode.id === 'mixed'
        ? '5 letters from the alphabet.'
        : mode.id === 'letter'
          ? `Repeat ${initial} for all 5 prompts.`
          : `6 prompts using only ${mode.letters.join(' and ')}, 3 of each.`;
    get('practice-focus-status').textContent =
      `${mode.label}: ${description}${active ? ' Finish this session to change focus.' : ''}`;
    updateAvailability();
  }
  function show() {
    checked = false;
    promptStartedAt = performance.now();
    camera.clearMotion();
    const motion = config.model.motion_labels.includes(target());
    get('practice-step').textContent =
      `LETTER ${index + 1} OF ${targets.length}${motion ? ' · MOVEMENT' : ''}`;
    get('practice-letter').textContent = target();
    get('practice-heading').textContent = `${motion ? 'Trace' : 'Show'} the ASL letter ${target()}`;
    get('practice-feedback').textContent = camera.running ? promptInstruction() : pausedFeedback;
    const previous = get('practice-reference'),
      canvas = previous.cloneNode();
    previous.replaceWith(canvas);
    canvas.hidden = false;
    if (motion) drawMotionReference(canvas, motionRefs[target()], target());
    else drawHand(canvas, refs[target()], { reference: true });
    get('practice-start').hidden = true;
    get('practice-check').hidden = false;
    get('practice-check').textContent = 'Check my sign';
    get('practice-check').disabled = !camera.running;
    get('practice-next').hidden = true;
    get('practice-skip').hidden = false;
    get('practice-finish').hidden = false;
  }
  function start() {
    if (active || saving) return;
    if (!camera.running || !camera.classifier)
      return toast('Enable your camera and load the classifier first.');
    const nextTargets = buildPracticeTargets({ labels, focus: focus.value, initial });
    if (!nextTargets.length) return toast('The selected focus is unavailable with this model.');
    results = [];
    index = 0;
    active = true;
    saved = false;
    targets = nextTargets;
    sessionFocus = selectedMode();
    get('session-summary').hidden = true;
    get('practice-save-retry').hidden = true;
    describeFocus();
    show();
  }
  async function save() {
    if (saved || saving || active || !results.length) return;
    if (!config.authenticated) {
      get('session-save-status').textContent =
        'This session is complete. Sign in to save future practice sessions to your dashboard.';
      return;
    }
    const completedResults = results.map((result) => ({ ...result }));
    saving = true;
    updateAvailability();
    get('practice-save-retry').disabled = true;
    try {
      await api('/api/practice', {
        method: 'POST',
        body: JSON.stringify({ results: completedResults }),
      });
      saved = true;
      get('session-save-status').textContent = 'Your session is saved to your account.';
      get('practice-save-retry').hidden = true;
    } catch (error) {
      get('session-save-status').textContent = `Session is not saved: ${error.message}`;
      get('practice-save-retry').hidden = false;
    } finally {
      saving = false;
      get('practice-save-retry').disabled = false;
      updateAvailability();
    }
  }
  async function finish() {
    if (!active) return;
    active = false;
    camera.stop();
    describeFocus();
    for (const id of ['practice-check', 'practice-next', 'practice-skip', 'practice-finish'])
      get(id).hidden = true;
    get('practice-heading').textContent = 'A little progress, made by you.';
    get('practice-feedback').textContent = results.length
      ? 'Every attempt is a step forward. Review your session below.'
      : 'No attempts recorded. Enable the camera to start a new session.';
    get('practice-start').hidden = false;
    get('practice-start').textContent = 'Start a new session';
    if (!results.length) return;
    const correct = results.filter((r) => r.predicted === r.expected).length;
    get('summary-correct').textContent = correct;
    get('summary-attempts').textContent = results.length;
    get('summary-accuracy').textContent = `${Math.round((correct / results.length) * 100)}%`;
    get('summary-focus').textContent = `${sessionFocus.label} · ${targets.length} prompts`;
    get('session-summary').hidden = false;
    await save();
  }
  function record(predicted) {
    results.push({ expected: target(), predicted });
    checked = true;
    const correct = predicted === target();
    get('practice-heading').textContent = correct
      ? 'You’ve got this.'
      : 'Try this shape once more.';
    get('practice-feedback').textContent = correct
      ? `The model accepted ${target()}. Ready for another letter?`
      : predicted
        ? `The model read ${predicted}. Review the reference, then try again or move on.`
        : 'Skipped. You can return to this handshape in the learning cards.';
    get('practice-next').hidden = false;
    get('practice-next').textContent =
      index === targets.length - 1 ? 'See my session' : 'Next letter →';
    get('practice-check').hidden = correct || predicted === null;
    get('practice-check').textContent = 'Try again';
    get('practice-skip').hidden = true;
    if (results.length >= 60) finish();
  }
  get('practice-start').addEventListener('click', start);
  get('practice-check').addEventListener('click', () => {
    if (!active) return;
    const prediction = camera.current(),
      motion = config.model.motion_labels.includes(target());
    if (
      !prediction?.label ||
      prediction.timestamp <= promptStartedAt ||
      !prediction.stable ||
      prediction.progress < 1 ||
      (motion && !prediction.motion)
    )
      return toast(
        motion
          ? 'Record your J/Z movement, then check the accepted result.'
          : 'Hold a supported letter steady until the indicator is full, then check again.',
      );
    record(prediction.label);
  });
  get('practice-skip').addEventListener('click', () => {
    if (active) record(null);
  });
  get('practice-next').addEventListener('click', () => {
    if (!active || !checked) return;
    index++;
    if (index >= targets.length) finish();
    else show();
  });
  get('practice-finish').addEventListener('click', finish);
  focus.addEventListener('change', describeFocus);
  get('practice-save-retry').addEventListener('click', save);
  get('practice-again').addEventListener('click', async () => {
    if (active || saving) return;
    await camera.start();
    if (camera.running) start();
  });
  describeFocus();
  if (!config.model.ready)
    get('practice-feedback').textContent =
      'A trained classifier must be installed before camera practice can begin.';
}
