import { EditorHistory } from './editor-history.js';

// Editing state and undo history belong to this page only. Camera capture can
// continue while insertion is paused; no private text is persisted here.
export class TranslatorEditor {
  constructor({
    onChange = () => {},
    resetRecognition = () => {},
    heldLetter = () => null,
    confirmClear = async () => true,
    stopSpeech = () => {},
    onLimit = () => {},
  } = {}) {
    this.text = document.getElementById('translation-text');
    this.history = new EditorHistory({ limit: 100 });
    this.onChange = onChange;
    this.resetRecognition = resetRecognition;
    this.heldLetter = heldLetter;
    this.confirmClear = confirmClear;
    this.stopSpeech = stopSpeech;
    this.onLimit = onLimit;
    this.paused = false;
    this.waitingForRelease = false;
    this.composing = false;
    this.compositionTimer = null;
    this.latest = this.snapshot();
    this.history.record(this.latest);
    this.bind();
    if (document.activeElement === this.text) this.pause();
    else this.render();
  }

  snapshot() {
    return {
      value: this.text.value,
      selectionStart: this.text.selectionStart,
      selectionEnd: this.text.selectionEnd,
      selectionDirection: this.text.selectionDirection,
      scrollTop: this.text.scrollTop,
    };
  }

  rememberSelection() {
    const current = this.snapshot();
    if (this.history.rememberSelection(current)) this.latest = current;
  }

  commit() {
    this.latest = this.snapshot();
    this.history.record(this.latest);
    this.render();
  }

  announce(message) {
    document.getElementById('editor-announcement').textContent = message;
  }

  pause() {
    if (this.paused) return;
    this.paused = true;
    this.waitingForRelease = false;
    this.resetRecognition();
    this.announce('Automatic letter insertion paused. Select Resume when you are ready.');
    this.render();
  }

  resume() {
    if (this.composing || !this.paused) return;
    this.flushComposition();
    this.waitingForRelease = Boolean(this.heldLetter());
    this.resetRecognition();
    this.paused = false;
    this.render();
    this.announce(
      this.waitingForRelease
        ? 'Release your hand, then sign again to add letters.'
        : 'Automatic letter insertion resumed.',
    );
    // Hiding Resume must not leave focus on a hidden control or return it to
    // the textarea, which would immediately pause insertion again.
    const target = document.getElementById('text-workspace').hidden
      ? document.getElementById('workspace-tab-camera')
      : document.getElementById('text-heading');
    target.focus({ preventScroll: true });
  }

  observePrediction(prediction) {
    if (!this.paused && this.waitingForRelease && !prediction?.label) {
      this.waitingForRelease = false;
      this.render();
      this.announce('Automatic letter insertion resumed.');
    }
  }

  appendRecognized(letter) {
    if (document.activeElement === this.text) this.pause();
    if (
      this.paused ||
      this.waitingForRelease ||
      this.composing ||
      document.querySelector('dialog[open]')
    )
      return false;
    if (!/^[A-Z]$/.test(letter)) return false;
    if (this.text.value.length >= this.text.maxLength) {
      this.onLimit();
      return false;
    }
    this.rememberSelection();
    this.text.value += letter;
    this.commit();
    return true;
  }

  flushComposition() {
    if (this.compositionTimer !== null) {
      clearTimeout(this.compositionTimer);
      this.compositionTimer = null;
      this.commit();
    }
  }

  focusText() {
    if (document.getElementById('text-workspace').hidden) {
      const tab = document.getElementById('workspace-tab-text');
      tab.focus({ preventScroll: true });
      tab.click();
    }
    this.text.focus({ preventScroll: true });
    this.text.scrollIntoView({ block: 'nearest', inline: 'nearest' });
  }

  restore(snapshot) {
    this.text.value = snapshot.value;
    this.focusText();
    this.text.setSelectionRange(
      snapshot.selectionStart,
      snapshot.selectionEnd,
      snapshot.selectionDirection,
    );
    this.text.scrollTop = snapshot.scrollTop;
    this.latest = this.snapshot();
    this.stopSpeech();
    this.render();
  }

  travel(direction, native = false) {
    if (this.composing) return;
    this.pause();
    if (!native) {
      this.flushComposition();
      this.rememberSelection();
    }
    const snapshot = direction === 'undo' ? this.history.undo() : this.history.redo();
    if (snapshot) {
      this.restore(snapshot);
      this.announce(
        `${direction === 'undo' ? 'Undo' : 'Redo'} complete. Automatic insertion is paused.`,
      );
    } else if (native) this.restore(this.latest);
  }

  replaceSelection(value, removePrevious = false) {
    if (this.composing) return;
    this.pause();
    this.flushComposition();
    this.rememberSelection();
    let start = this.text.selectionStart;
    const end = this.text.selectionEnd;
    if (removePrevious && start === end)
      start -= Array.from(this.text.value.slice(0, start)).at(-1)?.length || 0;
    if (this.text.value.length - (end - start) + value.length > this.text.maxLength) return;
    this.text.setRangeText(value, start, end, 'end');
    this.commit();
    this.focusText();
  }

  async clear() {
    if (this.composing) return;
    this.pause();
    this.flushComposition();
    this.rememberSelection();
    if (await this.confirmClear()) {
      this.text.value = '';
      this.stopSpeech();
      this.commit();
    }
  }

  render() {
    this.onChange();
    document.getElementById('undo-text').disabled = this.composing || !this.history.canUndo;
    document.getElementById('redo-text').disabled = this.composing || !this.history.canRedo;
    document.getElementById('add-space').disabled = this.composing;
    for (const id of ['delete-letter', 'clear-text'])
      document.getElementById(id).disabled = this.composing || !this.text.value.length;
    document.getElementById('editor-paused').hidden = !this.paused && !this.waitingForRelease;
    document.getElementById('editor-pause-message').textContent = this.paused
      ? 'Automatic insertion paused'
      : 'Release your hand to continue';
    const resume = document.getElementById('resume-insertion');
    resume.hidden = !this.paused;
    resume.disabled = this.composing;
  }

  bind() {
    this.text.addEventListener('focus', () => this.pause());
    this.text.addEventListener('select', () => this.rememberSelection());
    this.text.addEventListener('beforeinput', (event) => {
      this.pause();
      if (this.composing || event.isComposing) return;
      if (['historyUndo', 'historyRedo'].includes(event.inputType)) {
        this.rememberSelection();
        if (event.cancelable) {
          event.preventDefault();
          this.travel(event.inputType === 'historyUndo' ? 'undo' : 'redo');
        }
        return;
      }
      this.flushComposition();
      this.rememberSelection();
    });
    this.text.addEventListener('input', (event) => {
      this.pause();
      if (this.composing || event.isComposing) {
        this.render();
        return;
      }
      if (['historyUndo', 'historyRedo'].includes(event.inputType)) {
        this.travel(event.inputType === 'historyUndo' ? 'undo' : 'redo', true);
        return;
      }
      clearTimeout(this.compositionTimer);
      this.compositionTimer = null;
      this.commit();
    });
    this.text.addEventListener('compositionstart', () => {
      this.flushComposition();
      this.rememberSelection();
      this.composing = true;
      this.pause();
      this.render();
    });
    this.text.addEventListener('compositionend', () => {
      this.composing = false;
      this.compositionTimer = setTimeout(() => {
        this.compositionTimer = null;
        this.commit();
      }, 0);
      this.render();
    });
    document.addEventListener('keydown', (event) => {
      if (
        !(event.ctrlKey || event.metaKey) ||
        event.altKey ||
        event.isComposing ||
        this.composing ||
        document.querySelector('dialog[open]')
      )
        return;
      const target = event.target;
      if (
        target instanceof Element &&
        target !== this.text &&
        target.closest('input,textarea,select,[contenteditable]:not([contenteditable="false"])')
      )
        return;
      const key = event.key.toLowerCase();
      if (key !== 'z' && key !== 'y') return;
      event.preventDefault();
      this.travel(key === 'y' || event.shiftKey ? 'redo' : 'undo');
    });
    document.getElementById('undo-text').addEventListener('click', () => this.travel('undo'));
    document.getElementById('redo-text').addEventListener('click', () => this.travel('redo'));
    document
      .getElementById('add-space')
      .addEventListener('click', () => this.replaceSelection(' '));
    document
      .getElementById('delete-letter')
      .addEventListener('click', () => this.replaceSelection('', true));
    document.getElementById('clear-text').addEventListener('click', () => this.clear());
    document.getElementById('resume-insertion').addEventListener('click', () => this.resume());
  }
}
