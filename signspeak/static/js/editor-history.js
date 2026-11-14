// In-memory text history shared by typing, recognition and editor buttons.
// Remember the current selection before recording a text change so Undo can
// return to the correction the user was making, not just restore its characters.
function normalizeSnapshot(snapshot) {
  if (!snapshot || typeof snapshot.value !== 'string') {
    throw new TypeError('An editor snapshot needs a string value.');
  }
  const { value } = snapshot;
  const position = (index, fallback) =>
    Number.isFinite(index) ? Math.max(0, Math.min(value.length, Math.trunc(index))) : fallback;
  const selectionStart = position(snapshot.selectionStart, value.length);
  const selectionEnd = Math.max(selectionStart, position(snapshot.selectionEnd, selectionStart));
  return {
    value,
    selectionStart,
    selectionEnd,
    selectionDirection: ['forward', 'backward', 'none'].includes(snapshot.selectionDirection)
      ? snapshot.selectionDirection
      : 'none',
    scrollTop: Number.isFinite(snapshot.scrollTop) ? Math.max(0, snapshot.scrollTop) : 0,
  };
}

export class EditorHistory {
  constructor({ limit = 100 } = {}) {
    if (!Number.isInteger(limit) || limit < 1) {
      throw new RangeError('The undo limit must be a positive integer.');
    }
    this.limit = limit;
    this.states = [normalizeSnapshot({ value: '' })];
    this.index = 0;
  }

  get canUndo() {
    return this.index > 0;
  }

  get canRedo() {
    return this.index < this.states.length - 1;
  }

  // Same-value updates only remember caret/selection/scroll changes. They must
  // not add an undo step or discard a redo branch.
  record(snapshot) {
    const next = normalizeSnapshot(snapshot);
    if (next.value === this.states[this.index].value) {
      this.states[this.index] = next;
      return false;
    }
    this.states.splice(this.index + 1);
    this.states.push(next);
    this.index++;
    if (this.states.length > this.limit + 1) {
      this.states.shift();
      this.index--;
    }
    return true;
  }

  rememberSelection(snapshot) {
    const next = normalizeSnapshot(snapshot);
    if (next.value !== this.states[this.index].value) return false;
    this.states[this.index] = next;
    return true;
  }

  // All text changes should be recorded first; the optional current snapshot
  // updates the selection being left behind when navigating recorded states.
  undo(currentSnapshot) {
    if (currentSnapshot !== undefined) this.rememberSelection(currentSnapshot);
    if (!this.canUndo) return null;
    return { ...this.states[--this.index] };
  }

  redo(currentSnapshot) {
    if (currentSnapshot !== undefined) this.rememberSelection(currentSnapshot);
    if (!this.canRedo) return null;
    return { ...this.states[++this.index] };
  }
}
