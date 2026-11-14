import test from 'node:test';
import assert from 'node:assert/strict';
import { EditorHistory } from '../signspeak/static/js/editor-history.js';

const snapshot = (
  value,
  selectionStart = value.length,
  selectionEnd = selectionStart,
  extra = {},
) => ({
  value,
  selectionStart,
  selectionEnd,
  ...extra,
});

test('recognized additions, selected corrections and clearing share reversible history', () => {
  const history = new EditorHistory();
  assert.equal(history.canUndo, false);
  assert.equal(history.canRedo, false);
  assert.equal(history.undo(), null);
  assert.equal(history.redo(), null);
  history.record(snapshot('A'));
  history.record(snapshot('AB'));
  history.rememberSelection(snapshot('AB', 1, 2, { selectionDirection: 'forward', scrollTop: 24 }));
  history.record(snapshot('AC'));
  history.record(snapshot(''));

  assert.equal(history.undo().value, 'AC');
  assert.deepEqual(history.undo(), {
    value: 'AB',
    selectionStart: 1,
    selectionEnd: 2,
    selectionDirection: 'forward',
    scrollTop: 24,
  });
  assert.equal(history.undo().value, 'A');
  assert.equal(history.undo().value, '');
  assert.equal(history.undo(), null);
  assert.equal(history.canUndo, false);
  assert.deepEqual(
    Array.from({ length: 4 }, () => history.redo().value),
    ['A', 'AB', 'AC', ''],
  );
  assert.equal(history.redo(), null);
});

test('deletion and keyboard undo preserve the selection before and after edits', () => {
  const history = new EditorHistory();
  history.record(snapshot('HELLO', 1, 4, { selectionDirection: 'backward', scrollTop: 17 }));
  history.record(snapshot('HO', 1));
  const restored = history.undo(snapshot('HO', 0));
  assert.equal(restored.value, 'HELLO');
  assert.equal(restored.selectionStart, 1);
  assert.equal(restored.selectionEnd, 4);
  assert.equal(restored.selectionDirection, 'backward');
  assert.equal(restored.scrollTop, 17);
  assert.equal(history.redo().selectionStart, 0);
});

test('caret-only updates neither create undo steps nor erase redo', () => {
  const history = new EditorHistory();
  history.record(snapshot('A'));
  history.record(snapshot('AB'));
  history.undo();
  assert.equal(history.record(snapshot('A', 0)), false);
  assert.equal(history.canRedo, true);
  assert.equal(history.rememberSelection(snapshot('unrecorded')), false);
  assert.equal(history.redo().value, 'AB');
  assert.equal(history.undo().selectionStart, 0);
  assert.equal(history.undo().value, '');
  assert.equal(history.undo(), null);
});

test('a new edit after Undo replaces the redo branch', () => {
  const history = new EditorHistory();
  history.record(snapshot('A'));
  history.record(snapshot('AB'));
  history.undo();
  assert.equal(history.record(snapshot('AX')), true);
  assert.equal(history.canRedo, false);
  assert.equal(history.redo(), null);
  assert.equal(history.undo().value, 'A');
  assert.equal(history.redo().value, 'AX');
});

test('history bounds the number of undoable edits', () => {
  const history = new EditorHistory({ limit: 2 });
  for (const value of ['A', 'AB', 'ABC', 'ABCD']) history.record(snapshot(value));
  assert.equal(history.undo().value, 'ABC');
  assert.equal(history.undo().value, 'AB');
  assert.equal(history.canUndo, false);
  assert.equal(history.undo(), null);
  assert.equal(history.redo().value, 'ABC');
  assert.equal(history.redo().value, 'ABCD');
});

test('snapshots preserve UTF-16 selections and clamp malformed positions', () => {
  const history = new EditorHistory();
  history.record(snapshot('A👋B', 1, 3, { selectionDirection: 'backward' }));
  history.record(snapshot('AB'));
  const emoji = history.undo();
  assert.equal(emoji.value.slice(emoji.selectionStart, emoji.selectionEnd), '👋');
  history.record(snapshot('A👋B', -8, 999, { scrollTop: -20 }));
  history.redo();
  const clamped = history.undo();
  assert.equal(clamped.selectionStart, 0);
  assert.equal(clamped.selectionEnd, 4);
  assert.equal(clamped.scrollTop, 0);
  history.record(
    snapshot('next', Infinity, NaN, { selectionDirection: 'invalid', scrollTop: Infinity }),
  );
  assert.deepEqual(history.undo(), clamped);
  assert.deepEqual(history.redo(), {
    value: 'next',
    selectionStart: 4,
    selectionEnd: 4,
    selectionDirection: 'none',
    scrollTop: 0,
  });
});

test('history owns its snapshots and rejects invalid configuration or text', () => {
  const history = new EditorHistory();
  const input = snapshot('A');
  history.record(input);
  input.value = 'changed outside history';
  const previous = history.undo();
  previous.value = 'changed returned snapshot';
  assert.equal(history.redo().value, 'A');
  assert.equal(history.undo().value, '');
  for (const limit of [0, -1, 1.5, NaN, Infinity])
    assert.throws(() => new EditorHistory({ limit }), RangeError);
  for (const value of [null, undefined, {}, { value: 1 }])
    assert.throws(() => history.record(value), TypeError);
});
