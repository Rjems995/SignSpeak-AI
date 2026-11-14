import test from 'node:test';
import assert from 'node:assert/strict';
import { PRACTICE_FOCUSES, buildPracticeTargets } from '../signspeak/static/js/practice-targets.js';

const alphabet = Array.from('ABCDEFGHIJKLMNOPQRSTUVWXYZ');

test('confusion-pair sessions contain only the selected pair with equal exposure', () => {
  for (const mode of PRACTICE_FOCUSES.filter((option) => option.letters)) {
    let seed = 7;
    const random = () => {
      seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
      return seed / 2 ** 32;
    };
    for (let session = 0; session < 100; session++) {
      const targets = buildPracticeTargets({ labels: alphabet, focus: mode.id, random });
      assert.equal(targets.length, 6);
      for (const letter of mode.letters)
        assert.equal(targets.filter((target) => target === letter).length, 3);
      for (let index = 0; index < targets.length; index += 2)
        assert.deepEqual(new Set(targets.slice(index, index + 2)), new Set(mode.letters));
    }
  }
});

test('each pair round can change its order without changing the practice focus', () => {
  const choices = [0.1, 0.8, 0.2];
  assert.deepEqual(
    buildPracticeTargets({ labels: alphabet, focus: 'ru', random: () => choices.shift() }),
    ['R', 'U', 'U', 'R', 'R', 'U'],
  );
  assert.deepEqual(buildPracticeTargets({ labels: alphabet, focus: 'mn', random: () => 0.99 }), [
    'N',
    'M',
    'N',
    'M',
    'N',
    'M',
  ]);
});

test('unavailable or unknown focuses cannot silently become mixed sessions', () => {
  assert.deepEqual(buildPracticeTargets({ labels: ['R', 'A'], focus: 'ru' }), []);
  assert.deepEqual(buildPracticeTargets({ labels: ['M', 'A'], focus: 'mn' }), []);
  assert.deepEqual(buildPracticeTargets({ labels: alphabet, focus: 'unknown' }), []);
});

test('mixed practice honors the starting letter and avoids immediate repeats', () => {
  const labels = Object.freeze(['A', 'B', 'C', 'A', '', 'AB']);
  assert.deepEqual(buildPracticeTargets({ labels, initial: 'B', random: () => 0 }), [
    'B',
    'A',
    'B',
    'A',
    'B',
  ]);
  assert.deepEqual(buildPracticeTargets({ labels, initial: 'unsupported', random: () => 0.99 }), [
    'C',
    'B',
    'C',
    'B',
    'C',
  ]);
});

test('empty and single-letter models produce bounded, supported mixed sessions', () => {
  assert.deepEqual(buildPracticeTargets(), []);
  assert.deepEqual(buildPracticeTargets({ labels: ['', 'AA', 'a'] }), []);
  assert.deepEqual(buildPracticeTargets({ labels: ['R'] }), ['R', 'R', 'R', 'R', 'R']);
});

test('single-letter focus repeats only the requested supported letter', () => {
  assert.deepEqual(buildPracticeTargets({ labels: alphabet, focus: 'letter', initial: 'R' }), [
    'R',
    'R',
    'R',
    'R',
    'R',
  ]);
  assert.deepEqual(buildPracticeTargets({ labels: alphabet, focus: 'letter' }), []);
  assert.deepEqual(buildPracticeTargets({ labels: ['M', 'N'], focus: 'letter', initial: 'R' }), []);
});
