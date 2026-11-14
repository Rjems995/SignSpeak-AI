export const PRACTICE_FOCUSES = Object.freeze([
  Object.freeze({ id: 'mixed', label: 'Mixed alphabet', letters: null, rounds: 5 }),
  Object.freeze({ id: 'ru', label: 'R / U', letters: Object.freeze(['R', 'U']), rounds: 6 }),
  Object.freeze({ id: 'mn', label: 'M / N', letters: Object.freeze(['M', 'N']), rounds: 6 }),
]);

// Focused sessions never fall back to unrelated letters when a model lacks a target.
export function buildPracticeTargets({
  labels = [],
  focus = 'mixed',
  initial = '',
  random = Math.random,
} = {}) {
  const supported = [...new Set(labels.filter((letter) => /^[A-Z]$/.test(letter)))];
  if (!supported.length) return [];
  if (focus === 'letter') return supported.includes(initial) ? Array(5).fill(initial) : [];

  const mode = PRACTICE_FOCUSES.find((option) => option.id === focus);
  if (!mode) return [];
  if (mode.letters) {
    if (!mode.letters.every((letter) => supported.includes(letter))) return [];
    const targets = [];
    // Each two-prompt round covers both shapes; randomize which comes first.
    for (let round = 0; round < mode.rounds / mode.letters.length; round++)
      targets.push(...(random() < 0.5 ? mode.letters : [...mode.letters].reverse()));
    return targets;
  }

  const targets = supported.includes(initial) ? [initial] : [];
  while (targets.length < mode.rounds) {
    const alternatives = supported.filter((letter) => letter !== targets.at(-1));
    const options = alternatives.length ? alternatives : supported;
    targets.push(options[Math.floor(random() * options.length)]);
  }
  return targets;
}
