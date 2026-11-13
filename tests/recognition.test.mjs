import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as tf from '@tensorflow/tfjs';
import {
  normalize,
  staticFeatures,
  WORLD_FEATURE_VERSION,
  denseProbabilities,
  topologyFeatures,
  topologyProbability,
  Stabilizer,
  LandmarkClassifier,
  gatePrediction,
} from '../signspeak/static/js/recognition.js';
import { motionFeatures, MotionClassifier } from '../signspeak/static/js/motion.js';
const model = JSON.parse(
  fs.readFileSync(new URL('../signspeak/static/models/classifier.json', import.meta.url)),
);
const report = JSON.parse(
  fs.readFileSync(new URL('../signspeak/static/models/evaluation.json', import.meta.url)),
);
const refs = JSON.parse(
  fs.readFileSync(new URL('../signspeak/static/models/references.json', import.meta.url)),
);

test('JavaScript and Python normalization agree on a recorded reference', () => {
  const points = refs.L;
  const expected = normalize(points);
  const shifted = points.map((v, i) => v * 3 + (i % 2 ? -0.7 : 0.3));
  normalize(shifted).forEach((v, i) => assert.ok(Math.abs(v - expected[i]) < 1e-10));
  normalize(points.map((v, i) => (i % 2 ? v : -v))).forEach((v, i) =>
    assert.ok(Math.abs(v - expected[i]) < 1e-10),
  );
});
test('TensorFlow.js reproduces Python held-out predictions', () => {
  const classifier = new LandmarkClassifier(model, tf);
  for (const example of report.test_examples) {
    const probability = classifier.probabilities(example.features);
    probability.forEach((value, i) =>
      assert.ok(Math.abs(value - example.probabilities[i]) < 0.00002),
    );
  }
});
test('a held letter commits once and requires release before repetition', () => {
  const s = new Stabilizer(500);
  const commits = [];
  for (let now = 0; now < 2000; now += 100) {
    const r = s.update('A', now);
    if (r.commit) commits.push(r.commit);
  }
  assert.deepEqual(commits, ['A']);
  for (let now = 2000; now < 2500; now += 100) s.update(null, now);
  for (let now = 2500; now < 4000; now += 100) {
    const r = s.update('A', now);
    if (r.commit) commits.push(r.commit);
  }
  assert.deepEqual(commits, ['A', 'A']);
});
test('uncertain frames cannot stabilize and unknown geometry is rejected', () => {
  const s = new Stabilizer(500);
  for (let now = 0; now < 3000; now += 100)
    assert.equal(s.update(now % 200 ? 'A' : 'B', now).commit, null);
  const p = model.labels.map((x) => (x === 'A' ? 0.99 : 0.01 / 6));
  assert.equal(gatePrediction(p, Array(42).fill(100), model, 0.85).label, null);
  assert.equal(
    gatePrediction(
      model.labels.map(() => 1 / 7),
      Array(42).fill(0),
      model,
      0.85,
    ).label,
    null,
  );
});

test('real reviewed unknown poses match Python gates and cannot commit rejected letters', () => {
  const depth = model.feature_version === WORLD_FEATURE_VERSION;
  const evaluation = JSON.parse(
    fs.readFileSync(
      new URL(
        depth ? '../datasets/weak-letter-evaluation.json' : '../datasets/unknown-evaluation.json',
        import.meta.url,
      ),
    ),
  );
  const classifier = new LandmarkClassifier(model, tf);
  let accepted = 0;
  for (const row of depth ? evaluation.unknown_test_examples : evaluation.test_examples) {
    const result = classifier.predictFeatures(row.features, 0.85);
    assert.equal(result.label, row.accepted, `${row.category} ${row.fingerprint}`);
    if (result.label) {
      accepted++;
      continue;
    }
    const stabilizer = new Stabilizer(500);
    // A rejection interrupts a nearly completed valid-letter hold.
    for (let t = 0; t < 700; t += 100) stabilizer.update('A', t);
    assert.equal(stabilizer.update(null, 700).progress, 0);
    for (let t = 800; t < 2500; t += 100)
      assert.equal(stabilizer.update(result.label, t).commit, null);
  }
  assert.ok(accepted <= 2); // Keep any remaining failures visible.
  assert.equal(evaluation.candidate.test.unknown.false_acceptances, accepted);
});

test('image/world features and selective legacy rescue agree with Python', () => {
  const candidate = JSON.parse(
    fs.readFileSync(
      new URL('../datasets/candidates/focus-kernel-v4/classifier.json', import.meta.url),
    ),
  );
  const evaluation = JSON.parse(
    fs.readFileSync(
      new URL('../datasets/candidates/focus-kernel-v4/evaluation.json', import.meta.url),
    ),
  );
  const rows = JSON.parse(
    fs.readFileSync(new URL('../datasets/focus-depth.json', import.meta.url)),
  ).samples;
  const byFingerprint = new Map(rows.map((row) => [row.fingerprint, row]));
  const classifier = new LandmarkClassifier(candidate, tf);
  let rescued = 0;
  for (const [index, example] of evaluation.test_examples.entries()) {
    const row = byFingerprint.get(evaluation.split_fingerprints.test[index]);
    const features = staticFeatures(row.points, row.world_points, candidate.feature_version);
    features.forEach((v, i) => assert.ok(Math.abs(v - example.features[i]) < 1e-9));
    const result = classifier.predict(row.points, 0.85, row.world_points);
    assert.equal(result.label, example.accepted);
    if (result.expert) rescued++;
  }
  assert.ok(rescued > 0);
  assert.equal(classifier.predict(refs.L, 0.85).label, null, 'missing depth must reject');
  const legacy = candidate.rescue.model,
    example = evaluation.test_examples[0];
  const manual = denseProbabilities(legacy, example.features.slice(0, 42));
  const tfLegacy = new LandmarkClassifier(legacy, tf).probabilities(example.features.slice(0, 42));
  manual.forEach((v, i) => assert.ok(Math.abs(v - tfLegacy[i]) < 0.00002));
});

test('unreleased candidate rejects every reviewed unknown regression example', () => {
  const candidate = JSON.parse(
    fs.readFileSync(
      new URL('../datasets/candidates/focus-kernel-v4/classifier.json', import.meta.url),
    ),
  );
  const evaluation = JSON.parse(
    fs.readFileSync(new URL('../datasets/weak-letter-evaluation.json', import.meta.url)),
  );
  const classifier = new LandmarkClassifier(candidate, tf);
  assert.equal(evaluation.release_passed, false);
  for (const row of evaluation.unknown_test_examples)
    assert.equal(classifier.predictFeatures(row.features, 0.85).label, row.accepted);
  assert.equal(evaluation.candidate.test.unknown.false_acceptances, 0);
});

test('crossing expert matches Python geometry, probabilities and all reused benchmark gates', () => {
  const candidate = JSON.parse(
    fs.readFileSync(
      new URL('../datasets/candidates/focus-topology-v5/classifier.json', import.meta.url),
    ),
  );
  const evaluation = JSON.parse(
    fs.readFileSync(
      new URL('../datasets/candidates/focus-topology-v5/evaluation.json', import.meta.url),
    ),
  );
  const validation = JSON.parse(
    fs.readFileSync(new URL('../datasets/weak-letter-topology-validation.json', import.meta.url)),
  );
  const diagnostic = JSON.parse(
    fs.readFileSync(new URL('../datasets/weak-letter-topology-evaluation.json', import.meta.url)),
  );
  for (const row of validation.test_vectors) {
    topologyFeatures(row.xy).forEach((v, i) => assert.ok(Math.abs(v - row.features[i]) < 1e-12));
    assert.ok(
      Math.abs(topologyProbability(candidate.topology_expert, row.xy) - row.probability) < 1e-12,
    );
  }
  const classifier = new LandmarkClassifier(candidate, tf);
  let topology = 0;
  for (const row of evaluation.test_examples) {
    const probability = classifier.probabilities(row.features);
    probability.forEach((v, i) => assert.ok(Math.abs(v - row.probabilities[i]) < 1e-9));
    const result = classifier.predictFeatures(row.features, 0.85);
    assert.equal(result.label, row.accepted);
    if (result.expert === 'finger_topology') topology++;
  }
  assert.ok(topology >= 27, 'the R improvement must exercise the trained crossing expert');
  for (const row of diagnostic.unknown_test_examples)
    assert.equal(classifier.predictFeatures(row.features, 0.85).label, null);
  assert.equal(diagnostic.release_passed, false);
  assert.deepEqual(
    diagnostic.release_failures.map((row) => row.criterion),
    ['overall_gated_accuracy'],
  );
  const row = evaluation.test_examples[0],
    probabilities = row.probabilities;
  for (const changes of [
    { version: 'bad' },
    { scale: Array(12).fill(0) },
    { weights: Array(12).fill(NaN) },
    { threshold: 0 },
    { allowed_primary_labels: ['A'] },
  ]) {
    const expert = { ...candidate.topology_expert, ...changes };
    assert.throws(() => topologyProbability(expert, row.features.slice(0, 42)));
    assert.equal(
      gatePrediction(probabilities, row.features, { ...candidate, topology_expert: expert }, 0.85)
        .label,
      null,
    );
  }
});

test('rejection gates fail closed for malformed coordinates, probabilities and config', () => {
  const classifier = new LandmarkClassifier(model, tf);
  for (const points of [null, [], Array(42).fill(0), Array(42).fill(NaN)])
    assert.equal(classifier.predict(points, 0.85).label, null);
  const row = report.test_examples[0],
    probabilities = row.probabilities;
  for (const features of [[], row.features.map(() => Infinity)])
    assert.equal(gatePrediction(probabilities, features, model, 0.85).label, null);
  for (const probabilities of [[], model.labels.map(() => NaN), model.labels.map(() => -1)])
    assert.equal(gatePrediction(probabilities, row.features, model, 0.85).label, null);
  assert.equal(gatePrediction(probabilities, row.features, model, NaN).label, null);
  // A guaranteed pre-gate acceptance lets us test malformed veto config.
  const features = model.support.A[0],
    p = model.labels.map((label) => (label === 'A' ? 1 : 0));
  assert.equal(
    gatePrediction(p, features, { ...model, rejection: { version: 'bad' } }, 0.85).label,
    null,
  );
});

test('all 24 static letters are trained and orientation remains meaningful', () => {
  assert.deepEqual(
    [...model.labels].sort(),
    [...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'].filter((l) => l !== 'J' && l !== 'Z'),
  );
  assert.notDeepEqual(normalize(refs.K), normalize(refs.K.map((v) => -v)));
});
test('real J/Z clips match Python temporal features and TF.js probabilities', () => {
  const motion = JSON.parse(
    fs.readFileSync(new URL('../signspeak/static/models/motion-classifier.json', import.meta.url)),
  );
  const evaluation = JSON.parse(
    fs.readFileSync(new URL('../signspeak/static/models/motion-evaluation.json', import.meta.url)),
  );
  const classifier = new MotionClassifier(motion, tf);
  assert.deepEqual(
    motion.labels.filter((label) => label !== '_'),
    ['J', 'Z'],
  );
  for (const example of evaluation.test_examples) {
    const features = motionFeatures(example.frames);
    features.forEach((v, i) => assert.ok(Math.abs(v - example.features[i]) < 1e-9));
    classifier
      .probabilities(features)
      .forEach((v, i) => assert.ok(Math.abs(v - example.probabilities[i]) < 0.00002));
  }
  const points = evaluation.test_examples[0].frames[0].points;
  assert.throws(
    () => motionFeatures(Array.from({ length: 20 }, (_, i) => ({ t: i * 100, points }))),
    /stationary/,
  );
  assert.throws(() => motionFeatures(evaluation.test_examples[0].frames.slice(0, 3)), /eight/);
});

test('complete fast movement retains its trajectory and very short recordings are rejected', () => {
  const evaluation = JSON.parse(
    fs.readFileSync(new URL('../signspeak/static/models/motion-evaluation.json', import.meta.url)),
  );
  const frames = evaluation.test_examples.find((row) => row.expected === 'J').frames;
  const timed = (duration) =>
    frames.map((frame) => ({
      ...frame,
      t: ((frame.t - frames[0].t) / (frames.at(-1).t - frames[0].t)) * duration,
    }));
  motionFeatures(timed(300)).forEach((value, i) =>
    assert.ok(Math.abs(value - motionFeatures(frames)[i]) < 1e-9),
  );
  assert.throws(() => motionFeatures(timed(150)), /throughout/);
});
