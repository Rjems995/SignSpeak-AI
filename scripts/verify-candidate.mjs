// Check exported candidate math and rejection gates with the app's actual JS.
import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import {LandmarkClassifier,normalize,gatePrediction} from '../signspeak/static/js/recognition.js';
const directory=path.resolve(process.argv[2]||'artifacts/public-candidate');
const model=JSON.parse(fs.readFileSync(path.join(directory,'classifier.json')));
const report=JSON.parse(fs.readFileSync(path.join(directory,'evaluation.json')));
const dataset=JSON.parse(fs.readFileSync(process.argv[3]||'datasets/public-balanced.json'));
const byFingerprint=new Map(dataset.map(row=>[row.fingerprint,row]));
const classifier=new LandmarkClassifier(model,tf);
let maxDifference=0,rawCorrect=0,gatedCorrect=0;
for(const [index,example] of report.test_examples.entries()){
  const row=byFingerprint.get(report.split_fingerprints.test[index]);
  assert.equal(row.letter,example.expected);
  normalize(row.points).forEach((value,i)=>assert.ok(Math.abs(value-example.features[i])<1e-10));
  const probability=classifier.probabilities(example.features);
  for(const [i,value] of probability.entries())maxDifference=Math.max(maxDifference,Math.abs(value-example.probabilities[i]));
  const result=gatePrediction(probability,example.features,model,.85);
  rawCorrect+=Number(result.candidate===example.expected);
  gatedCorrect+=Number(result.label===example.expected);
}
assert.ok(maxDifference<.00002,`Probability difference ${maxDifference}`);
assert.ok(Math.abs(rawCorrect/report.test_examples.length-report.raw_classifier_accuracy)<1e-10);
assert.ok(Math.abs(gatedCorrect/report.test_examples.length-report.gated_accuracy_at_threshold_085)<1e-10);
console.log(`Candidate verified on ${report.test_examples.length} held-out samples: Python/JS features, TF.js probabilities and rejection gates agree. Maximum probability difference ${maxDifference}.`);
