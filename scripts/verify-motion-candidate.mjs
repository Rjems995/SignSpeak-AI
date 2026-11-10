import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import * as tf from '@tensorflow/tfjs';
import {MotionClassifier,motionFeatures} from '../signspeak/static/js/motion.js';
const directory=path.resolve(process.argv[2]||'artifacts/motion-candidate');
const model=JSON.parse(fs.readFileSync(path.join(directory,'motion-classifier.json')));
const report=JSON.parse(fs.readFileSync(path.join(directory,'motion-evaluation.json')));
const classifier=new MotionClassifier(model,tf);
let maxDifference=0,rawCorrect=0,gatedCorrect=0;
for(const example of report.test_examples){
  const features=motionFeatures(example.frames);
  features.forEach((value,i)=>assert.ok(Math.abs(value-example.features[i])<1e-9));
  const probability=classifier.probabilities(features);
  probability.forEach((value,i)=>maxDifference=Math.max(maxDifference,Math.abs(value-example.probabilities[i])));
  const result=classifier.predict(example.frames,.85);
  rawCorrect+=Number(result.candidate===example.expected);
  gatedCorrect+=Number((result.label??'_')===example.expected);
}
assert.ok(maxDifference<.00002);
assert.equal(rawCorrect/report.test_examples.length,report.raw_classifier_accuracy);
assert.equal(gatedCorrect/report.test_examples.length,report.gated_accuracy_at_threshold_085);
console.log(`Motion candidate: all ${report.test_examples.length} held-out clips match Python features, TF.js probabilities and rejection decisions. Max probability difference: ${maxDifference}.`);
