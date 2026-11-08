import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import * as tf from '@tensorflow/tfjs';
import {normalize,Stabilizer,LandmarkClassifier,gatePrediction} from '../signspeak/static/js/recognition.js';
import {motionFeatures,MotionClassifier} from '../signspeak/static/js/motion.js';
const model=JSON.parse(fs.readFileSync(new URL('../signspeak/static/models/classifier.json',import.meta.url)));
const report=JSON.parse(fs.readFileSync(new URL('../signspeak/static/models/evaluation.json',import.meta.url)));
const refs=JSON.parse(fs.readFileSync(new URL('../signspeak/static/models/references.json',import.meta.url)));

test('JavaScript and Python normalization agree on a recorded reference',()=>{
 const points=refs.L;const expected=normalize(points);const shifted=points.map((v,i)=>v*3+(i%2?-.7:.3));
 normalize(shifted).forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-10));
 normalize(points.map((v,i)=>i%2?v:-v)).forEach((v,i)=>assert.ok(Math.abs(v-expected[i])<1e-10));
});
test('TensorFlow.js reproduces Python held-out predictions',()=>{
 const classifier=new LandmarkClassifier(model,tf);
 for(const example of report.test_examples){
   const probability=tf.tidy(()=>{
     let x=tf.tensor2d([example.features]).sub(classifier.mean).div(classifier.scale);
     classifier.layers.forEach((layer,i)=>{x=x.matMul(layer.kernel).add(layer.bias);if(i<classifier.layers.length-1)x=x.relu();});
     return Array.from(x.softmax().dataSync());
   });
   probability.forEach((value,i)=>assert.ok(Math.abs(value-example.probabilities[i])<0.00002));
 }
});
test('a held letter commits once and requires release before repetition',()=>{
 const s=new Stabilizer(500);const commits=[];
 for(let now=0;now<2000;now+=100){const r=s.update('A',now);if(r.commit)commits.push(r.commit);}
 assert.deepEqual(commits,['A']);
 for(let now=2000;now<2500;now+=100)s.update(null,now);
 for(let now=2500;now<4000;now+=100){const r=s.update('A',now);if(r.commit)commits.push(r.commit);}
 assert.deepEqual(commits,['A','A']);
});
test('uncertain frames cannot stabilize and unknown geometry is rejected',()=>{
 const s=new Stabilizer(500);
 for(let now=0;now<3000;now+=100)assert.equal(s.update(now%200?'A':'B',now).commit,null);
 const p=model.labels.map(x=>x==='A'?.99:.01/6);
 assert.equal(gatePrediction(p,Array(42).fill(100),model,.85).label,null);
 assert.equal(gatePrediction(model.labels.map(()=>1/7),Array(42).fill(0),model,.85).label,null);
});

test('all 24 static letters are trained and orientation remains meaningful',()=>{
 assert.deepEqual([...model.labels].sort(),[...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'].filter(l=>l!=='J'&&l!=='Z'));
 assert.notDeepEqual(normalize(refs.K),normalize(refs.K.map(v=>-v)));
});
test('real J/Z clips match Python temporal features and TF.js probabilities',()=>{
 const motion=JSON.parse(fs.readFileSync(new URL('../signspeak/static/models/motion-classifier.json',import.meta.url)));
 const evaluation=JSON.parse(fs.readFileSync(new URL('../signspeak/static/models/motion-evaluation.json',import.meta.url)));
 const classifier=new MotionClassifier(motion,tf);
 assert.deepEqual(motion.labels,['J','Z']);
 for(const example of evaluation.test_examples){const features=motionFeatures(example.frames);features.forEach((v,i)=>assert.ok(Math.abs(v-example.features[i])<1e-9));classifier.probabilities(features).forEach((v,i)=>assert.ok(Math.abs(v-example.probabilities[i])<.00002));}
 const points=evaluation.test_examples[0].frames[0].points;
 assert.throws(()=>motionFeatures(Array.from({length:20},(_,i)=>({t:i*100,points}))),/stationary/);
 assert.throws(()=>motionFeatures(evaluation.test_examples[0].frames.slice(0,3)),/eight/);
});
