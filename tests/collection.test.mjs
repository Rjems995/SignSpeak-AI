import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {SampleCollection,CONSENT_VERSION,checkGeometry} from '../signspeak/static/js/collection.js';
const refs=JSON.parse(fs.readFileSync(new URL('../signspeak/static/models/references.json',import.meta.url)));
const plan=JSON.parse(fs.readFileSync(new URL('../signspeak/static/data/collection-plan.json',import.meta.url)));
const context={participant:'signer_01',session:'visit_01',letter:'L',hand:'right',lighting:'soft',angle:'front',label_review:'collector_confirmed',consent_version:CONSENT_VERSION};

test('real coordinate sample retains the original context and observation',()=>{
  const store=new SampleCollection(),metadata={...context};
  const sample=store.capture(metadata,{points:refs.L},{observed_hand:'right'});
  metadata.letter='Q';metadata.hand='left';
  assert.equal(sample.letter,'L');assert.equal(sample.hand,'right');assert.equal(sample.participant,'signer_01');
  assert.equal(sample.observation.observed_hand,'right');assert.equal(store.coverage().L.total,1);
  assert.equal(store.export('static').samples.length,1);assert.equal(store.export('motion').samples.length,0);
});
test('same sample is not counted twice and conflicting labels are rejected',()=>{
  const store=new SampleCollection();store.capture(context,{points:refs.L});
  assert.throws(()=>store.capture(context,{points:refs.L}),/already collected/);
  assert.throws(()=>store.capture({...context,letter:'Q'},{points:refs.L}),/labelled L/);
  assert.equal(store.samples.length,1);
  store.undo();assert.equal(store.samples.length,0);store.capture(context,{points:refs.L});assert.equal(store.samples.length,1);
});
test('missing consent, metadata and incomplete hand geometry cannot inflate coverage',()=>{
  const store=new SampleCollection();
  assert.throws(()=>store.capture({...context,consent_version:null},{points:refs.L}),/Consent/);
  assert.throws(()=>store.capture({...context,hand:''},{points:refs.L}),/actual hand/);
  assert.throws(()=>store.capture({...context,participant:'someone@example.com'},{points:refs.L}),/participant/);
  const clipped=[...refs.L];clipped[8]=1.2;assert.throws(()=>checkGeometry(clipped),/inside/);
  assert.equal(store.samples.length,0);
});
test('imports preserve original signers and skip duplicates without partial corrupt imports',()=>{
  const original=new SampleCollection();original.capture(context,{points:refs.L});
  const imported=new SampleCollection();assert.deepEqual(imported.merge(original.export('static')),{added:1,duplicates:0});
  assert.deepEqual(imported.merge(original.export('static')),{added:0,duplicates:1});
  const broken=original.export('static');broken.samples=[{...broken.samples[0],participant:'another'},{...broken.samples[0],hand:''}];
  assert.throws(()=>imported.merge(broken),/actual hand/);assert.equal(imported.samples.length,1);
  imported.clear();assert.equal(imported.coverage().L.total,0);
});
test('planner prioritizes Q and then balances missing cells for the selected signer/hand',()=>{
  const store=new SampleCollection();assert.equal(store.nextLetter('signer_01','right',plan),'Q');
  store.capture({...context,letter:'Q'},{points:refs.Q});
  assert.equal(store.nextLetter('signer_01','right',plan),'E');
  assert.equal(store.nextLetter('signer_01','left',plan),'Q');
});
