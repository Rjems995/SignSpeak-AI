// Pure collection rules; no camera, storage, or network side effects.
import {normalize} from './recognition.js';
import {motionFeatures} from './motion.js';
export const CONSENT_VERSION='coordinates-v3';
const alphabet=[...'ABCDEFGHIJKLMNOPQRSTUVWXYZ'];
const idPattern=/^[A-Za-z0-9_-]{1,40}$/;
const roundKey=values=>values.map(v=>Number(v.toFixed(5))).join(',');

export function validateContext(context){
  if(!idPattern.test(context.participant??''))throw new Error('Use a participant ID with 1–40 letters, numbers, underscores, or hyphens.');
  if(!idPattern.test(context.session??''))throw new Error('Enter a session ID. Use the same participant ID across sessions.');
  if(![...alphabet,'_'].includes(context.letter))throw new Error('Choose a supported letter.');
  if(!['left','right'].includes(context.hand))throw new Error('Choose the actual hand being used.');
  if(!['bright','soft','dim'].includes(context.lighting))throw new Error('Label the current lighting condition.');
  if(!['front','slight_left','slight_right'].includes(context.angle))throw new Error('Label the current viewing angle.');
  if(context.consent_version!==CONSENT_VERSION)throw new Error('Consent to collecting coordinates before capturing samples.');
  if(context.label_review!=='collector_confirmed')throw new Error('Check the intended sign against the ASL reference before capturing.');
  return {...context};
}

export function checkGeometry(points){
  if(!Array.isArray(points)||points.length!==42||points.some(v=>!Number.isFinite(v)))throw new Error('Wait for one clearly tracked hand.');
  if(points.some(v=>v<.01||v>.99))throw new Error('Keep your whole hand and fingertips inside the camera frame.');
  if(Math.hypot(points[18]-points[0],points[19]-points[1])<.045)throw new Error('Move your hand closer so the landmarks are clearer.');
  normalize(points);
}

function sampleKey(sample){
  const vector=sample.frames?motionFeatures(sample.frames):normalize(sample.points);
  return `${sample.participant}|${sample.hand}|${sample.frames?'motion':'static'}|${roundKey(vector)}`;
}

export class SampleCollection{
  constructor(){this.samples=[];this.keys=new Map();}
  add(sample){
    validateContext(sample);
    if(!sample.sample_id||typeof sample.sample_id!=='string')throw new Error('Sample ID is missing.');
    if(sample.frames){if(!['J','Z'].includes(sample.letter))throw new Error('Only J and Z use sequence samples.');sample.frames.forEach(frame=>checkGeometry(frame.points));motionFeatures(sample.frames);}
    else {if(['J','Z'].includes(sample.letter))throw new Error('J and Z require movement, not a still sample.');checkGeometry(sample.points);}
    const key=sampleKey(sample),existing=this.keys.get(key);
    if(existing)throw new Error(existing.letter===sample.letter?'That coordinate sample is already collected. Change the viewing position before capturing again.':`These same coordinates were labelled ${existing.letter}. Review the intended sign before collecting another label.`);
    this.keys.set(key,sample);this.samples.push(sample);return sample;
  }
  capture(context,data,observation={}){
    const checked=validateContext(context);
    return this.add({...checked,...data,sample_id:crypto.randomUUID(),captured_at:new Date().toISOString(),source:'consented-browser-capture',observation:{...observation}});
  }
  clear(){this.samples=[];this.keys.clear();}
  undo(){const sample=this.samples.pop();if(sample)this.keys.delete(sampleKey(sample));return sample;}
  merge(document){
    if(document?.version!==3||!Array.isArray(document.samples))throw new Error('Import a version 3 collection export. Audit older datasets with the Python tools.');
    const trial=new SampleCollection();trial.samples=[...this.samples];trial.keys=new Map(this.keys);let added=0,duplicates=0;
    for(const sample of document.samples){
      validateContext(sample);
      const existing=trial.keys.get(sampleKey(sample));
      if(existing?.letter===sample.letter){duplicates++;continue;}
      trial.add(structuredClone(sample));added++;
    }
    this.samples=trial.samples;this.keys=trial.keys;return {added,duplicates};
  }
  export(kind){return {version:3,source:'consented-browser-capture',exported_at:new Date().toISOString(),samples:this.samples.filter(s=>kind==='motion'?Boolean(s.frames):!s.frames)};}
  coverage(){
    return Object.fromEntries(alphabet.map(letter=>{
      const rows=this.samples.filter(s=>s.letter===letter);
      return [letter,{total:rows.length,participants:new Set(rows.map(s=>s.participant)).size,left:rows.filter(s=>s.hand==='left').length,right:rows.filter(s=>s.hand==='right').length,lighting:new Set(rows.map(s=>s.lighting)).size,angles:new Set(rows.map(s=>s.angle)).size}];
    }));
  }
  nextLetter(participant,hand,plan){
    const counts=Object.fromEntries(alphabet.map(letter=>[letter,this.samples.filter(s=>s.participant===participant&&s.hand===hand&&s.letter===letter).length]));
    const priorities=[...plan.priority_letters,...alphabet.filter(l=>!plan.priority_letters.includes(l))];
    const base=plan.samples_per_participant_letter_hand;
    const needed=priorities.filter(letter=>counts[letter]<base).sort((a,b)=>counts[a]-counts[b]);
    if(needed.length)return needed[0];
    return plan.priority_letters.find(letter=>counts[letter]<base+plan.priority_extra_samples)??null;
  }
}
