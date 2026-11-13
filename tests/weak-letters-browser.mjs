// Inject the unreleased candidate only into this fake-camera QA session.
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import {denseProbabilities,gatePrediction} from '../signspeak/static/js/recognition.js';
const base=process.env.TEST_BASE_URL||'http://127.0.0.1:5016';
const directory='datasets/candidates/focus-topology-v5';
const [body,evaluation,positive,negative]=await Promise.all([
 fs.readFile(directory+'/classifier.json'),fs.readFile(directory+'/evaluation.json','utf8').then(JSON.parse),
 fs.readFile('datasets/focus-depth.json','utf8').then(JSON.parse),fs.readFile('datasets/unknown-depth.json','utf8').then(JSON.parse)
]);
const byFingerprint=new Map(positive.samples.map(row=>[row.fingerprint,row]));
const candidate=JSON.parse(body.toString());
const fixtures=[...'EMRT'].map(letter=>{
 const i=evaluation.test_examples.findIndex(row=>row.expected===letter&&row.accepted===letter&&
   (letter!=='R'||gatePrediction(denseProbabilities(candidate,row.features),row.features,candidate,.85).expert==='finger_topology'));
 assert.ok(i>=0);return {...byFingerprint.get(evaluation.split_fingerprints.test[i]),expected:letter};
});
const unknown=['relaxed_hand','partial_flexion','unrelated_gesture'].map(category=>negative.samples.find(row=>row.split==='test'&&row.category===category));
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
try{
 const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
 page.on('pageerror',error=>errors.push(error.message));
 await page.route('**/static/models/classifier.json',route=>route.fulfill({body,contentType:'application/json'}));
 await page.goto(base+'/translate',{waitUntil:'networkidle'});
 await page.click('#camera-start');
 await page.waitForFunction(()=>document.getElementById('camera-status').textContent.includes('Camera on'),undefined,{timeout:90000});
 await page.click('#camera-stop');
 const result=await page.evaluate(async({fixtures,unknown})=>{
   const {CameraController}=await import('/static/js/camera.js');
   const {LandmarkClassifier,Stabilizer}=await import('/static/js/recognition.js');
   const model=await fetch('/static/models/classifier.json').then(r=>r.json());
   const commits=[],experts=[],camera=new CameraController({captureFrame:async()=>({close(){}}),onCommit:letter=>{commits.push(letter);document.getElementById('translation-text').value+=letter;}});
   camera.running=true;camera.classifier=new LandmarkClassifier(model,window.tf);camera.stabilizer=new Stabilizer(30);
   camera.video={readyState:2,currentTime:1};let row=fixtures[0],world=true;
   const unpack=points=>Array.from({length:21},(_,i)=>({x:points[i*3],y:points[i*3+1],z:points[i*3+2]}));
   camera.tracker={detect:async()=>({landmarks:[unpack(row.points)],worldLandmarks:world?[unpack(row.world_points)]:[]}),close(){}};
   const tick=async()=>{camera.lastFrame=-Infinity;camera.video.currentTime++;await camera.frame();cancelAnimationFrame(camera.raf);};
   for(const fixture of fixtures){row=fixture;camera.stabilizer.reset();for(let i=0;i<8;i++){await tick();await new Promise(resolve=>setTimeout(resolve,40));}experts.push(camera.lastResult.expert||null);}
   let rejected=0;
   for(const fixture of unknown){
     row=fixture;camera.stabilizer.reset();
     for(let i=0;i<8;i++){await tick();await new Promise(resolve=>setTimeout(resolve,40));}
     if(document.getElementById('prediction-label').textContent!=='Sign not recognized.')throw Error('Unknown pose accepted');
     if(document.getElementById('confidence-value').textContent!=='—')throw Error('Unknown pose shows confidence');
     rejected++;
   }
   world=false;await tick();const missingWorldRejected=camera.lastResult.label===null&&camera.running;
   camera.running=false;cancelAnimationFrame(camera.raf);
   return {commits,experts,text:document.getElementById('translation-text').value,rejected,missingWorldRejected};
 },{fixtures,unknown});
 assert.deepEqual(result.commits,[...'EMRT']);assert.equal(result.text,'EMRT');assert.equal(result.rejected,3);assert.equal(result.missingWorldRejected,true);
 assert.equal(result.experts[2],'finger_topology');
 await page.setViewportSize({width:390,height:844});assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth));
 assert.deepEqual(errors,[]);
 console.log('Unreleased candidate browser checks passed: real E/M/T fixtures and R crossing expert, world-coordinate forwarding, no unknown commits, missing-depth rejection and mobile layout. Production model was not replaced.');
}finally{await browser.close();}
