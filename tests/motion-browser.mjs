// Timing/lifecycle integration uses recorded public reference coordinates and a
// fake browser camera. These fixtures never enter collected/training datasets.
import {chromium} from 'playwright';
import assert from 'node:assert/strict';
const base=process.env.TEST_BASE_URL||'http://127.0.0.1:5013';
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'C:/Program Files/Google/Chrome/Application/chrome.exe',args:['--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream']});
try{
  const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
  page.on('pageerror',error=>errors.push(error.message));
  await page.goto(base+'/translate',{waitUntil:'networkidle'});
  await page.click('#camera-start');
  await page.waitForFunction(()=>document.getElementById('camera-status').textContent.includes('Camera on'),undefined,{timeout:90000});
  await page.click('#camera-stop');
  const result=await page.evaluate(async()=>{
    const {CameraController}=await import('/static/js/camera.js');
    const {MotionClassifier}=await import('/static/js/motion.js');
    const [model,references]=await Promise.all(['/static/models/motion-classifier.json','/static/models/motion-references.json'].map(url=>fetch(url).then(r=>r.json())));
    let commits=[],sequences=0,ended=[];
    const controller=new CameraController({onCommit:letter=>commits.push(letter),onSequence:()=>sequences++,onSequenceEnd:value=>ended.push(value.cancelled)});
    controller.running=true;controller.lastSeen=performance.now();
    controller.points=Array.from({length:21},(_,i)=>({x:references.J[0].points[i*2],y:references.J[0].points[i*2+1]}));
    controller.motionClassifier=new MotionClassifier(model,window.tf);
    controller.recordMotion();
    const duration=controller.motionDuration,startedText=document.getElementById('record-motion').textContent;
    controller.motionFrames=references.J;
    controller.recordMotion(); // Same button finishes early.
    controller.finishMotion(); // A stale timer/click must not commit twice.
    const finishedText=document.getElementById('record-motion').textContent;
    controller.lastSeen=performance.now();controller.recordMotion();controller.stop();
    const cancelledText=document.getElementById('record-motion').textContent;
    const sample=recording=>{
      let calls=0;
      const camera=new CameraController();
      camera.running=true;camera.recording=recording;camera.motionStart=performance.now();camera.motionDuration=4500;
      camera.video={readyState:2,currentTime:1};camera.lastVideoTime=-1;camera.lastFrame=performance.now()-50;
      camera.landmarker={detectForVideo:()=>{calls++;return {landmarks:[]};}};
      camera.frame();cancelAnimationFrame(camera.raf);camera.running=false;
      return calls;
    };
    return {duration,startedText,finishedText,cancelledText,commits,sequences,ended,recordingCalls:sample(true),staticCalls:sample(false)};
  });
  assert.equal(result.duration,4500);
  assert.equal(result.startedText,'Finish movement');
  assert.equal(result.finishedText,'Record J / Z');
  assert.equal(result.cancelledText,'Record J / Z');
  assert.deepEqual(result.commits,['J']);assert.equal(result.sequences,1);
  assert.deepEqual(result.ended,[false,true]);
  assert.equal(result.recordingCalls,1);assert.equal(result.staticCalls,0);
  assert.deepEqual(errors,[]);
  console.log('Motion browser checks passed: faster sampling, 4.5-second limit, early finish, exactly-once commit and cancellation without partial samples.');
}finally{await browser.close();}
