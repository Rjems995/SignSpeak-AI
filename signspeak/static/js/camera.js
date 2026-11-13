import {config,settings,toast} from './app.js';
import {LandmarkClassifier,Stabilizer,drawHand} from './recognition.js';
import {MotionClassifier} from './motion.js';
import {HandTracker} from './hand-tracker.js';

let resources;
function cameraErrorMessage(error){
  const messages={
    NotAllowedError:'Camera permission was denied. Allow camera access in your browser settings, then select Enable camera again.',
    SecurityError:'This browser has blocked camera access. Open the app over HTTPS or localhost and allow camera permission.',
    NotFoundError:'No camera was found. Connect a camera, then select Enable camera again.',
    NotReadableError:'Your camera is busy or unavailable. Close other apps using it, check the connection and retry.',
    OverconstrainedError:'The selected camera is unavailable. Choose the default camera in Settings, then retry.',
    AbortError:'The camera could not start. Check its connection, close other camera apps and retry.',
    NotSupportedError:'This browser cannot run background hand tracking. Try a current browser with Web Worker and camera frame support.'
  };
  return messages[error.name]||'Hand tracking could not load. Check your internet connection and select Enable camera to retry.';
}
function loadTf(){if(window.tf)return Promise.resolve(window.tf);return new Promise((resolve,reject)=>{const script=document.createElement('script');script.src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';script.crossOrigin='anonymous';script.onload=()=>resolve(window.tf);script.onerror=()=>{script.remove();reject(new Error('TensorFlow.js could not load. Check your connection and retry.'));};document.head.append(script);});}
async function loadResources(){
  if(!resources)resources=(async()=>{
    const tf=await loadTf();
    let classifier=null,motionClassifier=null;
    if(config.model.ready){const response=await fetch('/static/models/classifier.json');if(!response.ok)throw new Error('The trained classifier is missing.');const model=await response.json();await tf.setBackend('cpu');await tf.ready();classifier=new LandmarkClassifier(model,tf);}
    if(config.model.motion?.ready){const response=await fetch('/static/models/motion-classifier.json');if(!response.ok)throw new Error('The movement classifier is missing.');motionClassifier=new MotionClassifier(await response.json(),tf);}
    return {classifier,motionClassifier};
  })().catch(error=>{resources=null;throw error;});
  return resources;
}

export class CameraController {
  constructor({onCommit=()=>{},onResult=()=>{},onState=()=>{},onSequence=()=>{},onSequenceStart=()=>true,onSequenceEnd=()=>{},trackerFactory=()=>new HandTracker(),captureFrame=video=>createImageBitmap(video)}={}){
    this.onCommit=onCommit;this.onResult=onResult;this.onState=onState;this.onSequence=onSequence;this.onSequenceStart=onSequenceStart;this.onSequenceEnd=onSequenceEnd;this.running=false;this.token=0;this.lastResult=null;this.points=null;this.lastSeen=0;this.recording=false;this.lastMotion=null;this.motionDisplayUntil=0;this.observation=null;
    this.stabilizer=new Stabilizer(Number(settings.hold_ms));this.video=document.getElementById('webcam');this.canvas=document.getElementById('landmark-canvas');this.startButton=document.getElementById('camera-start');this.stopButton=document.getElementById('camera-stop');
    this.trackerFactory=trackerFactory;this.captureFrame=captureFrame;this.tracker=null;this.pendingFrame=null;this.sequenceEpoch=0;this.finishingMotion=false;
  }
  bind(){this.startButton.addEventListener('click',()=>this.start());this.stopButton.addEventListener('click',()=>this.stop());document.getElementById('record-motion').addEventListener('click',()=>this.recordMotion());document.getElementById('show-landmarks').checked=settings.landmarks;document.getElementById('show-landmarks').addEventListener('change',e=>settings.landmarks=e.target.checked);window.addEventListener('pagehide',()=>this.stop());document.addEventListener('visibilitychange',()=>{if(document.hidden)this.stop();});}
  status(text,active=false){const el=document.getElementById('camera-status');if(el.textContent.trim()===text)return;el.replaceChildren();const dot=document.createElement('span');dot.className=`status-dot${active?'':' gray'}`;dot.setAttribute('aria-hidden','true');el.append(dot,document.createTextNode(text));}
  clearError(){const el=document.getElementById('camera-error');if(el){el.hidden=true;el.textContent='';}}
  showError(message,status='Camera unavailable'){const el=document.getElementById('camera-error');if(el){el.textContent=message;el.hidden=false;}else toast(message);this.status(status);}
  announce(key,message){const el=document.getElementById('prediction-announcement');if(el&&this.announcementKey!==key){this.announcementKey=key;el.textContent=message;}}
  async start(){
    if(this.running||this.loading)return;
    this.clearError();
    if(typeof Worker!=='function'||typeof createImageBitmap!=='function')return this.showError(cameraErrorMessage({name:'NotSupportedError'}));
    if(!navigator.mediaDevices?.getUserMedia)return this.showError('Camera access needs HTTPS or localhost and a browser with webcam support.');
    const token=++this.token;this.loading=true;this.startButton.disabled=true;this.startButton.setAttribute('aria-busy','true');this.startButton.textContent='Preparing your camera…';this.stopButton.disabled=false;this.status('Requesting permission');
    try{
      const stream=await navigator.mediaDevices.getUserMedia({video:settings.camera?{deviceId:{exact:settings.camera},width:{ideal:640},height:{ideal:480}}:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
      if(token!==this.token){stream.getTracks().forEach(t=>t.stop());return;}
      this.stream=stream;this.status('Loading hand tracking…');
      const tracker=this.trackerFactory();this.tracker=tracker;
      const [loaded]=await Promise.all([loadResources(),tracker.init()]);
      if(token!==this.token)return;
      this.classifier=loaded.classifier;this.motionClassifier=loaded.motionClassifier;
      this.video.srcObject=stream;await this.video.play();if(token!==this.token)return;
      this.canvas.width=this.video.videoWidth;this.canvas.height=this.video.videoHeight;
      document.getElementById('camera-viewport').style.aspectRatio=`${this.video.videoWidth}/${this.video.videoHeight}`;
      document.getElementById('camera-viewport').classList.add('running');document.getElementById('camera-empty').hidden=true;document.getElementById('hand-status').hidden=false;
      this.running=true;this.lastVideoTime=-1;this.lastFrame=0;this.stabilizer.reset();this.status('Camera on',true);document.getElementById('record-motion').disabled=!this.motionClassifier;this.onState(true);if(!this.classifier)toast('Hand tracking is ready. Install a trained classifier to enable letter recognition.');
      stream.getVideoTracks()[0]?.addEventListener('ended',()=>{if(this.running){this.stop();this.showError('Your camera disconnected. Reconnect it, then select Enable camera again.','Camera disconnected');}},{once:true});
      this.frame();
    }catch(error){if(token===this.token){this.stop();this.showError(cameraErrorMessage(error));}}
    finally{if(token===this.token){this.loading=false;this.startButton.disabled=false;this.startButton.removeAttribute('aria-busy');this.startButton.textContent='Enable camera';}}
  }
  async frame(){
    if(!this.running)return;
    this.raf=requestAnimationFrame(()=>this.frame());const now=performance.now();
    const sampleInterval=this.recording?30:90;
    if(this.pendingFrame||this.finishingMotion||this.video.readyState<2||now-this.lastFrame<sampleInterval||this.video.currentTime===this.lastVideoTime)return;
    this.lastFrame=now;this.lastVideoTime=this.video.currentTime;
    const tracker=this.tracker;
    const job={token:this.token,sequenceEpoch:this.sequenceEpoch,recording:this.recording};
    job.done=new Promise(resolve=>job.complete=resolve);this.pendingFrame=job;
    try{
      const bitmap=await this.captureFrame(this.video);
      if(!this.running||job.token!==this.token){bitmap.close();return;}
      // detect owns the bitmap from this point, including failure cleanup.
      const result=await tracker.detect(bitmap,now);
      if(!this.running||job.token!==this.token||job.sequenceEpoch!==this.sequenceEpoch)return;
      this.processResult(result,now);
    }catch(error){
      if(this.running&&job.token===this.token){console.error(error);this.stop();this.showError('Hand tracking stopped unexpectedly. Select Enable camera to retry.','Tracking stopped');}
    }finally{
      if(this.pendingFrame===job)this.pendingFrame=null;
      job.complete();
    }
  }
  processResult(result,now){
    const single=result.landmarks.length===1;
    this.points=single?result.landmarks[0]:null;if(this.points)this.lastSeen=now;
    const hand=(result.handednesses??result.handedness)?.[0]?.[0];
    this.observation=single?{observed_hand:hand?.categoryName?.toLowerCase()??null,observed_hand_score:hand?.score??null,frame_width:this.video.videoWidth,frame_height:this.video.videoHeight,input_mirrored:false,preview_mirrored:true,landmarker_version:'0.10.21'}:null;
    drawHand(this.canvas,settings.landmarks?this.points:null);
    if(this.recording){
      if(this.points)this.motionFrames.push({t:now-this.motionStart,points:this.points.flatMap(p=>[p.x,p.y])});
      document.getElementById('motion-status').textContent=`Recording movement… ${Math.max(0,(this.motionDuration-(now-this.motionStart))/1000).toFixed(1)} s remaining. Trace J or Z once.`;
      this.lastResult=null;this.onResult(null,this.points);return;
    }
    if(now<this.motionDisplayUntil)return;
    document.getElementById('hand-status').textContent=result.landmarks.length>1?'Use one hand at a time':single?'Hand detected · hold your letter steady':'Bring one hand into view';
    let prediction={label:null,confidence:0,candidate:null};
    if(this.points&&this.classifier)prediction=this.classifier.predict(this.points,Number(settings.threshold),result.worldLandmarks?.[0]);
    const stable=this.stabilizer.update(prediction.label,now);this.lastResult={...prediction,stable:stable.stable,progress:stable.progress,timestamp:now};
    this.renderPrediction(single,prediction,stable);this.onResult(this.lastResult,this.points);
    if(stable.commit)this.onCommit(stable.commit);
  }
  renderPrediction(hasHand,prediction,stable){
    const letter=document.getElementById('prediction-letter');if(letter)letter.textContent=prediction.label||'—';
    const label=document.getElementById('prediction-label');if(label)label.textContent=prediction.label?`ASL letter ${prediction.label}`:!this.running?'Camera off':hasHand?'Sign not recognized.':'Looking for your hand';
    const hint=document.getElementById('prediction-hint');if(hint)hint.textContent=prediction.motion?'Movement accepted. Select Record J / Z to trace another.':prediction.label?(stable.progress>=1?'Letter held. Release briefly to repeat.':'Hold steady to confirm this letter.'):!this.running?'Enable your camera to continue.':hasHand?'Hold an A–Z static shape, or record movement for J/Z.':'Keep your whole hand inside the camera.';
    const confidence=document.getElementById('confidence-value');if(confidence)confidence.textContent=prediction.label?`${Math.round(prediction.confidence*100)}%`:'—';
    const confidenceFill=document.getElementById('confidence-fill');if(confidenceFill)confidenceFill.style.width=`${prediction.label?prediction.confidence*100:0}%`;
    const hold=document.getElementById('hold-fill');if(hold)hold.style.width=`${stable.progress*100}%`;
    if(prediction.label){
      const confirmed=stable.progress>=1;
      this.announce(`${prediction.motion?'motion':'letter'}:${prediction.label}:${confirmed?'confirmed':'holding'}`,
        prediction.motion?`Movement ${prediction.label} accepted.`:confirmed?`Letter ${prediction.label} confirmed. Release briefly to repeat.`:`Letter ${prediction.label} detected. Hold steady.`);
    }else this.announce(this.running?(hasHand?'unrecognized':'no-hand'):'camera-off',this.running?(hasHand?'Sign not recognized.':'No hand detected. Keep your whole hand inside the camera.'):'Camera off.');
  }
  recordMotion(){
    if(!this.running||!this.motionClassifier)return toast('Enable the camera and movement model first.');
    if(this.recording)return this.finishMotion();
    if(!this.points||performance.now()-this.lastSeen>400)return toast('Bring one clearly visible hand into view before recording.');
    if(this.onSequenceStart()===false)return false;
    this.sequenceEpoch++;this.recording=true;this.finishingMotion=false;this.lastMotion=null;this.motionFrames=[];this.motionStart=performance.now();this.motionDuration=Math.max(4500,Number(config.model.motion?.recording_ms)||0);this.stabilizer.reset();this.lastResult=null;
    this.announce('recording','Recording J or Z. Trace once, then select Finish movement.');
    document.getElementById('record-motion').textContent='Finish movement';document.getElementById('motion-status').textContent='Recording now. Trace J or Z once, then select Finish movement. Recording ends automatically after 4.5 seconds.';
    this.motionTimer=setTimeout(()=>this.finishMotion(),this.motionDuration);
    return true;
  }
  finishMotion(){
    if(!this.recording||this.finishingMotion)return;
    clearTimeout(this.motionTimer);
    const job=this.pendingFrame;
    if(job?.recording&&job.sequenceEpoch===this.sequenceEpoch){
      const token=this.token,epoch=this.sequenceEpoch;
      this.finishingMotion=true;document.getElementById('record-motion').disabled=true;document.getElementById('record-motion').textContent='Finishing movement…';
      // Include the final frame captured before Finish, without admitting a
      // frame from before recording started or from another camera session.
      return job.done.then(()=>{
        if(this.running&&this.recording&&token===this.token&&epoch===this.sequenceEpoch)this.completeMotion();
      });
    }
    this.completeMotion();
  }
  completeMotion(){
    this.sequenceEpoch++;this.finishingMotion=false;
    clearTimeout(this.motionTimer);this.recording=false;if(!this.running)return;document.getElementById('record-motion').disabled=false;document.getElementById('record-motion').textContent='Record J / Z';
    const frames=this.motionFrames;this.motionFrames=[];
    try{
      const prediction=this.motionClassifier.predict(frames,Number(settings.threshold));this.onSequence(frames);this.stabilizer.reset();
      if(!prediction.label){this.lastMotion=null;this.lastResult={...prediction,stable:null,progress:0,timestamp:performance.now()};this.renderPrediction(true,prediction,{progress:0});this.onResult(this.lastResult,this.points);document.getElementById('motion-status').textContent='Sign not recognized. Review the J/Z lesson and trace once more.';return;}
      const now=performance.now();this.lastMotion={...prediction,motion:true,stable:prediction.label,progress:1,timestamp:now};this.motionDisplayUntil=now+1600;
      this.renderPrediction(true,this.lastMotion,{progress:1});this.onResult(this.lastMotion,this.points);this.onCommit(prediction.label);
      document.getElementById('motion-status').textContent=`Movement model accepted ${prediction.label}. This experimental result may need correction.`;
    }catch(error){this.lastMotion=null;this.lastResult={label:null,confidence:0,stable:null,progress:0,timestamp:performance.now()};this.stabilizer.reset();this.renderPrediction(Boolean(this.points),this.lastResult,{progress:0});this.onResult(this.lastResult,this.points);document.getElementById('motion-status').textContent=`Sign not recognized. ${error.message}`;}finally{this.onSequenceEnd({cancelled:false});}
  }
  clearMotion(){
    this.sequenceEpoch++;
    if(this.recording){
      clearTimeout(this.motionTimer);this.recording=false;this.finishingMotion=false;this.motionFrames=[];
      document.getElementById('record-motion').disabled=!this.running||!this.motionClassifier;
      document.getElementById('record-motion').textContent='Record J / Z';
      document.getElementById('motion-status').textContent='Movement cancelled. Select Record J / Z to start again.';
      this.onSequenceEnd({cancelled:true});
    }
    this.lastMotion=null;this.motionDisplayUntil=0;this.stabilizer.reset();
  }
  current(){if(this.running&&this.lastMotion&&performance.now()-this.lastMotion.timestamp<8000)return this.lastMotion;return this.running&&this.lastResult&&performance.now()-this.lastResult.timestamp<400?this.lastResult:null;}
  stop(){
    ++this.token;this.sequenceEpoch++;this.finishingMotion=false;
    this.tracker?.close();this.tracker=null;this.pendingFrame=null;
    clearTimeout(this.motionTimer);
    const wasRecording=this.recording;this.recording=false;this.running=false;this.loading=false;
    if(wasRecording)this.onSequenceEnd({cancelled:true});
    this.lastMotion=null;this.motionFrames=[];this.motionDisplayUntil=0;
    cancelAnimationFrame(this.raf);
    this.stream?.getTracks().forEach(track=>track.stop());this.stream=null;
    this.video.pause();this.video.srcObject=null;
    this.points=null;this.observation=null;this.lastResult=null;this.stabilizer.reset();
    drawHand(this.canvas,null);
    document.getElementById('camera-viewport').classList.remove('running');
    document.getElementById('camera-viewport').style.removeProperty('aspect-ratio');
    document.getElementById('camera-empty').hidden=false;document.getElementById('hand-status').hidden=true;
    document.getElementById('record-motion').disabled=true;document.getElementById('record-motion').textContent='Record J / Z';
    document.getElementById('motion-status').textContent='Hold static letters steady. For J or Z, record one movement and select Finish movement. You have up to 4.5 seconds.';
    this.startButton.disabled=false;this.startButton.removeAttribute('aria-busy');this.startButton.textContent='Enable camera';this.stopButton.disabled=true;
    this.status('Camera off');this.renderPrediction(false,{label:null,confidence:0},{progress:0});
    this.onState(false);this.onResult(null,null);
  }

}
