import {config,settings,toast} from './app.js';
import {LandmarkClassifier,Stabilizer,drawHand} from './recognition.js';
import {MotionClassifier} from './motion.js';

const MP_BASE='https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21';
const LANDMARK_MODEL='https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
let resources;
function loadTf(){if(window.tf)return Promise.resolve(window.tf);return new Promise((resolve,reject)=>{const script=document.createElement('script');script.src='https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js';script.crossOrigin='anonymous';script.onload=()=>resolve(window.tf);script.onerror=()=>{script.remove();reject(new Error('TensorFlow.js could not load. Check your connection and retry.'));};document.head.append(script);});}
async function loadResources(){
  if(!resources)resources=(async()=>{
    const [vision,tf]=await Promise.all([import(`${MP_BASE}/vision_bundle.mjs`),loadTf()]);
    const fileset=await vision.FilesetResolver.forVisionTasks(`${MP_BASE}/wasm`);
    // CPU is supported on a wider range of devices than a required GPU delegate.
    const landmarker=await vision.HandLandmarker.createFromOptions(fileset,{baseOptions:{modelAssetPath:LANDMARK_MODEL,delegate:'CPU'},runningMode:'VIDEO',numHands:2,minHandDetectionConfidence:0.6,minHandPresenceConfidence:0.6,minTrackingConfidence:0.6});
    let classifier=null,motionClassifier=null;
    if(config.model.ready){try{const response=await fetch('/static/models/classifier.json');if(!response.ok)throw new Error('The trained classifier is missing.');const model=await response.json();await tf.setBackend('cpu');await tf.ready();classifier=new LandmarkClassifier(model,tf);}catch(error){landmarker.close();throw error;}}
    if(config.model.motion?.ready){try{const response=await fetch('/static/models/motion-classifier.json');if(!response.ok)throw new Error('The movement classifier is missing.');motionClassifier=new MotionClassifier(await response.json(),tf);}catch(error){landmarker.close();throw error;}}
    return {landmarker,classifier,motionClassifier};
  })().catch(error=>{resources=null;throw error;});
  return resources;
}

export class CameraController {
  constructor({onCommit=()=>{},onResult=()=>{},onState=()=>{},onSequence=()=>{},onSequenceStart=()=>true,onSequenceEnd=()=>{}}={}){
    this.onCommit=onCommit;this.onResult=onResult;this.onState=onState;this.onSequence=onSequence;this.onSequenceStart=onSequenceStart;this.onSequenceEnd=onSequenceEnd;this.running=false;this.token=0;this.lastResult=null;this.points=null;this.lastSeen=0;this.recording=false;this.lastMotion=null;this.motionDisplayUntil=0;this.observation=null;
    this.stabilizer=new Stabilizer(Number(settings.hold_ms));this.video=document.getElementById('webcam');this.canvas=document.getElementById('landmark-canvas');this.startButton=document.getElementById('camera-start');this.stopButton=document.getElementById('camera-stop');
  }
  bind(){this.startButton.addEventListener('click',()=>this.start());this.stopButton.addEventListener('click',()=>this.stop());document.getElementById('record-motion').addEventListener('click',()=>this.recordMotion());document.getElementById('show-landmarks').checked=settings.landmarks;document.getElementById('show-landmarks').addEventListener('change',e=>settings.landmarks=e.target.checked);window.addEventListener('pagehide',()=>this.stop());document.addEventListener('visibilitychange',()=>{if(document.hidden)this.stop();});}
  status(text,active=false){const el=document.getElementById('camera-status');el.replaceChildren();const dot=document.createElement('span');dot.className=`status-dot${active?'':' gray'}`;el.append(dot,document.createTextNode(text));}
  async start(){
    if(this.running||this.loading)return;
    if(!navigator.mediaDevices?.getUserMedia)return toast('Camera access needs HTTPS or localhost and a browser with webcam support.');
    const token=++this.token;this.loading=true;this.startButton.disabled=true;this.startButton.textContent='Preparing your camera…';this.stopButton.disabled=false;this.status('Requesting permission');
    try{
      const stream=await navigator.mediaDevices.getUserMedia({video:settings.camera?{deviceId:{exact:settings.camera},width:{ideal:640},height:{ideal:480}}:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
      if(token!==this.token){stream.getTracks().forEach(t=>t.stop());return;}
      this.stream=stream;this.status('Loading hand tracking…');const loaded=await loadResources();
      if(token!==this.token)return;
      this.landmarker=loaded.landmarker;this.classifier=loaded.classifier;this.motionClassifier=loaded.motionClassifier;
      this.video.srcObject=stream;await this.video.play();if(token!==this.token)return;
      this.canvas.width=this.video.videoWidth;this.canvas.height=this.video.videoHeight;
      document.getElementById('camera-viewport').style.aspectRatio=`${this.video.videoWidth}/${this.video.videoHeight}`;
      document.getElementById('camera-viewport').classList.add('running');document.getElementById('camera-empty').hidden=true;document.getElementById('hand-status').hidden=false;
      this.running=true;this.lastVideoTime=-1;this.lastFrame=0;this.stabilizer.reset();this.status('Camera on',true);document.getElementById('record-motion').disabled=!this.motionClassifier;this.onState(true);if(!this.classifier)toast('Hand tracking is ready. Install a trained classifier to enable letter recognition.');
      stream.getVideoTracks()[0]?.addEventListener('ended',()=>{if(this.running){this.stop();toast('Your camera disconnected. Reconnect it and enable the camera again.');}},{once:true});
      this.frame();
    }catch(error){if(token===this.token){this.stop();const messages={NotAllowedError:'Camera permission was denied. Allow camera access in your browser and try again.',NotFoundError:'No camera was found. Connect a camera and try again.',NotReadableError:'Your camera is busy. Close other apps using it and retry.',OverconstrainedError:'The selected camera is unavailable. Choose the default camera in Settings.'};toast(messages[error.name]||`${error.message||'Hand tracking could not load.'} Check your connection and try again.`);}}
    finally{if(token===this.token){this.loading=false;this.startButton.disabled=false;this.startButton.textContent='Enable camera';}}
  }
  frame(){
    if(!this.running)return;
    this.raf=requestAnimationFrame(()=>this.frame());const now=performance.now();
    if(this.video.readyState<2||now-this.lastFrame<90||this.video.currentTime===this.lastVideoTime)return;
    this.lastFrame=now;this.lastVideoTime=this.video.currentTime;
    try{
      const result=this.landmarker.detectForVideo(this.video,now);const single=result.landmarks.length===1;
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
      if(this.points&&this.classifier)prediction=this.classifier.predict(this.points,Number(settings.threshold));
      const stable=this.stabilizer.update(prediction.label,now);this.lastResult={...prediction,stable:stable.stable,progress:stable.progress,timestamp:now};
      this.renderPrediction(single,prediction,stable);this.onResult(this.lastResult,this.points);
      if(stable.commit)this.onCommit(stable.commit);
    }catch(error){console.error(error);this.stop();toast('Hand tracking stopped unexpectedly. Enable the camera to retry.');}
  }
  renderPrediction(hasHand,prediction,stable){
    const letter=document.getElementById('prediction-letter');if(letter)letter.textContent=prediction.label||'—';
    const label=document.getElementById('prediction-label');if(label)label.textContent=prediction.label?`ASL letter ${prediction.label}`:hasHand?'No supported letter accepted':'Looking for your hand';
    const hint=document.getElementById('prediction-hint');if(hint)hint.textContent=prediction.motion?'Movement accepted. Select Record J / Z to trace another.':prediction.label?(stable.progress>=1?'Letter held. Release briefly to repeat.':'Hold steady to confirm this letter.'):hasHand?'Hold an A–Z static shape, or record movement for J/Z.':'Keep your whole hand inside the camera.';
    const confidence=document.getElementById('confidence-value');if(confidence)confidence.textContent=hasHand&&this.classifier?`${Math.round(prediction.confidence*100)}%`:'—';
    const confidenceFill=document.getElementById('confidence-fill');if(confidenceFill)confidenceFill.style.width=`${hasHand?prediction.confidence*100:0}%`;
    const hold=document.getElementById('hold-fill');if(hold)hold.style.width=`${stable.progress*100}%`;
  }
  recordMotion(){
    if(!this.running||!this.motionClassifier)return toast('Enable the camera and movement model first.');
    if(this.recording)return;
    if(!this.points||performance.now()-this.lastSeen>400)return toast('Bring one clearly visible hand into view before recording.');
    if(this.onSequenceStart()===false)return false;
    this.recording=true;this.lastMotion=null;this.motionFrames=[];this.motionStart=performance.now();this.motionDuration=config.model.motion?.recording_ms||2400;this.stabilizer.reset();this.lastResult=null;
    document.getElementById('record-motion').disabled=true;document.getElementById('motion-status').textContent='Recording now. Trace J or Z once with one hand.';
    this.motionTimer=setTimeout(()=>this.finishMotion(),this.motionDuration);
    return true;
  }
  finishMotion(){
    this.recording=false;if(!this.running)return;document.getElementById('record-motion').disabled=false;
    const frames=this.motionFrames;this.motionFrames=[];
    try{
      const prediction=this.motionClassifier.predict(frames,Number(settings.threshold));this.onSequence(frames);this.stabilizer.reset();
      if(!prediction.label){document.getElementById('motion-status').textContent='No J/Z movement accepted. Review the lesson and trace once more.';return;}
      const now=performance.now();this.lastMotion={...prediction,motion:true,stable:prediction.label,progress:1,timestamp:now};this.motionDisplayUntil=now+1600;
      this.renderPrediction(true,this.lastMotion,{progress:1});this.onResult(this.lastMotion,this.points);this.onCommit(prediction.label);
      document.getElementById('motion-status').textContent=`Movement model accepted ${prediction.label}. This experimental result may need correction.`;
    }catch(error){document.getElementById('motion-status').textContent=error.message;}finally{this.onSequenceEnd({cancelled:false});}
  }
  clearMotion(){this.lastMotion=null;this.motionDisplayUntil=0;this.stabilizer.reset();}
  current(){if(this.running&&this.lastMotion&&performance.now()-this.lastMotion.timestamp<8000)return this.lastMotion;return this.running&&this.lastResult&&performance.now()-this.lastResult.timestamp<400?this.lastResult:null;}
  stop(){++this.token;clearTimeout(this.motionTimer);const wasRecording=this.recording;this.recording=false;if(wasRecording)this.onSequenceEnd({cancelled:true});this.lastMotion=null;this.motionFrames=[];this.motionDisplayUntil=0;document.getElementById('record-motion').disabled=true;document.getElementById('motion-status').textContent='Hold static letters steady. For J or Z, select Record J / Z and trace once in 2.4 seconds.';this.loading=false;this.running=false;cancelAnimationFrame(this.raf);this.stream?.getTracks().forEach(t=>t.stop());this.stream=null;this.video.pause();this.video.srcObject=null;this.points=null;this.observation=null;this.lastResult=null;this.stabilizer.reset();drawHand(this.canvas,null);document.getElementById('camera-viewport').classList.remove('running');document.getElementById('camera-empty').hidden=false;document.getElementById('hand-status').hidden=true;this.startButton.disabled=false;this.startButton.textContent='Enable camera';this.stopButton.disabled=true;this.status('Camera off');this.renderPrediction(false,{label:null,confidence:0},{progress:0});this.onState(false);this.onResult(null,null);}
}
