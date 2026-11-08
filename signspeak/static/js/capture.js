import {toast} from './app.js';
import {CameraController} from './camera.js';

export async function initCapture(){
  let samples=[],lastCapture=0;const get=id=>document.getElementById(id);
  const camera=new CameraController({onState:on=>get('capture-sample').disabled=!on});camera.bind();
  get('capture-form').addEventListener('submit',event=>{event.preventDefault();if(!get('capture-consent').checked)return;const prediction=camera.current();if(!camera.running||!camera.points||!prediction||performance.now()-camera.lastSeen>400)return toast('Bring one clearly visible hand into the camera first.');if(performance.now()-lastCapture<500)return toast('Move your hand slightly before collecting another sample.');lastCapture=performance.now();samples.push({letter:get('capture-letter').value,participant:get('participant').value,points:camera.points.flatMap(p=>[p.x,p.y])});get('capture-count').textContent=`${samples.length} samples collected`;get('download-samples').disabled=false;get('capture-status').textContent=`Captured ${get('capture-letter').value==='_'?'unsupported shape':get('capture-letter').value}. Vary your hand position and lighting between samples.`;});
  get('download-samples').addEventListener('click',()=>{if(!samples.length)return;const blob=new Blob([JSON.stringify({version:1,source:'consented-browser-capture',samples},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='asl-landmark-samples.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);toast('Coordinates downloaded. No camera footage was saved.');});
  get('capture-consent').addEventListener('change',async event=>{if(!event.target.checked&&samples.length){samples=[];get('capture-count').textContent='0 samples collected';get('download-samples').disabled=true;get('capture-status').textContent='In-memory samples were cleared when consent was withdrawn.';}});
}
