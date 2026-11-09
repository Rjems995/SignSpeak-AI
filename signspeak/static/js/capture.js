import {toast} from './app.js';
import {CameraController} from './camera.js';
import {SampleCollection,CONSENT_VERSION,checkGeometry} from './collection.js';

export async function initCapture(){
  const get=id=>document.getElementById(id);
  const plan=JSON.parse(get('collection-plan').textContent),store=new SampleCollection();
  const fields=['participant','capture-session','capture-letter','capture-hand','capture-lighting','capture-angle','capture-verified'];
  let pending=null,burstToken=0,burstActive=false,lastCapture=-Infinity;
  get('capture-session').value=`session_${crypto.randomUUID().slice(0,8)}`;
  const status=message=>get('capture-status').textContent=message;
  function context(){
    if(!get('capture-form').reportValidity())throw new Error('Complete the collection labels and consent first.');
    return Object.freeze({participant:get('participant').value.trim(),session:get('capture-session').value.trim(),letter:get('capture-letter').value,hand:get('capture-hand').value,lighting:get('capture-lighting').value,angle:get('capture-angle').value,label_review:get('capture-verified').checked?'collector_confirmed':null,consent_version:get('capture-consent').checked?CONSENT_VERSION:null});
  }
  function lock(locked){fields.forEach(id=>get(id).disabled=locked);get('capture-next').disabled=locked;get('import-samples').disabled=locked;}
  function cancelBurst(){burstToken++;burstActive=false;lock(false);get('capture-cancel').hidden=true;render();}
  const camera=new CameraController({
    onState:on=>{if(!on){cancelBurst();pending=null;}render();},
    onResult:(_result,points)=>{
      let message='Bring one complete hand into view.';
      if(points){try{checkGeometry(points.flatMap(p=>[p.x,p.y]));message='One hand tracked. Keep the sign valid while varying the viewpoint.';
        const observed=camera.observation;
        if(observed?.observed_hand&&observed.observed_hand!==get('capture-hand').value&&observed.observed_hand_score>.8&&get('capture-hand').value)message=`Detector suggests ${observed.observed_hand}; verify your actual-hand label. Labels are not changed automatically.`;
      }catch(error){message=error.message;}}
      get('capture-quality').textContent=message;
    },
    onSequenceStart:()=>{
      try{if(burstActive)throw new Error('Stop the static burst before recording movement.');const snapshot=context();if(!['J','Z'].includes(snapshot.letter))throw new Error('Choose J or Z in the letter selector before recording movement.');pending={context:snapshot,observation:{...camera.observation}};lock(true);render();status(`Recording a real ${snapshot.letter} sequence for ${snapshot.participant}, ${snapshot.hand} hand.`);return true;}
      catch(error){status(error.message);toast(error.message);return false;}
    },
    onSequence:frames=>{
      if(!pending||!get('capture-consent').checked)return;
      try{store.capture(pending.context,{frames,clip:crypto.randomUUID()},pending.observation);render();status('Movement sample added. Label and condition metadata came from the start of the recording.');}
      catch(error){status(error.message);throw error;}
    },
    onSequenceEnd:({cancelled})=>{pending=null;lock(false);render();if(cancelled)status('Recording cancelled. No partial movement sample was saved.');}
  });
  camera.bind();
  function render(){
    const samples=store.samples,counts=store.coverage();
    const busy=burstActive||camera.recording||Boolean(pending);
    get('capture-total').textContent=samples.length;
    get('capture-people').textContent=new Set(samples.map(s=>s.participant)).size;
    get('capture-hands').textContent=`${samples.filter(s=>s.hand==='left').length} / ${samples.filter(s=>s.hand==='right').length}`;
    get('capture-q').textContent=counts.Q.total;
    get('capture-count').textContent=`${samples.length} unique sample${samples.length===1?'':'s'} collected`;
    const motion=['J','Z'].includes(get('capture-letter').value);
    get('capture-sample').textContent=motion?'Record movement sample':`Capture ${get('capture-letter').value} sample`;
    get('capture-sample').disabled=!camera.running||busy;
    get('capture-burst').disabled=!camera.running||motion||busy;
    get('capture-undo').disabled=!samples.length||busy;
    const statics=samples.filter(s=>!s.frames).length,motions=samples.length-statics;
    get('download-static').disabled=!statics;get('download-static').textContent=`Download static (${statics})`;
    get('download-motion').disabled=!motions;get('download-motion').textContent=`Download J/Z (${motions})`;
    get('download-coverage').disabled=!samples.length;
    const person=get('participant').value.trim(),hand=get('capture-hand').value,letter=get('capture-letter').value;
    const cell=samples.filter(s=>s.participant===person&&s.hand===hand&&s.letter===letter).length;
    const target=plan.samples_per_participant_letter_hand;
    get('capture-cell-count').textContent=person&&hand?`${letter}: ${cell} / ${target} for ${person}, ${hand} hand`:'Choose a participant and hand to see the current target.';
    get('capture-cell-progress').value=Math.min(cell,target);get('capture-cell-progress').max=target;
    get('capture-priority').textContent=letter==='Q'?'Q needs extra real examples. Preserve its downward orientation.':plan.priority_letters.includes(letter)?'Priority letter: compare carefully with its similar handshapes.':motion?'Record a complete movement. A still J/Z pose is not a valid sequence.':'Cover every letter and both hands before collecting extra priority examples.';
    const body=get('coverage-rows');body.replaceChildren();
    for(const key of plan.alphabet){const row=document.createElement('tr');if(plan.priority_letters.includes(key))row.className='priority-row';const first=document.createElement('th');first.scope='row';const choose=document.createElement('button');choose.type='button';choose.className='letter-select';choose.textContent=key;choose.setAttribute('aria-label',`Collect ${key}${plan.priority_letters.includes(key)?', priority letter':''}`);choose.disabled=busy;choose.addEventListener('click',()=>{get('capture-letter').value=key;get('capture-verified').checked=false;render();get('capture-letter').focus();});first.append(choose);row.append(first);for(const value of ['total','participants','left','right','lighting','angles']){const cell=document.createElement('td');cell.textContent=counts[key][value];row.append(cell);}body.append(row);}
  }
  function captureOne(snapshot){
    if(!camera.running||!camera.points||performance.now()-camera.lastSeen>400)throw new Error('Bring one clearly visible hand into the camera first.');
    if(performance.now()-lastCapture<900)throw new Error('Pause and change your viewing position slightly before the next capture.');
    const points=camera.points.flatMap(p=>[p.x,p.y]);
    store.capture(snapshot,{points},{...camera.observation});lastCapture=performance.now();render();
  }
  get('capture-form').addEventListener('submit',event=>{
    event.preventDefault();try{const snapshot=context();if(['J','Z'].includes(snapshot.letter)){camera.recordMotion();return;}captureOne(snapshot);status(`Captured ${snapshot.letter}. Keep the sign correct and vary the next real example.`);}catch(error){status(error.message);toast(error.message);}
  });
  get('capture-burst').addEventListener('click',async()=>{
    let snapshot;try{snapshot=context();}catch(error){status(error.message);return;}
    if(['J','Z'].includes(snapshot.letter)||!camera.running)return;
    const token=++burstToken;burstActive=true;lock(true);get('capture-cancel').hidden=false;render();let added=0;
    try{for(let attempts=0;attempts<15&&added<5&&token===burstToken;attempts++){
      if(!camera.running||!get('capture-consent').checked)break;
      try{captureOne(snapshot);added++;status(`Added ${added}/5 distinct ${snapshot.letter} samples. Change your viewing position while keeping the letter valid.`);}catch(error){status(`${added}/5 added. ${error.message}`);}
      if(added<5)await new Promise(resolve=>setTimeout(resolve,1100));
    }}finally{if(token===burstToken){burstActive=false;lock(false);get('capture-cancel').hidden=true;render();status(`Burst finished: ${added} actual samples added. Copied or identical coordinates were not counted.`);}}
  });
  get('capture-cancel').addEventListener('click',()=>{cancelBurst();status('Collection stopped. Samples already captured remain in memory.');});
  get('capture-next').addEventListener('click',()=>{if(!get('participant').value.trim()||!get('capture-hand').value)return toast('Choose the participant and actual hand first.');const next=store.nextLetter(get('participant').value.trim(),get('capture-hand').value,plan);if(!next)return status('This signer and hand reached the plan. Start another hand or session with real new captures.');get('capture-letter').value=next;get('capture-verified').checked=false;render();status(`Suggested ${next}: a missing or underrepresented cell for this signer and hand.`);});
  get('capture-undo').addEventListener('click',()=>{store.undo();render();status('Last sample removed.');});
  for(const id of fields)get(id).addEventListener('change',()=>{if(['participant','capture-letter','capture-hand'].includes(id))get('capture-verified').checked=false;render();});
  get('participant').addEventListener('input',render);
  get('capture-consent').addEventListener('change',event=>{if(!event.target.checked){cancelBurst();camera.stop();store.clear();pending=null;render();status('Consent withdrawn. Camera stopped and all in-memory samples cleared.');}});
  function download(name,data){const url=URL.createObjectURL(new Blob([JSON.stringify(data,null,2)],{type:'application/json'})),link=document.createElement('a');link.href=url;link.download=name;link.click();setTimeout(()=>URL.revokeObjectURL(url),1000);status('Local download requested. No coordinates or footage were uploaded.');}
  get('download-static').addEventListener('click',()=>download('asl-static-samples-v3.json',store.export('static')));
  get('download-motion').addEventListener('click',()=>download('asl-motion-samples-v3.json',store.export('motion')));
  get('download-coverage').addEventListener('click',()=>download('asl-collection-coverage.json',{plan,counts:store.coverage(),participants:new Set(store.samples.map(s=>s.participant)).size,note:'Counts cover only explicitly captured/imported samples. Conditions and labels are collector-reported.'}));
  get('import-samples').addEventListener('change',async event=>{let added=0,duplicates=0;try{for(const file of event.target.files){if(file.size>20*1024*1024)throw new Error('Import files smaller than 20 MB. Use the Python tools for larger datasets.');const result=store.merge(JSON.parse(await file.text()));added+=result.added;duplicates+=result.duplicates;}render();status(`Imported ${added} original samples; skipped ${duplicates} duplicates. Participant and condition labels were preserved.`);}catch(error){render();status(error.message);toast(error.message);}finally{event.target.value='';}});
  render();
}
