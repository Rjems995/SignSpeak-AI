import test from 'node:test';
import assert from 'node:assert/strict';
import {HandTracker} from '../signspeak/static/js/hand-tracker.js';

class FakeWorker {
  constructor(){this.messages=[];this.terminated=false;}
  postMessage(message,transfer){this.messages.push({message,transfer});if(this.postError)throw this.postError;}
  terminate(){this.terminated=true;}
  reply(data){this.onmessage?.({data});}
}
function harness(options={}) {
  const workers=[];
  const tracker=new HandTracker({...options,workerFactory:(url,options)=>{
    const worker=new FakeWorker();worker.url=url;worker.options=options;workers.push(worker);return worker;
  }});
  return {tracker,workers};
}
function bitmap(){return {closed:0,close(){this.closed++;}};}
async function ready(tracker,workers){const init=tracker.init();const worker=workers.at(-1);worker.reply({type:'ready',id:worker.messages[0].message.id});await init;return worker;}

test('initializes once and transfers one frame with its capture timestamp',async()=>{
  const {tracker,workers}=harness();
  const init=tracker.init();assert.equal(tracker.init(),init);
  const worker=workers[0];assert.equal(worker.options.type,'classic');
  assert.ok(worker.url.pathname.endsWith('/hand-tracking-worker.js'));
  worker.reply({type:'ready',id:worker.messages[0].message.id});await init;
  const frame=bitmap(),result={landmarks:[[]],worldLandmarks:[[]],handednesses:[[]]};
  const detection=tracker.detect(frame,1234);
  const request=worker.messages[1];
  assert.deepEqual(request.transfer,[frame]);assert.equal(request.message.timestamp,1234);
  assert.equal(frame.closed,0,'the worker owns a successfully transferred frame');
  worker.reply({type:'result',id:request.message.id,result});
  assert.equal(await detection,result);tracker.close();assert.equal(worker.terminated,true);
});

test('rejects a second frame without queuing and closes frames it cannot transfer',async()=>{
  const {tracker,workers}=harness();const worker=await ready(tracker,workers);
  const first=bitmap(),second=bitmap();
  const pending=tracker.detect(first,1);
  await assert.rejects(tracker.detect(second,2),{name:'InvalidStateError'});
  assert.equal(second.closed,1);assert.equal(worker.messages.length,2);
  const invalid=bitmap();
  worker.reply({type:'result',id:worker.messages[1].message.id,result:{landmarks:[]}});await pending;
  await assert.rejects(tracker.detect(invalid,NaN),{name:'DataError'});assert.equal(invalid.closed,1);
  tracker.close();const stopped=bitmap();
  await assert.rejects(tracker.detect(stopped,3),{name:'InvalidStateError'});assert.equal(stopped.closed,1);
});

test('close cancels initialization or detection and immediate restart isolates late replies',async()=>{
  const {tracker,workers}=harness();
  const init=tracker.init();const old=workers[0],late=old.onmessage;
  const cancelled=assert.rejects(init,{name:'AbortError'});
  tracker.close();await cancelled;assert.equal(old.terminated,true);
  const worker=await ready(tracker,workers);
  late({data:{type:'ready',id:old.messages[0].message.id}});
  assert.equal(tracker.worker,worker);
  const detection=tracker.detect(bitmap(),4),aborted=assert.rejects(detection,{name:'AbortError'});
  tracker.close();await aborted;assert.equal(worker.terminated,true);
  tracker.close();assert.equal(tracker.pending.size,0);
});

test('postMessage failures retain bitmap ownership and stop a broken worker',async()=>{
  const {tracker,workers}=harness();const worker=await ready(tracker,workers);
  worker.postError=new Error('Transfer failed');const frame=bitmap();
  await assert.rejects(tracker.detect(frame,1),/Transfer failed/);
  assert.equal(frame.closed,1);assert.equal(worker.terminated,true);assert.equal(tracker.ready,false);
});

test('worker failures and invalid results reject pending work and permit retry',async()=>{
  for(const failure of ['error','messageerror','protocol','remote']){
    const {tracker,workers}=harness();const worker=await ready(tracker,workers);
    const pending=tracker.detect(bitmap(),1);const rejected=assert.rejects(pending);
    if(failure==='error')worker.onerror({preventDefault(){}});
    if(failure==='messageerror')worker.onmessageerror();
    if(failure==='protocol')worker.reply({type:'result',id:worker.messages[1].message.id,result:{}});
    if(failure==='remote')worker.reply({type:'error',id:worker.messages[1].message.id,error:{name:'Error',message:'Detection failed'}});
    await rejected;assert.equal(worker.terminated,true);assert.equal(tracker.pending.size,0);
    await ready(tracker,workers);tracker.close();
  }
});

test('initialization and detection timeouts terminate stalled inference',async()=>{
  const {tracker,workers}=harness({initTimeout:10,detectTimeout:10});
  await assert.rejects(tracker.init(),{name:'TimeoutError'});
  assert.equal(workers[0].terminated,true);
  const worker=await ready(tracker,workers);
  await assert.rejects(tracker.detect(bitmap(),1),{name:'TimeoutError'});
  assert.equal(worker.terminated,true);assert.equal(tracker.pending.size,0);
});

test('unsupported worker creation fails initialization without a UI-thread fallback',async()=>{
  const tracker=new HandTracker({workerFactory:()=>{throw new Error('Workers unavailable');}});
  await assert.rejects(tracker.init(),/Workers unavailable/);assert.equal(tracker.worker,null);
});
