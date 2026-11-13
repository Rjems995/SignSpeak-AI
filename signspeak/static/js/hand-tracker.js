// Owns a dedicated inference worker. ImageBitmaps leave the UI thread once only;
// callers must wait for the current detection before submitting another frame.
function trackerError(message,name='Error') {
  const error=new Error(message);error.name=name;return error;
}

export class HandTracker {
  constructor({workerFactory,initTimeout=90000,detectTimeout=10000}={}) {
    if(!Number.isFinite(initTimeout)||initTimeout<=0||!Number.isFinite(detectTimeout)||detectTimeout<=0)throw new TypeError('Worker timeouts must be positive.');
    this.workerFactory=workerFactory||((url,options)=>new Worker(url,options));
    this.initTimeout=initTimeout;this.detectTimeout=detectTimeout;
    this.worker=null;this.ready=false;this.initializing=null;this.pending=new Map();this.nextId=0;
  }

  init() {
    if(this.ready)return Promise.resolve(this);
    if(this.initializing)return this.initializing;
    let worker;
    try {
      // MediaPipe 0.10.21's WASM loader uses importScripts(). A classic worker
      // can dynamically import its ES module while retaining that loader API.
      worker=this.workerFactory(new URL('./hand-tracking-worker.js',import.meta.url),{type:'classic',name:'signspeak-hand-tracking'});
      this.worker=worker;
      worker.onmessage=event=>{
        if(worker!==this.worker)return;
        const message=event.data,request=this.pending.get(message?.id);
        if(!request)return; // Ignore late responses from an already settled job.
        if(message.type==='error'){
          this.fail(trackerError(message.error?.message||'Hand tracking failed.',message.error?.name||'Error'));return;
        }
        if((request.type==='init'&&message.type==='ready')||
           (request.type==='detect'&&message.type==='result'&&message.result&&Array.isArray(message.result.landmarks))){
          clearTimeout(request.timer);this.pending.delete(message.id);request.resolve(message.result);return;
        }
        this.fail(trackerError('Hand tracking returned an invalid response.','DataError'));
      };
      worker.onerror=event=>{
        event.preventDefault?.();
        if(worker===this.worker)this.fail(trackerError('Hand tracking worker stopped unexpectedly.','WorkerError'));
      };
      worker.onmessageerror=()=>{
        if(worker===this.worker)this.fail(trackerError('Hand tracking results could not be read.','DataError'));
      };
    }catch(error){
      this.fail(error);return Promise.reject(error);
    }
    const operation=this.request('init',{},[],this.initTimeout).then(()=>{
      if(this.worker!==worker)throw trackerError('Hand tracking was cancelled.','AbortError');
      this.ready=true;return this;
    }).finally(()=>{if(this.initializing===operation)this.initializing=null;});
    this.initializing=operation;return operation;
  }

  detect(frame,timestamp) {
    const reject=(message,name)=>{frame?.close?.();return Promise.reject(trackerError(message,name));};
    if(!this.ready||!this.worker)return reject('Hand tracking is not ready.','InvalidStateError');
    if(this.pending.size)return reject('A hand tracking frame is already in flight.','InvalidStateError');
    if(!frame||typeof frame.close!=='function'||!Number.isFinite(timestamp))return reject('A transferable frame and finite timestamp are required.','DataError');
    return this.request('detect',{frame,timestamp},[frame],this.detectTimeout,()=>frame.close());
  }

  request(type,body,transfer,timeout,onTransferFailure=()=>{}) {
    const id=++this.nextId;
    return new Promise((resolve,reject)=>{
      const timer=setTimeout(()=>this.fail(trackerError('Hand tracking timed out. Please retry the camera.','TimeoutError')),timeout);
      this.pending.set(id,{type,resolve,reject,timer});
      try {this.worker.postMessage({type,id,...body},transfer);}
      catch(error){onTransferFailure();this.fail(error);}
    });
  }

  fail(error) {
    const worker=this.worker;this.worker=null;this.ready=false;this.initializing=null;
    if(worker){worker.onmessage=null;worker.onerror=null;worker.onmessageerror=null;worker.terminate();}
    for(const request of this.pending.values()){clearTimeout(request.timer);request.reject(error);}
    this.pending.clear();
  }

  close() {this.fail(trackerError('Hand tracking was cancelled.','AbortError'));}
}
