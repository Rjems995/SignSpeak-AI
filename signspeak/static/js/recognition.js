// Shared math stays free of DOM dependencies so it can be checked against Python.
export const FEATURE_VERSION = 'wrist-scale-orientation-xy-v2';
export const CONNECTIONS = [[0,1],[1,2],[2,3],[3,4],[0,5],[5,6],[6,7],[7,8],[5,9],[9,10],[10,11],[11,12],[9,13],[13,14],[14,15],[15,16],[13,17],[0,17],[17,18],[18,19],[19,20]];

export function normalize(points) {
  if (!Array.isArray(points)) throw new Error('Expected hand landmarks.');
  let p = points;
  if (points.length === 42) p = Array.from({length:21}, (_, i) => [points[i*2], points[i*2+1]]);
  if (p.length !== 21) throw new Error('Expected 21 hand landmarks.');
  const xy = p.map(v => Array.isArray(v) ? [v[0],v[1]] : [v.x,v.y]);
  if (xy.some(v => v.some(n => !Number.isFinite(n)))) throw new Error('Invalid hand coordinates.');
  const relative = xy.map(([x,y]) => [x-xy[0][0], y-xy[0][1]]);
  const [vx,vy] = relative[9];
  const squared = vx*vx + vy*vy;
  if (squared < 1e-12) throw new Error('Hand is too small to read.');
  const cross=relative[5][0]*relative[17][1]-relative[5][1]*relative[17][0];
  const mirror=cross<0?-1:1,size=Math.sqrt(squared);
  return relative.flatMap(([x,y])=>[x*mirror/size,y/size]);
}

export function nearestDistance(features, support) {
  if (!support?.length) return Infinity;
  return Math.min(...support.map(row => Math.sqrt(row.reduce((sum,v,i) => sum+(v-features[i])**2,0)/features.length)));
}

export function gatePrediction(probabilities, features, model, threshold) {
  const index = probabilities.indexOf(Math.max(...probabilities));
  const confidence = probabilities[index];
  const candidate = model.labels[index];
  const accepted = candidate !== '_' && confidence >= threshold && nearestDistance(features,model.support[candidate]) <= model.distance_thresholds[candidate];
  return {label:accepted ? candidate : null, confidence, candidate};
}

export class Stabilizer {
  constructor(holdMs=900) {this.holdMs=holdMs;this.reset();}
  reset() {this.frames=[];this.since=null;this.candidate=null;this.latched=null;this.releaseSince=null;}
  update(label, now) {
    // Three hundred milliseconds of a different/absent sign re-arms a repeated letter.
    if (this.latched && label !== this.latched) {
      this.releaseSince ??= now;
      if (now-this.releaseSince >= 300) this.latched=null;
    } else this.releaseSince=null;
    this.frames.push(label);
    if (this.frames.length>6) this.frames.shift();
    const stable = label && this.frames.length >= 4 && this.frames.filter(x=>x===label).length/this.frames.length >= 0.8;
    if (!stable) {this.candidate=null;this.since=null;return {stable:null,progress:0,commit:null};}
    if (this.candidate !== label) {this.candidate=label;this.since=now;}
    const progress=Math.min((now-this.since)/this.holdMs,1);
    let commit=null;
    if (progress>=1 && this.latched!==label) {commit=label;this.latched=label;}
    return {stable:label,progress,commit};
  }
}

export class LandmarkClassifier {
  constructor(model,tf,featureVersion=FEATURE_VERSION) {
    if (model.feature_version !== featureVersion) throw new Error('Model feature version does not match this app.');
    this.data=model;this.tf=tf;
    this.mean=tf.tensor1d(model.mean);this.scale=tf.tensor1d(model.scale);
    this.layers=model.layers.map(l=>({kernel:tf.tensor2d(l.kernel),bias:tf.tensor1d(l.bias)}));
  }
  predict(points,threshold) {
    const features=normalize(points);
    return this.predictFeatures(features,threshold);
  }
  probabilities(features) {
    return this.tf.tidy(()=>{
      let x=this.tf.tensor2d([features]).sub(this.mean).div(this.scale);
      this.layers.forEach((layer,i)=>{x=x.matMul(layer.kernel).add(layer.bias);if(i<this.layers.length-1)x=x.relu();});
      return Array.from(x.softmax().dataSync());
    });
  }
  predictFeatures(features,threshold) {
    const probabilities=this.probabilities(features);
    return gatePrediction(probabilities,features,this.data,threshold);
  }
}

export function drawHand(canvas,points,{reference=false}={}) {
  const ctx=canvas.getContext('2d');ctx.clearRect(0,0,canvas.width,canvas.height);
  if (!points) return;
  let xy=points.length===42?Array.from({length:21},(_,i)=>[points[i*2],points[i*2+1]]):points.map(v=>Array.isArray(v)?v:[v.x,v.y]);
  if(reference) {
    const normalized=normalize(points);xy=Array.from({length:21},(_,i)=>[normalized[i*2],normalized[i*2+1]]);
    const xs=xy.map(p=>p[0]),ys=xy.map(p=>p[1]);const minX=Math.min(...xs),minY=Math.min(...ys);
    const scale=Math.min((canvas.width-40)/(Math.max(...xs)-minX),(canvas.height-25)/(Math.max(...ys)-minY));
    const ox=(canvas.width-(Math.max(...xs)-minX)*scale)/2,oy=(canvas.height-(Math.max(...ys)-minY)*scale)/2;
    xy=xy.map(([x,y])=>[ox+(x-minX)*scale,oy+(y-minY)*scale]);
  } else xy=xy.map(([x,y])=>[x*canvas.width,y*canvas.height]);
  ctx.lineCap='round';ctx.lineJoin='round';ctx.lineWidth=reference?3:3;ctx.strokeStyle=reference?'#b4a3d7':'#c5acef';
  for(const [a,b] of CONNECTIONS){ctx.beginPath();ctx.moveTo(...xy[a]);ctx.lineTo(...xy[b]);ctx.stroke();}
  for(let i=0;i<xy.length;i++){ctx.beginPath();ctx.arc(...xy[i],reference?3.4:4,0,Math.PI*2);ctx.fillStyle=reference?'#fcfaff':'#f2ecff';ctx.fill();ctx.strokeStyle='#8e75b7';ctx.lineWidth=reference?1.2:1.5;ctx.stroke();}
}
