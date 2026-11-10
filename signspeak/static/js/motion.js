import {LandmarkClassifier,drawHand} from './recognition.js';
export const MOTION_FEATURE_VERSION='temporal-palm-scale-16x5xy-v1';
const median=values=>{const sorted=[...values].sort((a,b)=>a-b),middle=Math.floor(sorted.length/2);return sorted.length%2?sorted[middle]:(sorted[middle-1]+sorted[middle])/2;};
export function motionFeatures(frames){
  if(frames.length<8)throw new Error('At least eight tracked frames are needed. Keep one hand in view.');
  let time=frames.map(f=>f.t);
  const points=frames.map(f=>Array.from({length:21},(_,i)=>[f.points[i*2],f.points[i*2+1]]));
  if(time.some(t=>!Number.isFinite(t))||points.some(p=>p.some(v=>v.some(n=>!Number.isFinite(n))))||time.some((t,i)=>i&&t<=time[i-1]))throw new Error('Invalid movement coordinates.');
  if(time.at(-1)-time[0]<250||time.some((t,i)=>i&&t-time[i-1]>500))throw new Error('Keep one hand in view throughout the movement.');
  const scale=median(points.map(p=>Math.hypot(p[9][0]-p[0][0],p[9][1]-p[0][1])));
  if(scale<1e-6)throw new Error('Bring your hand closer.');
  const cross=median(points.map(p=>(p[5][0]-p[0][0])*(p[17][1]-p[0][1])-(p[5][1]-p[0][1])*(p[17][0]-p[0][0]))),mirror=cross<0?-1:1;
  let centered=points.map(p=>p.map(([x,y])=>[(x-points[0][0][0])*mirror/scale,(y-points[0][0][1])/scale]));
  const movement=Math.max(...[8,20].map(j=>{const xs=centered.map(p=>p[j][0]),ys=centered.map(p=>p[j][1]);return Math.hypot(Math.max(...xs)-Math.min(...xs),Math.max(...ys)-Math.min(...ys));}));
  if(movement<.45)throw new Error('Trace a clear movement; a stationary hand is not J or Z.');
  const speed=centered.slice(1).map((p,i)=>Math.max(...[8,20].map(j=>Math.hypot(p[j][0]-centered[i][j][0],p[j][1]-centered[i][j][1]))));
  const threshold=Math.max(.035,Math.max(...speed)*.12),active=speed.flatMap((v,i)=>v>=threshold?[i]:[]);
  const lo=active.length?Math.max(0,active[0]-1):0,hi=active.length?Math.min(frames.length-1,active.at(-1)+2):frames.length-1;
  time=time.slice(lo,hi+1);centered=centered.slice(lo,hi+1);const origin=[...centered[0][0]];
  centered=centered.map(p=>p.map(([x,y])=>[x-origin[0],y-origin[1]]));
  const selected=centered.map(p=>[0,4,8,12,20].flatMap(j=>p[j]));
  const output=[];let cursor=0;
  for(let step=0;step<16;step++){const t=time[0]+(time.at(-1)-time[0])*step/15;while(cursor<time.length-2&&time[cursor+1]<t)cursor++;const ratio=(t-time[cursor])/(time[cursor+1]-time[cursor]);output.push(...selected[cursor].map((v,i)=>v+(selected[cursor+1][i]-v)*ratio));}
  return output;
}
export class MotionClassifier extends LandmarkClassifier{
  constructor(model,tf){super(model,tf,MOTION_FEATURE_VERSION);}
  predict(frames,threshold){return this.predictFeatures(motionFeatures(frames),threshold);}
}
export function drawMotionReference(canvas,frames,letter){
  if(!frames?.length)return;
  const all=frames.flatMap(f=>f.points),xs=all.filter((_,i)=>i%2===0),ys=all.filter((_,i)=>i%2===1);
  const minX=Math.min(...xs),minY=Math.min(...ys),maxX=Math.max(...xs),maxY=Math.max(...ys);
  const scale=Math.min((canvas.width-25)/(maxX-minX),(canvas.height-20)/(maxY-minY));
  const ox=(canvas.width-(maxX-minX)*scale)/2,oy=(canvas.height-(maxY-minY)*scale)/2;
  const map=points=>Array.from({length:21},(_,i)=>({x:(ox+(points[i*2]-minX)*scale)/canvas.width,y:(oy+(points[i*2+1]-minY)*scale)/canvas.height}));
  const context=canvas.getContext('2d'),joint=letter==='J'?20:8;
  let start=null,lastFrame=-1;
  function animate(now){if(!canvas.isConnected)return;start??=now;const phase=(now-start)%2600/2600,index=Math.min(frames.length-1,Math.floor(phase*frames.length));if(index!==lastFrame){lastFrame=index;drawHand(canvas,map(frames[index].points));context.beginPath();frames.slice(0,index+1).forEach((f,i)=>{const p=map(f.points)[joint],x=p.x*canvas.width,y=p.y*canvas.height;if(i)context.lineTo(x,y);else context.moveTo(x,y);});context.strokeStyle='#7560d8';context.lineWidth=2;context.stroke();}if(!window.matchMedia('(prefers-reduced-motion: reduce)').matches)requestAnimationFrame(animate);}
  if(window.matchMedia('(prefers-reduced-motion: reduce)').matches){drawHand(canvas,map(frames.at(-1).points));context.beginPath();frames.forEach((f,i)=>{const p=map(f.points)[joint];if(i)context.lineTo(p.x*canvas.width,p.y*canvas.height);else context.moveTo(p.x*canvas.width,p.y*canvas.height);});context.strokeStyle='#7560d8';context.stroke();}else requestAnimationFrame(animate);
}
