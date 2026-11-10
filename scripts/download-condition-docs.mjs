// Fetch source documentation only. Never execute downloaded scripts or HTML.
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
const specs=JSON.parse(await fs.readFile('datasets/condition-documents.json','utf8'));
const directory='datasets/public/conditions-research';await fs.mkdir(directory,{recursive:true});
for(const [name,spec] of Object.entries(specs)){
  if(!/^[a-z-]+\.(txt|pdf)$/.test(name))throw Error('Unexpected documentation filename');
  let data;
  try{data=await fs.readFile(`${directory}/${name}`);}catch(error){if(error.code!=='ENOENT')throw error;const response=await fetch(spec.url,{signal:AbortSignal.timeout(30000)});if(!response.ok)throw Error(`Documentation HTTP ${response.status}`);data=Buffer.from(await response.arrayBuffer());}
  if(data.length!==spec.bytes||crypto.createHash('sha256').update(data).digest('hex')!==spec.sha256)throw Error(`Source documentation hash mismatch: ${name}`);
  await fs.writeFile(`${directory}/${name}`,data);console.log(`Verified original source document: ${name}`);
}
