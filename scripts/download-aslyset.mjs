// Node fallback for the publisher's download endpoint, which rejects some
// Python HTTP clients. Credentials and signed redirect URLs are never logged.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {Readable,Transform} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const directory=path.join(root,'datasets','public','aslyset');
fs.mkdirSync(directory,{recursive:true});
const file=path.join(directory,'ASLYset.rar');
const expected='e50202c30dfff4795b88762b765bbeb357e184c13fe22b8f6e0d836d02235c80';
const url='https://data.mendeley.com/public-files/datasets/xs6mvhx6rh/files/8c504c17-1765-4935-9f4f-4a2aa68cb9f0/file_downloaded?download=1';
if(fs.existsSync(file)){console.log('Archive exists; Python will verify it.');process.exit(0);}
const response=await fetch(url,{headers:{Range:'bytes=0-'}});
if(!response.ok)throw new Error(`Publisher returned HTTP ${response.status}`);
let bytes=0;const hash=crypto.createHash('sha256');
await pipeline(Readable.fromWeb(response.body),new Transform({transform(chunk,encoding,next){bytes+=chunk.length;hash.update(chunk);next(null,chunk);}}),fs.createWriteStream(file+'.part'));
if(bytes!==872115989||hash.digest('hex')!==expected)throw new Error('Archive size/hash mismatch. Partial file was not imported.');
fs.renameSync(file+'.part',file);
console.log('Downloaded and verified ASLYset.');
