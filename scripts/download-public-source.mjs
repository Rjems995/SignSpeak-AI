// Public archives are streamed to the ignored cache and verified before use.
import fs from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import {Readable,Transform} from 'node:stream';
import {pipeline} from 'node:stream/promises';
import {fileURLToPath} from 'node:url';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const sources=JSON.parse(fs.readFileSync(path.join(root,'datasets','public-sources.json'),'utf8'));
const key=process.argv[2],source=sources[key];if(!source)throw new Error(`Choose ${Object.keys(sources).join(', ')}.`);
const directory=path.join(root,'datasets','public',key);fs.mkdirSync(directory,{recursive:true});
const file=path.join(directory,source.archive);
async function digest(file){const hash=crypto.createHash('sha256');for await(const chunk of fs.createReadStream(file))hash.update(chunk);return hash.digest('hex');}
if(fs.existsSync(file)){if(fs.statSync(file).size!==source.bytes||await digest(file)!==source.sha256)throw new Error('Cached archive hash mismatch.');console.log(`${key}: cached archive verified.`);process.exit(0);}
console.log(`Downloading ${key}: ${(source.bytes/1e6).toFixed(0)} MB`);
const response=await fetch(source.download_url??`https://data.mendeley.com/public-files/datasets/${source.dataset}/files/${source.file}/file_downloaded?download=1`,{headers:{Range:'bytes=0-'}});
if(!response.ok)throw new Error(`Public download returned HTTP ${response.status}`);
const hash=crypto.createHash('sha256');let bytes=0;
await pipeline(Readable.fromWeb(response.body),new Transform({transform(chunk,encoding,next){bytes+=chunk.length;hash.update(chunk);next(null,chunk);}}),fs.createWriteStream(file+'.part'));
const actualHash=hash.digest('hex');
if(bytes!==source.bytes||actualHash!==source.sha256)throw new Error('Archive hash/size mismatch; partial data was not imported.');
fs.renameSync(file+'.part',file);console.log(`${key}: downloaded ${bytes} bytes; SHA-256 ${actualHash}.`);
