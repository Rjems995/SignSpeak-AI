// Research/education-only source. Raw clips/annotations stay in the ignored cache.
import fs from 'node:fs/promises';
import crypto from 'node:crypto';
const directory = 'datasets/public/asllvd';
const manifest = JSON.parse(await fs.readFile(`${directory}/selection.json`, 'utf8'));
for (const row of manifest.files) {
  if (
    !/^[A-Za-z]+_[0-9]+\.mov$/.test(row.file) ||
    !row.url.startsWith('https://csr.bu.edu/ftp/asl/asllvd/')
  )
    throw Error('Unexpected research clip path');
  const file = `${directory}/${row.file}`;
  let data;
  try {
    data = await fs.readFile(file);
  } catch (error) {
    if (error.code !== 'ENOENT') throw error;
    const response = await fetch(row.url, { signal: AbortSignal.timeout(45000) });
    if (!response.ok) throw Error(`Clip HTTP ${response.status}`);
    data = Buffer.from(await response.arrayBuffer());
    if (data.length > 15000000) throw Error('Unexpected research clip size');
    await fs.writeFile(file, data);
  }
  const digest = crypto.createHash('sha256').update(data).digest('hex');
  if (row.sha256 && row.sha256 !== digest) throw Error('Research clip SHA-256 mismatch');
  row.sha256 = digest;
}
await fs.writeFile(`${directory}/selection.json`, JSON.stringify(manifest, null, 2));
console.log(
  `Verified ${manifest.files.length} research-only clips; excluded from training and redistribution.`,
);
