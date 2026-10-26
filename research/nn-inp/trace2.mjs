// Standalone CDP trace of an idle chapter page: what does the renderer do natively with zero input?
import {spawn} from 'node:child_process';
import {mkdtempSync, rmSync} from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {serve} from '../../scripts/serve.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const local = await serve();
const profile = mkdtempSync(path.join(os.tmpdir(), 'trace-cdp-'));
const child = spawn('/usr/bin/chromium', ['--headless=new', '--disable-gpu', '--no-first-run',
  '--no-default-browser-check', '--disable-dev-shm-usage', '--window-size=1000,800',
  `--user-data-dir=${profile}`, '--remote-debugging-port=0', 'about:blank'], {stdio: ['ignore', 'pipe', 'pipe'], detached: true});
// The child is detached, so it outlives this process: any throw below would leak a browser
// (one did, until the reaper killed it 46 minutes later). Everything runs inside try/finally.
try {
const wsUrl = await new Promise((resolve, reject) => {
  let buffer = '';
  const timer = setTimeout(() => reject(new Error('no devtools endpoint')), 20000);
  const scan = c => { buffer += String(c); const m = buffer.match(/ws:\/\/127\.0\.0\.1:\d+\/devtools\/browser\/[0-9a-f-]+/); if (m) { clearTimeout(timer); resolve(m[0]); } };
  child.stderr.on('data', scan); child.stdout.on('data', scan);
});
const socket = new WebSocket(wsUrl);
await new Promise(r => socket.addEventListener('open', r, {once: true}));
let nextId = 1; const pending = new Map(); const traceChunks = [];
socket.addEventListener('message', ev => {
  const d = JSON.parse(String(ev.data));
  if (d.id && pending.has(d.id)) { const {resolve, reject} = pending.get(d.id); pending.delete(d.id);
    d.error ? reject(new Error(d.error.message)) : resolve(d.result); }
  else if (d.method === "Tracing.dataCollected" && Array.isArray(d.params?.value)) traceChunks.push(...d.params.value);
});
const send = (method, params = {}) => new Promise((resolve, reject) => { const id = nextId++;
  pending.set(id, {resolve, reject}); socket.send(JSON.stringify({id, method, params})); });
const {targetId} = await send('Target.createTarget', {url: 'about:blank'});
const {sessionId} = await send('Target.attachToTarget', {targetId, flatten: true});
// Flat-session commands must carry the sessionId IN THE MESSAGE BODY (learnings-z8f); the old
// sendP forwarded it to send() as a third argument, which send() does not take, so the session
// id was silently dropped and a future caller got a browser-level command instead of a
// page-level one. It is now a fire-and-forget wrapper over s2, which addresses the body.
const s2 = (method, params = {}) => new Promise((resolve, reject) => { const id = nextId++;
  pending.set(id, {resolve, reject}); socket.send(JSON.stringify({id, method, params, sessionId})); });
const sendP = (method, params = {}) => s2(method, params).catch(() => {});
const evalJs = async expression => (await s2('Runtime.evaluate', {expression, returnByValue: true})).result?.value;

await s2('Page.enable');
await s2('Tracing.start', {traceConfig: {includedCategories: ['devtools.timeline']}});
await s2('Page.navigate', {url: local.url + 'neural-networks/gradient-descent.html'});
await sleep(9000); // load + idle window
await s2('Tracing.end').catch(() => {});
await sleep(1500);

const events = traceChunks;
console.log(`trace events: ${events.length}`);
const names = ['UpdateLayoutTree', 'Layout', 'Paint', 'PrePaint', 'Layerize', 'RasterTask', 'ImageDecodeTask', 'MajorGC', 'MinorGC', 'FunctionCall', 'TimerFire', 'FireAnimationFrame', 'InvalidateLayout', 'ScheduleStyleRecalculation', 'ResourceSendRequest', 'ResourceReceiveData', 'ResourceFinish', 'DecodeImage', 'ResizeImage', 'PaintImage', 'Animate'];
const rows = events.filter(e => names.includes(e.name) && e.ph === 'X' && (e.dur ?? 0) > 8000);
const byName = new Map();
for (const e of rows) { (byName.get(e.name) ?? byName.set(e.name, []).get(e.name)).push(e); }
const totalAll = rows.reduce((s, e) => s + e.dur, 0) / 1000;
console.log(`events >8ms in tracked categories: ${rows.length}, total ${totalAll.toFixed(0)}ms`);
for (const [k, arr] of [...byName].sort((a, b) => b[1].reduce((s, e) => s + e.dur, 0) - a[1].reduce((s, e) => s + e.dur, 0))) {
  console.log(`${k}: n=${arr.length} total=${(arr.reduce((s, e) => s + e.dur, 0) / 1000).toFixed(0)}ms`);
  for (const e of arr.slice(0, 6)) {
    const d = e.args?.beginData ?? e.args?.data ?? {};
    const extra = (k === 'FunctionCall' || k === 'TimerFire' || k === 'FireAnimationFrame')
      ? ` ${String(d.functionName ?? d.url ?? '').slice(0, 60)}` : '';
    console.log(`  ${(e.dur / 1000).toFixed(0)}ms @ ${(e.ts / 1000 - 0).toFixed(0)}${extra}`);
  }
}
// Absolute timeline: biggest 10 events of any name > 20ms
console.log('\nbiggest 10 events (>20ms) of any tracked kind:');
for (const e of [...rows].sort((a, b) => b.dur - a.dur).slice(0, 10)) {
  const d = e.args?.beginData ?? e.args?.data ?? {};
  console.log(`  ${e.name} ${(e.dur / 1000).toFixed(0)}ms @ ${(e.ts / 1000).toFixed(0)} ${JSON.stringify(d).slice(0, 100)}`);
}
child.kill('SIGKILL'); try { process.kill(-child.pid, 'SIGKILL'); } catch {}
rmSync(profile, {recursive: true, force: true});
await new Promise((resolve, reject) => local.server.close(err => err ? reject(err) : resolve()));
} finally {
  child.kill('SIGKILL'); try { process.kill(-child.pid, 'SIGKILL'); } catch {}
  rmSync(profile, {recursive: true, force: true});
}
