// Standalone CDP trace of an idle chapter page: what does the renderer do natively with zero input?
//
// Uses tests/lib/cdp.mjs rather than hand-rolling a launcher (learnings-19b8). The hand-rolled
// version spawned a detached chromium and did its own handshake, so it got none of the shared
// driver's guarantees: no process-group kill on a throw, no exit-hook cleanup, no devtools-endpoint
// retry. Its own comment records the cost of that - a leaked browser the reaper collected 46
// minutes later. Trace chunks come from the `page.on('Tracing.dataCollected')` subscription added
// for exactly this probe.
import {launch} from '../../tests/lib/cdp.mjs';
import {serve} from '../../scripts/serve.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const local = await serve();
const page = await launch({candidates: ['/usr/bin/chromium']});

try {
const traceChunks = [];
page.on('Tracing.dataCollected', params => {
  if (Array.isArray(params?.value)) traceChunks.push(...params.value);
});
await page.send('Tracing.start', {traceConfig: {includedCategories: ['devtools.timeline']}});
// Load + a 9s idle window, the same window the hand-rolled version measured, so the numbers stay
// comparable with the ones recorded in FINDINGS.md.
await page.goto(local.url + 'neural-networks/gradient-descent.html', {timeout: 30000});
await sleep(9000);
await page.send('Tracing.end').catch(() => {});
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
} finally {
  await page.close();
  await new Promise((resolve, reject) => local.server.close(err => err ? reject(err) : resolve()));
}
