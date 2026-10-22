// INP / presentation-delay probe for the neural-network chapter pages (learnings-6nb).
// Uses the repo's own zero-dependency CDP driver. Trusted input only (Input.dispatchMouseEvent),
// because synthetic dispatchEvent produces no Event Timing entries.
import {launch} from '/home/exedev/worktrees/learnings-nnflaky/tests/lib/cdp.mjs';
import {serve} from '/home/exedev/worktrees/learnings-nnflaky/scripts/serve.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const local = await serve();
const browser = await launch({candidates: ['/usr/bin/chromium']});
const pages = process.argv.slice(2).length ? process.argv.slice(2)
  : ['neural-networks/architectures.html', 'neural-networks/gradient-descent.html'];

const INSTALL = () => {
  window.__events = []; window.__longtasks = []; window.__raf = [];
  new PerformanceObserver(list => { for (const e of list.getEntries()) window.__events.push(
    {name: e.name, start: Math.round(e.startTime), dur: Math.round(e.duration),
      ps: Math.round(e.processingStart), pe: Math.round(e.processingEnd), id: e.interactionId});
  }).observe({type: 'event', durationThreshold: 8, buffered: true});
  new PerformanceObserver(list => { for (const e of list.getEntries()) window.__longtasks.push(
    {start: Math.round(e.startTime), dur: Math.round(e.duration), who: e.attribution?.[0]?.name});
  }).observe({type: 'longtask', buffered: true});
  // rAF cadence probe: timestamps of 120 consecutive frames, to find stalls.
  let n = 0; const t0 = performance.now();
  const tick = now => { if (n++ < 120) { window.__raf.push(Math.round(now - t0)); requestAnimationFrame(tick); } };
  requestAnimationFrame(tick);
};

const CANVASES = () => ({
  dpr: devicePixelRatio,
  canvases: [...document.querySelectorAll('canvas')].map(c => {
    const r = c.getBoundingClientRect();
    return {id: c.id, w: c.width, h: c.height, cssW: Math.round(r.width), cssH: Math.round(r.height)};
  }),
  math: document.querySelectorAll('math').length,
  nodes: document.getElementsByTagName('*').length,
});

const center = async sel => {
  const at = await browser.evaluate(sel => {
    const el = document.querySelector(sel);
    el.scrollIntoView({block: 'center'});
    const b = el.getBoundingClientRect();
    return {x: Math.round(b.left + b.width / 2), y: Math.round(b.top + b.height / 2)};
  }, sel);
  return at;
};

try {
  for (const page of pages) {
    const url = local.url + page;
    await browser.goto(url);
    await browser.waitFor(() => Boolean(document.querySelector('canvas')));
    await browser.evaluate(INSTALL);
    const info = await browser.evaluate(CANVASES);
    console.log(`\n=== ${page} ===`);
    console.log(`doc: ${info.nodes} elements, ${info.math} <math>, dpr ${info.dpr}`);
    for (const c of info.canvases) {
      const scale = (c.cssW * info.dpr / c.w).toFixed(2);
      console.log(`canvas #${c.id}: backing ${c.w}x${c.h}, css ${c.cssW}x${c.cssH}, raster-scale ~${scale}x`);
    }

    // Idle rAF cadence (2s, no input).
    await sleep(2000);
    const idle = await browser.evaluate(() => {
      const t = window.__raf; const gaps = [];
      for (let i = 1; i < t.length; i++) gaps.push(t[i] - t[i - 1]);
      gaps.sort((a, b) => a - b);
      return {frames: t.length, p50: gaps[Math.floor(gaps.length / 2)] ?? null,
        max: gaps.at(-1) ?? null, over100: gaps.filter(g => g > 100).length};
    });
    console.log(`idle rAF: ${JSON.stringify(idle)}`);

    // Trusted hover sweeps over the first canvas (mouseenter/pointerover/mousemove).
    const sel = 'canvas';
    const at = await center(sel);
    for (const [dx, dy] of [[-80, -60], [-40, 0], [0, 0], [40, 0], [80, 60]]) {
      await browser.send('Input.dispatchMouseEvent', {type: 'mouseMoved', x: at.x + dx, y: at.y + dy});
      await sleep(120);
    }
    // Trusted clicks on the canvas.
    for (let i = 0; i < 6; i++) {
      for (const type of ['mousePressed', 'mouseReleased']) {
        await browser.send('Input.dispatchMouseEvent', {type, x: at.x, y: at.y, button: 'left', clickCount: 1});
      }
      await sleep(200);
    }
    await sleep(2500);

    const data = await browser.evaluate(() => ({e: window.__events, lt: window.__longtasks, raf: window.__raf}));
    console.log(`long tasks: ${data.lt.length}` + (data.lt.length
      ? ` -> ${data.lt.map(t => `${t.dur}ms@${t.start}(${t.who})`).join(', ')}` : ''));
    const rafGaps = [];
    for (let i = 1; i < data.raf.length; i++) rafGaps.push(data.raf[i] - data.raf[i - 1]);
    rafGaps.sort((a, b) => a - b);
    console.log(`rAF during interaction: n=${data.raf.length} p50=${rafGaps[Math.floor(rafGaps.length / 2)]}ms max=${rafGaps.at(-1)}ms >100ms:${rafGaps.filter(g => g > 100).length}`);
    console.log('event timing (dur - (pe-start) = presentation delay):');
    for (const e of data.e) {
      const wait = e.ps - e.start;             // queued behind busy main thread
      const proc = e.pe - e.ps;                // handler cost
      const present = e.dur - (e.pe - e.start); // render/present tail
      console.log(`  ${e.name.padEnd(12)} dur=${String(e.dur).padStart(5)}ms  queue=${String(wait).padStart(5)}ms  proc=${String(proc).padStart(3)}ms  presentTail=${String(present).padStart(5)}ms  id=${e.id}`);
    }
  }
} finally {
  await browser.close();
  await new Promise((resolve, reject) => local.server.close(err => err ? reject(err) : resolve()));
}
