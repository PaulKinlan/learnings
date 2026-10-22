// CPU-profile probe: idle profile for gradient-descent, click-phase profile for architectures.
import {launch} from '../../tests/lib/cdp.mjs';
import {serve} from '../../scripts/serve.mjs';
import {writeFileSync} from 'node:fs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const local = await serve();
const browser = await launch({candidates: ['/usr/bin/chromium']});

async function profile(page, seconds, outFile, drive) {
  await browser.send('Profiler.enable');
  await browser.send('Profiler.setSamplingInterval', {interval: 500});
  await browser.send('Profiler.start');
  if (drive) await drive();
  await sleep(seconds * 1000);
  const {profile: p} = await browser.send('Profiler.stop');
  writeFileSync(outFile, JSON.stringify(p));
  // Summarise self time per function.
  const {nodes, samples = [], timeDeltas = []} = p;
  const byId = new Map(nodes.map(n => [n.id, n]));
  const self = new Map();
  for (let i = 0; i < samples.length; i++) {
    const id = samples[i];
    const n = byId.get(id);
    if (!n) continue;
    const key = `${(n.callFrame.functionName || '(anonymous)')} @ ${(n.callFrame.url || '').split('/').pop()}:${n.callFrame.lineNumber + 1}`;
    self.set(key, (self.get(key) ?? 0) + (timeDeltas[i] ?? 0));
  }
  const rows = [...self.entries()].sort((a, b) => b[1] - a[1]).slice(0, 15)
    .map(([k, v]) => `  ${(v / 1000).toFixed(0)}ms  ${k}`);
  console.log(`\n=== ${page} (${seconds}s, ${outFile}) ===\n${rows.join('\n')}`);
}

const clickCanvas = async sel => {
  const at = await browser.evaluate(sel => {
    const el = document.querySelector(sel);
    el.scrollIntoView({block: 'center'});
    const b = el.getBoundingClientRect();
    return {x: Math.round(b.left + b.width / 2), y: Math.round(b.top + b.height / 2)};
  }, sel);
  for (let i = 0; i < 4; i++) {
    for (const type of ['mousePressed', 'mouseReleased']) {
      await browser.send('Input.dispatchMouseEvent', {type, x: at.x, y: at.y, button: 'left', clickCount: 1});
    }
    await sleep(250);
  }
};

try {
  await browser.goto(local.url + 'neural-networks/gradient-descent.html');
  await browser.waitFor(() => Boolean(document.querySelector('canvas')));
  await sleep(500);
  await profile('gradient-descent idle', 6, '/tmp/nn-inp/gd-idle.json');

  await browser.goto(local.url + 'neural-networks/architectures.html');
  await browser.waitFor(() => Boolean(document.querySelector('canvas')));
  await sleep(500);
  await profile('architectures canvas clicks', 4, '/tmp/nn-inp/arch-clicks.json',
    () => clickCanvas('canvas'));
} finally {
  await browser.close();
  await new Promise((resolve, reject) => local.server.close(err => err ? reject(err) : resolve()));
}
