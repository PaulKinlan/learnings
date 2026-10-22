// Control: same trusted-input probe against a trivial page — isolates renderer cost from page cost.
import {launch} from '/home/exedev/worktrees/learnings-nnflaky/tests/lib/cdp.mjs';
import {serve} from '/home/exedev/worktrees/learnings-nnflaky/scripts/serve.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const local = await serve();
const browser = await launch({candidates: ['/usr/bin/chromium']});
try {
  await browser.goto(local.url + 'neural-networks/index.html');
  await browser.waitFor(() => Boolean(document.querySelector('#nn-stack li h3 a')));
  await browser.evaluate(() => {
    document.addEventListener('click', event => { event.preventDefault(); event.stopImmediatePropagation(); }, {capture: true});
    window.__events = [];
    new PerformanceObserver(list => { for (const e of list.getEntries()) window.__events.push(
      {name: e.name, dur: Math.round(e.duration), ps: Math.round(e.processingStart), pe: Math.round(e.processingEnd)});
    }).observe({type: 'event', durationThreshold: 8});
  });
  const at = await browser.evaluate(sel => {
    const el = document.querySelector(sel);
    el.scrollIntoView({block: 'center'});
    const b = el.getBoundingClientRect();
    return {x: Math.round(b.left + b.width / 2), y: Math.round(b.top + b.height / 2)};
  }, '#nn-stack li h3 a');
  for (const [dx, dy] of [[-80, -40], [0, 0], [40, 0]]) {
    await browser.send('Input.dispatchMouseEvent', {type: 'mouseMoved', x: at.x + dx, y: at.y + dy});
    await sleep(150);
  }
  for (let i = 0; i < 6; i++) {
    for (const type of ['mousePressed', 'mouseReleased']) {
      await browser.send('Input.dispatchMouseEvent', {type, x: at.x, y: at.y, button: 'left', clickCount: 1});
    }
    await sleep(200);
  }
  await sleep(1500);
  const ev = await browser.evaluate(() => window.__events);
  console.log(`index.html (hub, light DOM, no canvas) trusted input: ev=${JSON.stringify(ev)?.slice(0, 60)}`);
  for (const e of ev) {
    const present = e.dur - (e.pe - e.ps);
    console.log(`  ${e.name.padEnd(12)} dur=${String(e.dur).padStart(5)}ms  presentTail=${String(present).padStart(5)}ms`);
  }
} finally {
  await browser.close();
  await new Promise((resolve, reject) => local.server.close(err => err ? reject(err) : resolve()));
}
