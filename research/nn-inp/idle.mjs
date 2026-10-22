// Idle-only probe: load a page, install longtask observer, sit idle N seconds, report.
import {launch} from '/home/exedev/worktrees/learnings-nnflaky/tests/lib/cdp.mjs';
import {serve} from '/home/exedev/worktrees/learnings-nnflaky/scripts/serve.mjs';

const sleep = ms => new Promise(r => setTimeout(r, ms));
const page = process.argv[2] ?? 'neural-networks/gradient-descent.html';
const idleMs = Number(process.argv[3] ?? 8000);
const local = await serve();
const browser = await launch({candidates: ['/usr/bin/chromium']});
try {
  await browser.goto(local.url + page);
  await browser.waitFor(() => Boolean(document.querySelector('canvas')));
  await browser.evaluate(() => {
    window.__lt = [];
    new PerformanceObserver(list => { for (const e of list.getEntries()) window.__lt.push(
      {start: Math.round(e.startTime), dur: Math.round(e.duration)}); }).observe({type: 'longtask', buffered: true});
  });
  await sleep(idleMs);
  const lt = await browser.evaluate(() => window.__lt);
  console.log(`${page}: idle ${idleMs}ms -> ${lt.length} long tasks`);
  for (const t of lt) console.log(`  ${t.dur}ms @ ${t.start}`);
} finally {
  await browser.close();
  await new Promise((resolve, reject) => local.server.close(err => err ? reject(err) : resolve()));
}
