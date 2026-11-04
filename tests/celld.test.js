import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import {
  CasEpochSimulator,
  simulateLoadBalancing,
  calculateFleetAccounting
} from '../site/celld/app.js';

test('celld CasEpochSimulator enforces monotonic epoch fencing and rejects stale zombie writes', () => {
  const sim = new CasEpochSimulator();
  const s0 = sim.snapshot();
  assert.equal(s0.bucketRecord.owner, 'node-a');
  assert.equal(s0.bucketRecord.epoch, 1);

  const s1 = sim.commitTransaction('fleet');
  assert.equal(s1.bucketRecord.maxTxid, 2);
  assert.equal(s1.nodes['node-b'].peerReplicatedTxid, 2);

  const s2 = sim.triggerPartitionTakeover();
  assert.equal(s2.bucketRecord.owner, 'node-b');
  assert.equal(s2.bucketRecord.epoch, 2);
  assert.equal(s2.nodes['node-a'].role, 'fenced');

  const zombieAttempt = sim.attemptZombieWrite();
  assert.equal(zombieAttempt.rejected, true);
  assert.equal(zombieAttempt.staleEpoch, 1);
  assert.equal(zombieAttempt.activeEpoch, 2);
});

test('celld simulateLoadBalancing models peer tunneling and bucket vs fleet hot-cell saturation', () => {
  const fleetRes = simulateLoadBalancing({
    nodes: 4,
    totalRps: 2000,
    hotCellSharePct: 25,
    routingMode: 'random-l7',
    durability: 'fleet'
  });
  assert.equal(fleetRes.proxiedRatio, 0.75);
  assert.equal(fleetRes.hotCell503Pct, 0);

  const bucketRes = simulateLoadBalancing({
    nodes: 4,
    totalRps: 2000,
    hotCellSharePct: 25,
    routingMode: 'random-l7',
    durability: 'bucket'
  });
  assert.ok(bucketRes.hotCell503Pct > 50, 'Bucket mode should saturate single hot cell at 500 req/s');
});

test('celld calculateFleetAccounting shows >98% cross-DO RPC reduction with credit leases and alarms', () => {
  const res = calculateFleetAccounting({
    fleetRps: 10000,
    activeCells: 2500,
    leaseBlockSize: 500,
    alarmIntervalSec: 30
  });
  assert.ok(res.naiveDroppedPct > 85, 'Naive central BillingDO should bottleneck at 10,000 req/s');
  assert.ok(res.rpcReductionPct > 98, `Expected >98% RPC reduction, got ${res.rpcReductionPct}%`);
});

test('celld index.html defines valid speculationrules for prefetching siblings and prerendering hub (learnings-bjx)', () => {
  const htmlPath = new URL('../site/celld/index.html', import.meta.url);
  const celldHtml = readFileSync(htmlPath, 'utf8');
  const specRulesMatch = celldHtml.match(/<script type="speculationrules">([\s\S]*?)<\/script>/);
  assert.ok(specRulesMatch, 'speculationrules script block must exist in site/celld/index.html');
  const rules = JSON.parse(specRulesMatch[1]);
  assert.ok(Array.isArray(rules.prefetch), 'rules must declare prefetch rules');
  assert.ok(Array.isArray(rules.prerender), 'rules must declare prerender rules');
  assert.equal(rules.prefetch[0].source, 'list');
  assert.deepEqual(rules.prefetch[0].urls, [
    '../neural-networks/',
    '../decision-models/',
    '../opt-chronicles/'
  ]);
  assert.equal(rules.prerender[0].source, 'list');
  assert.deepEqual(rules.prerender[0].urls, ['../index.html']);

  // CSP script-src must explicitly permit the inline speculationrules block by sha256 hash
  const scriptContent = specRulesMatch[1];
  const computedHash = createHash('sha256').update(scriptContent, 'utf8').digest('base64');
  const expectedCspToken = `'sha256-${computedHash}'`;
  const cspMatch = celldHtml.match(/<meta http-equiv="Content-Security-Policy" content="([^"]+)">/);
  assert.ok(cspMatch, 'CSP meta tag must exist in site/celld/index.html');
  assert.ok(
    cspMatch[1].includes(expectedCspToken),
    `CSP script-src must permit speculationrules inline hash ${expectedCspToken}, got CSP:\n${cspMatch[1]}`
  );
});

test('celld speculationrules script is permitted by CSP and initiates prefetch requests in browser (learnings-bjx)', async () => {
  const { serve } = await import('../scripts/serve.mjs');
  const { launch } = await import('./lib/cdp.mjs');
  const { server, url } = await serve();
  const prefetchedPaths = new Set();
  server.on('request', (req) => {
    const purpose = req.headers['sec-purpose'];
    if (purpose && purpose.includes('prefetch')) {
      prefetchedPaths.add(decodeURIComponent(new URL(req.url, 'http://localhost').pathname));
    }
  });

  let page;
  try {
    page = await launch();
    await page.send('Page.addScriptToEvaluateOnNewDocument', {
      source: `
        window.__cspViolations = [];
        document.addEventListener('securitypolicyviolation', (e) => {
          window.__cspViolations.push({
            violatedDirective: e.violatedDirective,
            effectiveDirective: e.effectiveDirective,
            blockedURI: e.blockedURI
          });
        });
      `
    });

    await page.goto(url + 'celld/index.html');
    await page.waitFor(() => document.querySelector('.celld-hero'),
      { label: 'celld hero to render', timeout: 60000 });

    // Wait for the speculation rules to fire all four prefetch requests. Prefetches are
    // best-effort and low-priority, so the old fixed 2000ms window flaked under fleet load
    // (1/460 on tree ffa8d9e, learnings-xjaf): poll the full expected SET with a labeled 60s
    // ceiling instead. The assertion stays real — a genuinely absent prefetch still fails when
    // the ceiling expires, and the failure names both the missing and the observed paths.
    const expectedPrefetches = ['/neural-networks/', '/decision-models/', '/opt-chronicles/', '/index.html'];
    const deadline = Date.now() + 60000;
    while (Date.now() < deadline && !expectedPrefetches.every((p) => prefetchedPaths.has(p))) {
      await new Promise((r) => setTimeout(r, 50));
    }
    const missingPrefetches = expectedPrefetches.filter((p) => !prefetchedPaths.has(p));
    assert.deepEqual(missingPrefetches, [],
      `timed out waiting for celld speculation-rules prefetches; missing ${JSON.stringify(missingPrefetches)}, observed ${JSON.stringify([...prefetchedPaths])}`);

    const violations = await page.evaluate(() => window.__cspViolations);
    assert.deepEqual(violations, [], 'speculationrules script must not trigger CSP violations');

    assert.ok(prefetchedPaths.has('/neural-networks/'), 'must prefetch /neural-networks/');
    assert.ok(prefetchedPaths.has('/decision-models/'), 'must prefetch /decision-models/');
    assert.ok(prefetchedPaths.has('/opt-chronicles/'), 'must prefetch /opt-chronicles/');
    assert.ok(prefetchedPaths.has('/index.html'), 'must prefetch /index.html');
  } finally {
    if (page) await page.close();
    server.close();
  }
});

// learnings-a3e9: the no-@media half of the guard below used to be one regex —
// /@media[^{]*\{[^}]*\.celld-grid/ — and `[^}]*` cannot cross a closing brace, so it only ever saw
// a media query whose FIRST rule targeted the grid. `@media (max-width: 860px) { .celld-hero
// { color: red } .celld-grid { grid-template-columns: 1fr } }` restores the viewport-owned collapse
// this test exists to prevent and left that regex green. The read is now structural: comments go
// first (so a comment that merely mentions @media is not a query), then every @media block is cut
// out by brace matching — from its at-keyword to its own closing brace, so nested rules and nested
// at-rules are INSIDE it — and no block may mention .celld-grid anywhere, in any position, at any
// depth. At-rules are case-insensitive in CSS, so the keyword is matched /i.
function mediaBlocksIn(css) {
  const source = css.replace(/\/\*[\s\S]*?\*\//g, '');
  const blocks = [];
  const at = /@media\b/gi;
  for (let found = at.exec(source); found; found = at.exec(source)) {
    const open = source.indexOf('{', found.index);
    // No opening brace: broken CSS. Read to the end of the file rather than skipping, so a
    // truncated query cannot hide the rules that follow it.
    if (open === -1) { blocks.push(source.slice(found.index)); break; }
    let depth = 0;
    let end = source.length;
    for (let i = open; i < source.length; i += 1) {
      if (source[i] === '{') depth += 1;
      else if (source[i] === '}') {
        depth -= 1;
        if (depth === 0) { end = i + 1; break; }
      }
    }
    blocks.push(source.slice(found.index, end));
  }
  return blocks;
}

// learnings-wn6: .celld-grid collapses to one column when ITS SPACE is too narrow, not when the
// viewport is. The page's reading column steals 190px plus a gap, so at a 900px viewport the grid
// was squeezing two columns into 511px while the viewport media query still said "wide". This pins
// the MECHANISM, not the pixel value: the section must declare itself a size container and the
// collapse must be a container query, so a future edit cannot quietly restore a viewport query.
test('celld grid collapse is a container query on its own section, not a viewport media query', () => {
  const css = readFileSync(new URL('../site/celld/celld.css', import.meta.url), 'utf8');
  const section = css.match(/\.celld-section\s*\{[^}]*\}/)?.[0] ?? '';

  assert.match(section, /container-type:\s*inline-size/, '.celld-section must be an inline-size container');
  assert.match(section, /container-name:\s*celld/, '.celld-section must name its container so the query cannot match an unrelated one');
  assert.match(
    css,
    /@container\s+celld\s*\(max-width:\s*\d+px\)\s*\{\s*\.celld-grid\s*\{[^}]*grid-template-columns:\s*1fr/,
    'the .celld-grid collapse must be a container query on the celld container'
  );
  const media = mediaBlocksIn(css);
  assert.deepEqual(
    media.filter((block) => block.includes('.celld-grid')),
    [],
    'no viewport media query may own the .celld-grid collapse (learnings-a3e9: every @media block in '
    + `the file was read whole — ${media.length} found — and one of them targets the grid)`
  );
});

test('the @media walker the grid guard depends on is itself pinned (learnings-a3e9)', () => {
  // A guard that reads blocks can also quietly read NONE, and `[]` passes an "no block mentions X"
  // assertion vacuously. So the walker is driven over a string that has the discriminating shape:
  // the grid rule is NOT first inside the query, and the query is followed by more CSS.
  const css = [
    '.celld-hero { color: red }',
    '/* a comment that mentions @media and .celld-grid is not a query */',
    '@media (max-width: 860px) {',
    '  .celld-hero { color: blue }',
    '  .celld-grid { grid-template-columns: 1fr }',
    '}',
    '.celld-grid { gap: 2rem }',
  ].join('\n');
  const blocks = mediaBlocksIn(css);
  assert.equal(blocks.length, 1, 'exactly the one real @media block is found (the comment is not one)');
  assert.match(blocks[0], /@media/, 'the block starts at its at-keyword');
  assert.equal(blocks[0].includes('.celld-grid'), true,
    'a .celld-grid rule that is NOT the first in the block is still inside it');
  assert.equal(blocks[0].includes('gap: 2rem'), false,
    'the block ends at its own closing brace, so the grid rule outside it is not swallowed');
  // Nested at-rules stay inside the media block, and an unterminated one reads to the end.
  assert.equal(mediaBlocksIn('@media screen { @supports (display: grid) { .celld-grid { color: red } } } .x { color: blue }')[0].includes('.x'), false);
  assert.equal(mediaBlocksIn('@media (max-width: 1px) { .celld-grid { color: red }').length, 1);
  assert.deepEqual(mediaBlocksIn('.celld-grid { color: red }'), [], 'no @media at all means no blocks, and the guard says so');
});
