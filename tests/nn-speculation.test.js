import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { CHAPTERS } from '../site/neural-networks/chapters.js';

// Speculation Rules on the chapter pages (learnings-q6p). Two things can silently rot here and both
// are cheap to check without a browser:
//   1. the inline block's sha256 must match the CSP script-src token, or the browser BLOCKS the block
//      and the prefetch never happens - the page still works, so nothing else would notice;
//   2. the prefetched/prerendered URLs must be the chapter's real neighbours, or the rule points at a
//      page the reader will not go to (and a renamed chapter turns into a 404 prefetch).

const SITE = fileURLToPath(new URL('../site/neural-networks/', import.meta.url));

function readPage(file) {
  const html = readFileSync(path.join(SITE, file), 'utf8');
  const block = html.match(/<script type="speculationrules">([\s\S]*?)<\/script>/);
  assert.ok(block, `${file} must declare a speculationrules script block`);
  const csp = html.match(/<meta http-equiv="Content-Security-Policy" content="([^"]+)">/);
  assert.ok(csp, `${file} must declare a CSP meta tag`);
  return { html, scriptContent: block[1], csp: csp[1] };
}

function cspTokenFor(scriptContent) {
  return `'sha256-${createHash('sha256').update(scriptContent, 'utf8').digest('base64')}'`;
}

// The chapter order the site itself uses, so a reordered curriculum re-points the rules instead of
// quietly leaving them aimed at the wrong page.
function neighboursOf(file) {
  const files = CHAPTERS.filter((c) => c.slug.endsWith('.html')).map((c) => c.slug);
  const i = files.indexOf(file);
  assert.notEqual(i, -1, `${file} must appear in the chapter list`);
  return {
    prev: i > 0 ? path.basename(files[i - 1]) : null,
    next: i + 1 < files.length ? path.basename(files[i + 1]) : null,
  };
}

// One flattened URL list per kind, so assertions do not depend on how many rule objects the block
// happens to use.
function urlsIn(rules, kind) {
  const entries = rules[kind] ?? [];
  return entries.flatMap((rule) => rule.urls ?? []);
}

for (const file of ['activations.html', 'architectures.html']) {
  test(`${file}: speculation rules prefetch the way back, prerender the next chapter, and the CSP hash matches (learnings-q6p)`, () => {
    const { scriptContent, csp } = readPage(file);
    const rules = JSON.parse(scriptContent);
    const { prev, next } = neighboursOf(file);

    const prefetched = urlsIn(rules, 'prefetch');
    const prerendered = urlsIn(rules, 'prerender');
    assert.ok(prefetched.length > 0, 'the block must prefetch at least one URL');
    assert.deepEqual(prerendered, [next], `${file} must prerender exactly its next chapter (${next})`);
    assert.ok(prefetched.includes('index.html'), 'the hub must be prefetched - it is the other exit from a chapter');
    if (prev) assert.ok(prefetched.includes(prev), `${file} must prefetch the previous chapter (${prev})`);

    // A URL that does not exist on disk is a prefetch of a 404, which the browser caches as a failure
    // and never retries for the transition.
    for (const url of [...prefetched, ...prerendered]) {
      if (/^[a-z]+:/i.test(url)) continue;
      assert.ok(existsSync(path.join(SITE, url)), `${file} speculation URL ${url} must exist next to the page`);
    }

    const token = cspTokenFor(scriptContent);
    assert.ok(
      csp.includes(token),
      `${file}: CSP script-src must carry ${token} for the inline speculationrules block, got: ${csp}`
    );
  });
}

test('every chapter page that ships a speculationrules block authorizes that exact block in its CSP (learnings-q6p)', () => {
  // Covers pages this bead did not touch: adding rules elsewhere without the hash makes the block
  // silently inert, and this test is where that shows up.
  const pages = CHAPTERS.filter((c) => c.slug.endsWith('.html')).map((c) => path.basename(c.slug));
  const withRules = pages.filter((file) => {
    const filePath = path.join(SITE, file);
    return existsSync(filePath) && readFileSync(filePath, 'utf8').includes('<script type="speculationrules">');
  });
  assert.ok(withRules.length >= 2, 'at least the two chapter pages from learnings-q6p must declare rules');
  for (const file of withRules) {
    const { scriptContent, csp } = readPage(file);
    JSON.parse(scriptContent); // must stay valid JSON, or Chromium drops the whole block
    assert.ok(csp.includes(cspTokenFor(scriptContent)), `${file}: CSP must authorize its inline speculationrules hash`);
  }
});
