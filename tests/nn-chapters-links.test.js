// tests/nn-chapters-links.test.js — structure and link integrity for the neural-network chapters.
// The chapter list in site/neural-networks/chapters.js drives the hub and every page's sidebar,
// so these checks keep the list, the files on disk and the links between them in agreement.
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, statSync, readdirSync } from 'node:fs';
import { dirname, join, resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
import { CHAPTERS, PARTS, readyChapters, labRedirectFor, LAB_ANCHORS } from '../site/neural-networks/chapters.js';

const site = fileURLToPath(new URL('../site/', import.meta.url));
const nn = join(site, 'neural-networks');

function htmlFiles(dir) {
  const out = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const p = join(dir, entry.name);
    if (entry.isDirectory()) out.push(...htmlFiles(p));
    else if (entry.name.endsWith('.html')) out.push(p);
  }
  return out;
}

function localTargets(file) {
  const html = readFileSync(file, 'utf8');
  const targets = [];
  for (const m of html.matchAll(/\b(?:href|src)="([^"]+)"/g)) {
    if (/^(?:[a-z][a-z0-9+.-]*:|\/\/|#)/i.test(m[1])) continue; // external, data: and same-page anchors
    targets.push(m[1]);
  }
  return targets;
}

function resolveTarget(fromFile, url) {
  const path = url.split('#')[0].split('?')[0];
  let p = resolve(dirname(fromFile), path || '.');
  if (existsSync(p) && statSync(p).isDirectory()) p = join(p, 'index.html');
  return p;
}

test('all sixteen chapters are live and the hub embeds all six playgrounds', () => {
  assert.equal(CHAPTERS.length, 16);
  assert.equal(readyChapters().length, CHAPTERS.length);
  const hub = readFileSync(join(nn, 'index.html'), 'utf8');
  for (const widget of ['perceptron', 'landscape', 'backprop', 'convolution', 'probabilities', 'attention']) {
    assert.match(hub, new RegExp(`data-widget="${widget}"`));
  }
  for (const c of CHAPTERS.filter(c => c.slug !== 'gradient-descent.html')) {
    const html = readFileSync(join(nn, c.slug), 'utf8');
    assert.match(html, /data-widget="[a-z]+"/, `${c.slug} has an embedded experiment`);
    assert.match(html, /Sources and further reading/, `${c.slug} cites its sources`);
  }
});

test('every chapter has a known part, a unique slug and only known prerequisites', () => {
  const parts = new Set(PARTS.map((p) => p.id));
  const slugs = new Set();
  for (const c of CHAPTERS) {
    assert.ok(parts.has(c.part), `${c.slug} has unknown part ${c.part}`);
    assert.ok(!slugs.has(c.slug), `duplicate slug ${c.slug}`);
    slugs.add(c.slug);
    assert.ok(c.title && c.summary, `${c.slug} needs a title and summary`);
  }
  for (const c of CHAPTERS) for (const dep of c.buildsOn) assert.ok(slugs.has(dep), `${c.slug} builds on unknown ${dep}`);
});

test('every chapter not marked planned exists, declares itself and loads the shared navigation', () => {
  for (const c of readyChapters()) {
    const file = join(nn, c.slug);
    assert.ok(existsSync(file), `${c.slug} is not planned but has no file`);
    const html = readFileSync(file, 'utf8');
    assert.match(html, new RegExp(`data-chapter="${c.slug.replace('.', '\\.')}"`), `${c.slug} must declare data-chapter`);
    assert.match(html, /chapters\.js/, `${c.slug} must load chapters.js for its sidebar`);
    assert.match(html, /http-equiv="Content-Security-Policy"/, `${c.slug} needs the CSP meta`);
    assert.match(html, /<html lang="en">/, `${c.slug} needs lang="en"`);
    assert.match(html, /<main id="main"/, `${c.slug} needs <main id="main"> for the skip link`);
  }
});

test('every neural-network page has balanced MathML and no inline script or style (CSP)', () => {
  for (const file of htmlFiles(nn)) {
    const html = readFileSync(file, 'utf8');
    const rel = relative(site, file);
    assert.equal((html.match(/<math[\s>]/g) ?? []).length, (html.match(/<\/math>/g) ?? []).length, `${rel}: unbalanced <math>`);
    assert.doesNotMatch(html, /<script(?![^>]*\bsrc=)[^>]*>/, `${rel}: inline <script> is blocked by script-src 'self'`);
    assert.doesNotMatch(html, /<style[\s>]/, `${rel}: inline <style> is blocked by style-src 'self'`);
    assert.doesNotMatch(html, /\sstyle="/, `${rel}: style="" attributes are blocked by style-src 'self'`);
  }
});

test('chapter.css keeps display: block math on block formulas (Chrome otherwise stacks them)', () => {
  const css = readFileSync(join(nn, 'chapter.css'), 'utf8');
  const rule = css.match(/math\[display="block"\]\s*\{([^}]*)\}/);
  assert.ok(rule, 'chapter.css has a math[display="block"] rule');
  assert.match(rule[1], /display:\s*block math;/);
});

test('every local link, stylesheet and script in the neural-network pages resolves', () => {
  for (const file of htmlFiles(nn)) {
    for (const url of localTargets(file)) {
      const target = resolveTarget(file, url);
      assert.ok(existsSync(target), `${relative(site, file)} → ${url} (${relative(site, target)}) is missing`);
    }
  }
});

test('the Learnings hub links into the chapters and the workbench, and those links resolve', () => {
  const index = join(site, 'index.html');
  const nnLinks = localTargets(index).filter((u) => u.startsWith('neural-networks/'));
  assert.ok(nnLinks.length >= 1);
  for (const url of nnLinks) assert.ok(existsSync(resolveTarget(index, url)), `index.html → ${url} is missing`);
});

test('old workbench anchors redirect to lab.html; anything else stays on the hub', () => {
  assert.equal(labRedirectFor('#cnn-lab'), './lab.html#cnn-lab');
  assert.equal(labRedirectFor('block-builder'), './lab.html#block-builder');
  assert.equal(labRedirectFor('#three-ideas'), null);
  assert.equal(labRedirectFor(''), null);
  assert.equal(labRedirectFor(undefined), null);
  const lab = readFileSync(join(nn, 'lab.html'), 'utf8');
  for (const id of LAB_ANCHORS) assert.match(lab, new RegExp(`id="${id}"`), `lab.html has no #${id}`);
});

test('backends/webgpu.html listings are the exact shader and host code from engine.js, open and in order', async () => {
  const engine = await import('../site/neural-networks/engine.js');
  const html = readFileSync(join(nn, 'backends/webgpu.html'), 'utf8');
  assert.doesNotMatch(html, /<details/, 'webgpu.html must not hide code in <details>');
  const unescape = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  const shown = { wgsl: [], init: [], dispatch: [] };
  for (const m of html.matchAll(/<pre data-listing="(\w+)"><code>([\s\S]*?)<\/code><\/pre>/g)) shown[m[1]].push(...unescape(m[2]).split('\n'));
  const nonBlank = (lines) => lines.filter((l) => l.trim());
  const sources = { wgsl: engine.WGSL_GEMM_SHADER, init: engine.initWebGPUBackend.toString(), dispatch: engine.webgpuMatmulAsync.toString() };
  for (const [key, text] of Object.entries(sources)) {
    assert.deepEqual(nonBlank(shown[key]), nonBlank(text.split('\n')), `${key} listings drifted from engine.js`);
  }
});
