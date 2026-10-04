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

// Listings marked data-listing="key" in a backend chapter, unescaped and concatenated in page order.
function shownListings(html) {
  const unescape = (s) => s.replace(/&lt;/g, '<').replace(/&gt;/g, '>').replace(/&amp;/g, '&');
  const shown = {};
  for (const m of html.matchAll(/<pre data-listing="(\w+)"><code>([\s\S]*?)<\/code><\/pre>/g)) (shown[m[1]] ??= []).push(...unescape(m[2]).split('\n'));
  return shown;
}
const nonBlankLines = (lines) => lines.filter((l) => l.trim());

test('backends/javascript.html prints jsMatmul from engine.js, open and in order', async () => {
  const engine = await import('../site/neural-networks/engine.js');
  const html = readFileSync(join(nn, 'backends/javascript.html'), 'utf8');
  assert.doesNotMatch(html, /<details/, 'javascript.html must not hide code in <details>');
  const shown = shownListings(html);
  assert.deepEqual(Object.keys(shown), ['jsmatmul']);
  assert.deepEqual(nonBlankLines(shown.jsmatmul), nonBlankLines(engine.jsMatmul.toString().split('\n')), 'jsMatmul listings drifted from engine.js');
});

test('backends/webassembly.html prints the generator from engine.js and full WAT listings', async () => {
  const engine = await import('../site/neural-networks/engine.js');
  const html = readFileSync(join(nn, 'backends/webassembly.html'), 'utf8');
  assert.doesNotMatch(html, /<details/, 'webassembly.html must not hide code in <details>');
  const shown = shownListings(html);
  const engineSrc = readFileSync(join(nn, 'engine.js'), 'utf8');
  const sources = {
    uleb: engineSrc.match(/^function uleb128[\s\S]*?^}/m)[0],
    generator: engine.buildWasmGemmBytes.toString(),
    backend: engine.getWasmBackend.toString(),
  };
  for (const [key, text] of Object.entries(sources)) {
    assert.deepEqual(nonBlankLines(shown[key] ?? []), nonBlankLines(text.split('\n')), `${key} listings drifted from engine.js`);
  }

  // Verify WAT listings are present and unhidden
  const watText = (shown['wat'] ?? []).join('\n');
  assert.match(watText, /export "gemm_f32"/, 'WAT listing must export gemm_f32');
  assert.match(watText, /export "memory"/, 'WAT listing must export memory');
  assert.match(watText, /f32\.load/, 'WAT listing must contain f32.load');
  assert.match(watText, /f32\.mul/, 'WAT listing must contain f32.mul');
  assert.match(watText, /f32\.add/, 'WAT listing must contain f32.add');
  assert.match(watText, /f32\.store/, 'WAT listing must contain f32.store');

  const simdText = (shown['simd_wat'] ?? []).join('\n');
  assert.match(simdText, /func \$axpy_f32x4/, 'SIMD WAT listing must declare $axpy_f32x4');
  assert.match(simdText, /f32x4\.splat/, 'SIMD WAT listing must use f32x4.splat');
  assert.match(simdText, /v128\.load/, 'SIMD WAT listing must use v128.load');
  assert.match(simdText, /f32x4\.mul/, 'SIMD WAT listing must use f32x4.mul');
  assert.match(simdText, /f32x4\.add/, 'SIMD WAT listing must use f32x4.add');
  assert.match(simdText, /v128\.store/, 'SIMD WAT listing must use v128.store');
});

test('backends/litert.html prints quantizers from math.js, open and in order', async () => {
  const math = await import('../site/neural-networks/math.js');
  const html = readFileSync(join(nn, 'backends/litert.html'), 'utf8');
  assert.doesNotMatch(html, /<details/, 'litert.html must not hide code in <details>');
  const shown = shownListings(html);
  assert.deepEqual(Object.keys(shown).sort(), ['affine', 'symmetric'].sort());
  assert.deepEqual(nonBlankLines(shown.affine), nonBlankLines(math.quantizeAffineInt8.toString().split('\n')), 'quantizeAffineInt8 drifted from math.js');
  assert.deepEqual(nonBlankLines(shown.symmetric), nonBlankLines(math.quantizeSymmetricInt8.toString().split('\n')), 'quantizeSymmetricInt8 drifted from math.js');
});

test('backends/pytorch.html presents autograd graph and stride derivation openly', async () => {
  const html = readFileSync(join(nn, 'backends/pytorch.html'), 'utf8');
  assert.doesNotMatch(html, /<details/, 'pytorch.html must not hide code in <details>');
  assert.match(html, /UntypedStorage/);
  assert.match(html, /PowBackward0/);
  assert.match(html, /MvBackward0/);
  assert.match(html, /torch\.compile/);
  assert.match(html, /graph_breaks/);
});

