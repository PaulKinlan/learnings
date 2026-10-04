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

test('backends/webassembly.html prints the generator from engine.js and every module byte with its instruction', async () => {
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

  // Hex rows: "offset  bytes │ meaning". Offsets must run on without gaps.
  const OPS = { '02': 'block', '03': 'loop', '0b': 'end', '0c': 'br', '0d': 'br_if', '20': 'local.get', '21': 'local.set', '2a': 'f32.load', '38': 'f32.store', '41': 'i32.const', '43': 'f32.const', '4f': 'i32.ge_u', '6a': 'i32.add', '6c': 'i32.mul', '71': 'i32.and', '74': 'i32.shl', '92': 'f32.add', '94': 'f32.mul', 'fd 00': 'v128.load', 'fd 0b': 'v128.store', 'fd 13': 'f32x4.splat', 'fd e4 01': 'f32x4.add', 'fd e6 01': 'f32x4.mul' };
  const parse = (key, start) => {
    let at = start; const bytes = [];
    for (const line of shown[key]) {
      const m = line.match(/^([0-9a-f]{4})  ([0-9a-f ]+?)\s*│ (.*)$/);
      assert.ok(m, `${key}: unparsable row ${line}`);
      assert.equal(parseInt(m[1], 16), at, `${key}: row offset ${m[1]}`);
      const hex = m[2].split(' ');
      if (key !== 'wasmheader' && !m[3].startsWith('(local')) {
        const op = [3, 2, 1].map((n) => hex.slice(0, n).join(' ')).find((k) => OPS[k]);
        assert.equal(m[3].trim().split(/\s/)[0], OPS[op], `${key} @${m[1]}: ${m[2]} is not ${m[3].trim()}`);
      }
      bytes.push(...hex.map((h) => parseInt(h, 16)));
      at += hex.length;
    }
    return bytes;
  };
  const module = parse('wasmheader', 0);
  module.push(...parse('wasmcode', module.length));
  assert.deepEqual(module, [...engine.buildWasmGemmBytes()], 'hex tables drifted from buildWasmGemmBytes()');

  // The SIMD axpy listing is not in engine.js: assemble its bytes and check it against scalar f32 math.
  const body = parse('simd', 0);
  const leb = (n) => { const o = []; do { let b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; o.push(b); } while (n); return o; };
  const code = [1, ...leb(body.length), ...body];
  const bytes = new Uint8Array([0, 0x61, 0x73, 0x6d, 1, 0, 0, 0,
    1, 8, 1, 0x60, 4, 0x7f, 0x7f, 0x7d, 0x7f, 0,       // (i32 cPtr, i32 bPtr, f32 a, i32 n) -> ()
    3, 2, 1, 0, 5, 3, 1, 0, 1,
    7, 14, 2, 4, 0x61, 0x78, 0x70, 0x79, 0, 0, 3, 0x6d, 0x65, 0x6d, 2, 0,
    10, ...leb(code.length), ...code]);
  assert.ok(WebAssembly.validate(bytes), 'SIMD listing is not a valid function body');
  const { axpy, mem } = new WebAssembly.Instance(new WebAssembly.Module(bytes)).exports;
  const a = Math.fround(1.7);
  for (let n = 0; n <= 13; n++) {
    const f = new Float32Array(mem.buffer);
    f.fill(0);
    const C = Float32Array.from({ length: n }, (_, i) => i * 0.37 - 1), B = Float32Array.from({ length: n }, (_, i) => 2 - i * 0.11);
    f.set(C, 0); f.set(B, 64);
    axpy(0, 256, a, n);
    assert.deepEqual([...f.subarray(0, n + 1)], [...C.map((c, j) => Math.fround(c + Math.fround(a * B[j]))), 0], `SIMD axpy wrong for n = ${n}`);
  }
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

