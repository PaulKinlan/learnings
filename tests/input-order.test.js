import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, readdirSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const site = fileURLToPath(new URL('../site/', import.meta.url));

function* scripts(dir) {
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === 'vendor') continue;
    const path = join(dir, entry.name);
    if (entry.isDirectory()) yield* scripts(path);
    else if (/\.(?:js|mjs)$/.test(entry.name)) yield path;
  }
}

// Check the two input-building forms used by site/: helper attribute objects and
// direct createElement calls. Keep source order: Object.entries(attrs) sets attributes in order.
function orderingErrors(source) {
  const errors = [];
  const geometry = ['min', 'max', 'step'];
  const check = (fields, location) => {
    const value = fields.findIndex(f => f === 'value');
    if (value < 0) return;
    for (const name of geometry) {
      const index = fields.findIndex(f => f === name);
      if (index > value) errors.push(`${location}: ${name} follows value`);
    }
  };

  // Helper calls with a literal attribute object (h('input', ..., { ... })).
  for (const match of source.matchAll(/\b(?:h|el)\s*\(\s*['"]input['"][^\n]*?\{([^{}]*)\}/g)) {
    const attrs = match[1];
    const fields = [...attrs.matchAll(/(?:^|,)\s*(['"]?)(min|max|step|value)\1\s*(?=:|,|$)/g)].map(m => m[2]);
    check(fields, `input helper at line ${source.slice(0, match.index).split('\n').length}`);
  }

  // Direct constructions: only the statements on this element up to its first use
  // are relevant. Both property writes and setAttribute are accepted.
  for (const match of source.matchAll(/\b(?:const|let)\s+(\w+)\s*=\s*(?:document\.)?createElement\s*\(\s*['"]input['"]\s*\)/g)) {
    const name = match[1];
    const tail = source.slice(match.index + match[0].length).split(/\b(?:return|append|replaceChildren)\s*\(/, 1)[0];
    const writes = [...tail.matchAll(new RegExp(`\\b${name}\\.(min|max|step|value)\\s*=|\\b${name}\\.setAttribute\\s*\\(\\s*['"](min|max|step|value)['"]`, 'g'))]
      .map(m => m[1] || m[2]);
    check(writes, `createElement input at line ${source.slice(0, match.index).split('\n').length}`);
  }
  return errors;
}

test('JS-built inputs set min/max/step before value', () => {
  const errors = [...scripts(site)].flatMap(path => {
    const input = path === join(site, 'neural-networks/widgets.js') && process.env.INPUT_ORDER_WIDGETS_COPY
      ? process.env.INPUT_ORDER_WIDGETS_COPY : path;
    return orderingErrors(readFileSync(input, 'utf8')).map(error => `${path}: ${error}`);
  });
  assert.deepEqual(errors, []);
});

test('input-order guard rejects a reverted fractional slider and direct assignments', () => {
  const widgets = readFileSync(join(site, 'neural-networks/widgets.js'), 'utf8');
  const reverted = widgets.replace('min, max, step, value}', 'min, max, value, step}');
  assert.notEqual(reverted, widgets, 'slider construction must remain covered');
  assert.match(orderingErrors(reverted).join('\n'), /step follows value/);
  assert.match(orderingErrors("const input = document.createElement('input'); input.value = '0.5'; input.step = '0.1';").join('\n'), /step follows value/);
  assert.match(orderingErrors("const input = document.createElement('input'); input.setAttribute('value', '0.5'); input.setAttribute('min', '0');").join('\n'), /min follows value/);
});
