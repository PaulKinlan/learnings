import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { test } from 'node:test';
import { allTests, selectTests } from '../scripts/test-changed.mjs';

const root = new URL('../', import.meta.url);
const full = allTests();
const fullSet = new Set(full);
const choose = paths => selectTests(paths);

function subset(paths) {
  const picked = choose(paths);
  assert.ok(picked.length > 0, `empty selection for ${paths}`);
  for (const path of picked) assert.ok(fullSet.has(path), `fast-only test ${path}`);
  assert.ok(picked.includes('tests/lint.test.js'));
  return picked;
}

test('changed-test selection is always part of the full suite', () => {
  for (const testPath of full) assert.ok(subset([testPath]).includes(testPath));
  assert.deepEqual(subset(['README.md']), ['tests/lint.test.js']);
  assert.ok(subset(['site/decision-models/image-lab.js']).includes('tests/image-decision.test.js'));
  assert.ok(subset(['site/decision-models/image-lab.js']).includes('tests/input-order.test.js'));
  assert.deepEqual(subset(['site/style.css']), full);
  assert.deepEqual(subset(['tests/lib/cdp.mjs']), full);
  assert.deepEqual(subset(['package.json']), full);
  assert.deepEqual(subset(['.github/workflows/check.yml']), full);
  assert.deepEqual(subset([`site/${'zzyzx-'}${'never-referenced'}.${'js'}`]), full);
  assert.ok(subset(['README.md', 'tests/image-decision.test.js']).includes('tests/image-decision.test.js'));
});

test('each tracked source maps to a full-suite test or triggers full fallback', () => {
  const paths = execFileSync('git', ['ls-files', 'site/', 'scripts/'], { cwd: root, encoding: 'utf8' }).trim().split('\n');
  for (const path of paths) {
    const selected = subset([path]);
    assert.ok(selected.length === full.length || selected.some(testPath => testPath !== 'tests/lint.test.js'),
      `${path} silently maps to lint only`);
  }
});
