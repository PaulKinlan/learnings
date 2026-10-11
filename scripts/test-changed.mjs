#!/usr/bin/env node
// Conservative affected-test selection. The full gate is `node --test tests/*.test.js`.
import { execFileSync, spawnSync } from 'node:child_process';
import { readFileSync, readdirSync } from 'node:fs';
import { basename, join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const root = fileURLToPath(new URL('../', import.meta.url));
const git = (...args) => execFileSync('git', args, { cwd: root, encoding: 'utf8' }).trim();
export const allTests = () => readdirSync(join(root, 'tests'))
  .filter(name => name.endsWith('.test.js'))
  .map(name => `tests/${name}`).sort();

function isShared(path) {
  return path === 'site/style.css' || path === 'scripts/serve.mjs' || path === 'scripts/test-changed.mjs' ||
    path.startsWith('tests/lib/') || path === 'package.json' ||
    path === 'package-lock.json' || path.startsWith('.github/workflows/');
}

// Accept only literal file references; basename is deliberately conservative and
// may include extra tests. Unknown sources always take the full fallback.
export function selectTests(changed, tests = allTests(), contents = path => readFileSync(join(root, path), 'utf8')) {
  const selected = new Set(['tests/lint.test.js']);
  const full = () => [...tests];
  const texts = new Map(tests.map(path => [path, contents(path)]));
  for (const path of changed) {
    if (isShared(path)) return full();
    if (tests.includes(path)) { selected.add(path); continue; }
    if (path.startsWith('site/') || path.startsWith('scripts/')) {
      const name = basename(path);
      const matches = tests.filter(test => texts.get(test).includes(path) || texts.get(test).includes(name));
      if (!matches.length) return full();
      for (const test of matches) selected.add(test);
      // These tests enumerate directories instead of spelling out every filename.
      if (path.startsWith('site/') && /\.m?js$/.test(path) && !path.startsWith('site/vendor/')) selected.add('tests/input-order.test.js');
      if (path.startsWith('site/neural-networks/') && path.endsWith('.html')) selected.add('tests/nn-chapters-links.test.js');
      continue;
    }
    // Markdown-only documentation is safe to lint; other unclassified changes
    // (including deleted/renamed tests and gate configuration) require full coverage.
    if (path.endsWith('.md')) continue;
    return full();
  }
  return tests.filter(test => selected.has(test));
}

export function changedFiles(base) {
  // With rename detection Git reports only the destination, hiding tests that
  // still reference the deleted source. Report both paths instead.
  const diff = git('diff', '--name-only', '--no-renames', '--diff-filter=ACMRTUXBD', base, '--');
  const untracked = git('ls-files', '--others', '--exclude-standard');
  return [...new Set([diff, untracked].filter(Boolean).flatMap(lines => lines.split('\n')))].sort();
}

function defaultBase() {
  try { return { ref: git('merge-base', 'origin/main', 'HEAD'), fallback: false }; }
  catch { return { ref: 'HEAD~1', fallback: true }; }
}

const invoked = process.argv[1] && pathToFileURL(process.argv[1]).href === import.meta.url;
if (invoked) {
  const args = process.argv.slice(2);
  let base;
  let run = false;
  for (let i = 0; i < args.length; i++) {
    if (args[i] === '--base' && args[i + 1]) base = args[++i];
    else if (args[i] === '--run') run = true;
    else { console.error('Usage: node scripts/test-changed.mjs [--base <ref>] [--run]'); process.exit(2); }
  }
  try {
    const chosenBase = base ? { ref: base, fallback: false } : defaultBase();
    // HEAD~1 is the documented fallback ref, but it may omit earlier commits
    // on a multi-commit branch. Select the full suite rather than a false green.
    const tests = chosenBase.fallback ? allTests() : selectTests(changedFiles(chosenBase.ref));
    if (!run) console.log(tests.join('\n'));
    else {
      const result = spawnSync(process.execPath, ['--test', ...tests], { cwd: root, stdio: 'inherit' });
      process.exit(result.status ?? 1);
    }
  } catch (error) { console.error(error.message); process.exit(1); }
}
