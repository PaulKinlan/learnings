import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { measureTree } from './lib/worktree.mjs';

const git = (dir, ...args) => execFileSync('git', args, { cwd: dir, encoding: 'utf8' }).trim();
const makeRepo = () => {
  const dir = mkdtempSync(join(tmpdir(), 'worktree-measure-'));
  git(dir, 'init', '-q');
  git(dir, 'config', 'user.email', 'gate@example.com');
  git(dir, 'config', 'user.name', 'gate');
  writeFileSync(join(dir, 'tracked.txt'), 'one\n');
  git(dir, 'add', '-A');
  git(dir, 'commit', '-q', '-m', 'init');
  return dir;
};

test('a clean worktree measures the committed tree, not just a HEAD name', () => {
  const dir = makeRepo();
  try {
    const m = measureTree(dir);
    assert.equal(m.dirty, false);
    assert.equal(m.tree, m.headTree);
    assert.match(m.commit, /^[0-9a-f]{12}$/);
    assert.match(m.tree, /^[0-9a-f]{40}$/);
    assert.equal(m.tree, git(dir, 'rev-parse', 'HEAD^{tree}'));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('an uncommitted tracked edit produces a different measured tree', () => {
  const dir = makeRepo();
  try {
    const before = measureTree(dir);
    writeFileSync(join(dir, 'tracked.txt'), 'two\n');
    const after = measureTree(dir);
    assert.equal(after.dirty, true);
    assert.notEqual(after.tree, before.tree);
    assert.equal(after.headTree, before.headTree, 'committed tree is unchanged by a working-tree edit');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('untracked files are part of the measured tree', () => {
  const dir = makeRepo();
  try {
    const before = measureTree(dir);
    writeFileSync(join(dir, 'untracked.txt'), 'new content\n');
    const after = measureTree(dir);
    assert.equal(after.dirty, true);
    assert.notEqual(after.tree, before.tree);
    // Measuring twice over identical content agrees: the id is content-addressed.
    assert.equal(measureTree(dir).tree, after.tree);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('once committed, the measured tree equals git write-tree for the commit', () => {
  const dir = makeRepo();
  try {
    writeFileSync(join(dir, 'tracked.txt'), 'two\n');
    git(dir, 'commit', '-q', '-am', 'second');
    const m = measureTree(dir);
    assert.equal(m.dirty, false);
    assert.equal(m.tree, git(dir, 'rev-parse', 'HEAD^{tree}'));
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

test('measuring never touches the real index or working tree state', () => {
  const dir = makeRepo();
  try {
    writeFileSync(join(dir, 'tracked.txt'), 'unstaged edit\n');
    const statusBefore = git(dir, 'status', '--porcelain');
    const stagedBefore = git(dir, 'diff', '--cached', '--name-only');
    measureTree(dir);
    assert.equal(git(dir, 'diff', '--cached', '--name-only'), stagedBefore, 'staged set must stay empty');
    assert.equal(git(dir, 'status', '--porcelain'), statusBefore, 'worktree status must be unchanged');
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

// Wiring guard: the browser gate must pin the measured tree in its receipt and name
// its evidence directory for that tree, so a PASS on a dirty worktree cannot be
// mistaken for a PASS on HEAD (learnings-j38).
test('browser receipt names the measured tree, not only HEAD', () => {
  const browser = readFileSync(new URL('./browser.mjs', import.meta.url), 'utf8');
  assert.match(browser, /import \{measureTree\} from '\.\/lib\/worktree\.mjs';/);
  assert.match(browser, /const measured=measureTree\(repoRoot\);/);
  assert.match(browser, /measured\.tree\.slice\(0,12\)/, 'evidence directory must be named for the measured tree');
  assert.doesNotMatch(browser, /\$\{basename\(repoRoot\)\}-\$\{commit\}/, 'evidence directory must not be named for HEAD');
  for (const field of ['head:measured.head', 'headTree:measured.headTree', 'tree:measured.tree', 'dirty:measured.dirty']) {
    assert.match(browser, new RegExp(field.replace(/\./g, '\\.')), `receipt must record ${field}`);
  }
  assert.match(browser, /receipt names the tree it measured/);
});
