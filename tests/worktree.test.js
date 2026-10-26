import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { mkdtempSync, rmSync, mkdirSync, writeFileSync, readFileSync, symlinkSync, cpSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { measureTree } from './lib/worktree.mjs';

const git = (dir, ...args) => execFileSync('git', args, { cwd: dir, encoding: 'utf8' }).trim();

// learnings-kcu: the fixture repo is init+committed ONCE; every test copies it (a plain
// filesystem copy costs milliseconds and no git subprocesses, where an init/add/commit cycle
// costs ~0.3s each). Copies keep the tests independent: each starts from the same clean commit
// (objects, index, working tree and identity config), and a test's edits and commits stay
// inside its own throwaway checkout.
let template = null;
before(() => {
  template = mkdtempSync(join(tmpdir(), 'worktree-template-'));
  git(template, 'init', '-q');
  git(template, 'config', 'user.email', 'gate@example.com');
  git(template, 'config', 'user.name', 'gate');
  writeFileSync(join(template, 'tracked.txt'), 'one\n');
  git(template, 'add', '-A');
  git(template, 'commit', '-q', '-m', 'init');
});
after(() => {
  if (template) rmSync(template, { recursive: true, force: true });
});

const makeRepo = () => {
  const dir = mkdtempSync(join(tmpdir(), 'worktree-measure-'));
  cpSync(template, dir, { recursive: true });
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

// Regression for learnings-2pr: add -A fingerprints a symlink's PATH, but the serve
// script reads through readFile and serves the TARGET's bytes. Overwriting the target
// must change the measured tree (and the receipt manifest must name the link).
test('a symlinked source file is measured by its target bytes, not its link path', () => {
  const dir = makeRepo();
  const outside = mkdtempSync(join(tmpdir(), 'link-target-'));
  try {
    const target = join(outside, 'target.html');
    writeFileSync(target, 'BEFORE: served bytes\n');
    mkdirSync(join(dir, 'site'));
    symlinkSync(target, join(dir, 'site', 'index.html'));
    git(dir, 'add', '-A');
    git(dir, 'commit', '-q', '-m', 'link');
    const before = measureTree(dir);
    assert.equal(before.dirty, true, 'a resolved symlink never matches its literal committed link entry');
    assert.deepEqual(before.links.map(l => l.path), ['site/index.html']);
    assert.equal(before.links[0].kind, 'file');
    writeFileSync(target, 'AFTER: served bytes changed\n');
    const after = measureTree(dir);
    assert.equal(after.dirty, true);
    assert.notEqual(after.tree, before.tree, 'changing the target bytes must change the measured tree');
    assert.equal(measureTree(dir).tree, after.tree, 'the id stays content-addressed: identical bytes agree');
    // Retargeting to a different path holding identical bytes must NOT change the id.
    const target2 = join(outside, 'target2.html');
    writeFileSync(target2, 'AFTER: served bytes changed\n');
    rmSync(join(dir, 'site', 'index.html')); // removes the link, never its target
    symlinkSync(target2, join(dir, 'site', 'index.html'));
    assert.equal(measureTree(dir).tree, after.tree, 'same served bytes through a new target measure the same');
  } finally { rmSync(dir, { recursive: true, force: true }); rmSync(outside, { recursive: true, force: true }); }
});

test('an untracked symlink is part of the measured tree, and a dangling one is deterministic', () => {
  const dir = makeRepo();
  try {
    const before = measureTree(dir);
    symlinkSync(join(dir, 'missing.html'), join(dir, 'dangling.html'));
    const dangling = measureTree(dir);
    assert.equal(dangling.dirty, true);
    assert.notEqual(dangling.tree, before.tree);
    assert.equal(measureTree(dir).tree, dangling.tree, 'an unreadable link still measures deterministically');
    assert.deepEqual(dangling.links.map(l => [l.path, l.kind]), [['dangling.html', 'unreadable']]);
  } finally { rmSync(dir, { recursive: true, force: true }); }
});

// Regression for learnings-tuj: the walk used to stop at MAX_DEPTH=16, so a deep
// file behind a directory link changed served bytes without moving the measured
// tree. Mirror the reviewer's repro: site/assets -> external dir with visible.html
// and d/…/d/deep.html, 17 nested directories down.
test('a deep file behind a directory symlink is measured, not truncated', () => {
  const dir = makeRepo();
  const outside = mkdtempSync(join(tmpdir(), 'link-deep-'));
  try {
    let deep = outside;
    for (let i = 0; i < 17; i += 1) {
      deep = join(deep, 'd');
      mkdirSync(deep);
    }
    const deepFile = join(deep, 'deep.html');
    writeFileSync(join(outside, 'visible.html'), 'visible\n');
    writeFileSync(deepFile, 'BEFORE\n');
    mkdirSync(join(dir, 'site'));
    symlinkSync(outside, join(dir, 'site', 'assets'));
    const before = measureTree(dir);
    assert.equal(readFileSync(join(dir, 'site', 'assets', 'd', ...Array(16).fill('d'), 'deep.html'), 'utf8'), 'BEFORE\n');
    writeFileSync(deepFile, 'AFTER\n');
    const after = measureTree(dir);
    assert.notEqual(after.tree, before.tree, 'a change 17 directories deep must move the measured tree');
  } finally { rmSync(dir, { recursive: true, force: true }); rmSync(outside, { recursive: true, force: true }); }
});

// The uncapped walk must fail loudly on a link loop instead of spinning forever.
test('a link cycle is an explicit error, not a truncation or a hang', () => {
  const dir = makeRepo();
  const outside = mkdtempSync(join(tmpdir(), 'link-cycle-'));
  try {
    writeFileSync(join(outside, 'visible.html'), 'visible\n');
    symlinkSync(outside, join(outside, 'loop'));
    mkdirSync(join(dir, 'site'));
    symlinkSync(outside, join(dir, 'site', 'assets'));
    assert.throws(() => measureTree(dir), /gate-receipt: symlink cycle at site\/assets\/loop/);
  } finally { rmSync(dir, { recursive: true, force: true }); rmSync(outside, { recursive: true, force: true }); }
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
  for (const field of ['head:measured.head', 'headTree:measured.headTree', 'tree:measured.tree', 'dirty:measured.dirty', 'links:measured.links']) {
    assert.match(browser, new RegExp(field.replace(/\./g, '\\.')), `receipt must record ${field}`);
  }
  assert.match(browser, /receipt names the tree it measured/);
});
