// tests/nn-inp-artifacts.test.js
// The research/nn-inp probes write profiling artifacts to a path nobody else can influence
// (learnings-ilrx): a fresh 0700 directory, and files opened O_CREAT|O_EXCL|O_NOFOLLOW at 0600.
// These checks are the reason the hardening cannot be quietly undone by "just write to /tmp"
// later — the pre-created-path cases are the actual attack, so they are the cases tested.
import test from 'node:test';
import assert from 'node:assert/strict';
import { existsSync, lstatSync, mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, symlinkSync, writeFileSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { createArtifactDir, writeArtifact } from '../research/nn-inp/artifacts.mjs';

const scratch = () => mkdtempSync(path.join(os.tmpdir(), 'nn-inp-test-'));
const mode = target => lstatSync(target).mode & 0o777;

test('the artifact directory is private (0700) and not a predictable name', t => {
  const first = createArtifactDir();
  const second = createArtifactDir();
  t.after(() => { rmSync(first, { recursive: true, force: true }); rmSync(second, { recursive: true, force: true }); });
  assert.equal(mode(first), 0o700, 'directory mode is 0700');
  assert.notEqual(first, second, 'each run gets a fresh, unpredictable directory');
  assert.equal(path.dirname(first), os.tmpdir());
  assert.match(path.basename(first), /^nn-inp-[A-Za-z0-9]{6}$/, `mkdtemp suffix in ${path.basename(first)}`);
});

test('a written artifact is 0600 and round-trips its JSON', t => {
  const dir = createArtifactDir();
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const file = writeArtifact(dir, 'gd-idle.json', { nodes: [{ id: 1 }], samples: [1] });
  assert.equal(file, path.join(dir, 'gd-idle.json'));
  assert.equal(mode(file), 0o600, 'file mode is 0600');
  assert.deepEqual(JSON.parse(readFileSync(file, 'utf8')), { nodes: [{ id: 1 }], samples: [1] });
  assert.equal(writeArtifact(dir, 'note.txt', 'plain text'), path.join(dir, 'note.txt'), 'strings are written verbatim');
});

test('a pre-created file at the target name is refused, not overwritten', t => {
  const dir = createArtifactDir();
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const target = path.join(dir, 'gd-idle.json');
  writeFileSync(target, 'someone else was here');
  assert.throws(() => writeArtifact(dir, 'gd-idle.json', { nodes: [] }), { code: 'EEXIST' });
  assert.equal(readFileSync(target, 'utf8'), 'someone else was here', 'the existing bytes survive');
});

test('a symlink at the target name cannot redirect a write, even to a same-named file elsewhere', t => {
  const dir = createArtifactDir();
  const elsewhere = scratch();
  t.after(() => {
    rmSync(dir, { recursive: true, force: true });
    rmSync(elsewhere, { recursive: true, force: true });
  });
  const victim = path.join(elsewhere, 'gd-idle.json');
  writeFileSync(victim, 'untouched');
  symlinkSync(victim, path.join(dir, 'gd-idle.json'));
  // O_EXCL refuses the existing symlink (EEXIST); O_NOFOLLOW is the second line of defence if a
  // future edit drops O_EXCL, and reports ELOOP. Either way the write must not land on the victim.
  assert.throws(() => writeArtifact(dir, 'gd-idle.json', { nodes: [] }), error => ['EEXIST', 'ELOOP'].includes(error.code));
  assert.equal(readFileSync(victim, 'utf8'), 'untouched', 'the symlink target is not written');
});

test('a directory at the target name is refused and no artifact is left behind on failure', t => {
  const dir = createArtifactDir();
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  mkdirSync(path.join(dir, 'gd-idle.json'));
  assert.throws(() => writeArtifact(dir, 'gd-idle.json', { nodes: [] }), error => ['EEXIST', 'EISDIR'].includes(error.code));
  assert.ok(lstatSync(path.join(dir, 'gd-idle.json')).isDirectory(), 'the directory is still a directory, not a file');
});

test('the artifact name must be a bare filename, so a caller cannot write outside the directory', t => {
  const dir = createArtifactDir();
  const parent = path.dirname(dir);
  const escaped = path.join(parent, 'escaped-artifact.json');
  t.after(() => { rmSync(dir, { recursive: true, force: true }); rmSync(escaped, { force: true }); });
  const names = ['../escaped-artifact.json', ['nested', 'out.json'].join(path.sep), '.', '..', ''];
  for (const name of names) {
    assert.throws(() => writeArtifact(dir, name, 'x'), /bare filename/, `rejected ${JSON.stringify(name)}`);
  }
  assert.equal(existsSync(escaped), false, 'nothing was written outside the directory');
  assert.deepEqual(readdirSync(dir), [], 'the directory listing is unchanged');
});

test('cpu.mjs writes through the guarded helper, not to a fixed /tmp path', () => {
  // Source-level guard, because running the probe needs a browser and ~10s. The defect this bead
  // fixes was a literal fixed path in this file, so the literal is what the check pins.
  const source = readFileSync(new URL('../research/nn-inp/cpu.mjs', import.meta.url), 'utf8');
  assert.match(source, /createArtifactDir\(/, 'uses a private per-run directory');
  assert.match(source, /writeArtifact\(artifacts, /, 'writes through the exclusive-create helper');
  assert.doesNotMatch(source, /\/tmp\/nn-inp/);
  assert.doesNotMatch(source, /writeFileSync/);
});
