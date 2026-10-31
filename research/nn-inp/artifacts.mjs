// Artifact writes for the research/nn-inp probes, done without trusting the path.
//
// WHY THIS EXISTS (learnings-ilrx): cpu.mjs used to write its CPU profiles to a fixed
// `/tmp/nn-inp/<name>.json`. A predictable name inside a world-writable directory is the classic
// temp-file race: anything on the box can pre-create `/tmp/nn-inp` (or the files in it, or
// symlinks pointing somewhere else) and receive, redirect or block the write. The probe needs no
// trust at all, so it gets none: an unpredictable directory created 0700, and every file opened
// O_CREAT|O_EXCL|O_NOFOLLOW at 0600, which makes an existing path — regular file, symlink or FIFO —
// an error instead of a silent overwrite.
import { chmodSync, closeSync, constants, mkdtempSync, openSync, unlinkSync, writeSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';

/**
 * A fresh private directory for this run's artifacts.
 *
 * `mkdtemp` creates the directory atomically with mode 0700, so it cannot be pre-created by
 * anyone else. The explicit chmod is belt and braces for platforms whose mkdtemp honours a
 * looser umask; it can only tighten the mode.
 */
export function createArtifactDir(prefix = 'nn-inp-') {
  const dir = mkdtempSync(path.join(os.tmpdir(), prefix));
  chmodSync(dir, 0o700);
  return dir;
}

/**
 * Write `data` to `dir/name`, refusing anything already at that path.
 *
 * `name` must be a bare filename: a caller passing `../` or a path separator gets an error rather
 * than a write outside the directory. Returns the absolute path written.
 */
export function writeArtifact(dir, name, data) {
  if (typeof name !== 'string' || name === '' || name !== path.basename(name) || name === '.' || name === '..') {
    throw new Error(`artifact name must be a bare filename, got ${JSON.stringify(name)}`);
  }
  const file = path.join(dir, name);
  const fd = openSync(file, constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW, 0o600);
  try {
    writeSync(fd, typeof data === 'string' ? data : JSON.stringify(data));
  } catch (error) {
    closeSync(fd);
    // A half-written artifact is worse than none: the summary code would read a truncated profile.
    try { unlinkSync(file); } catch {}
    throw error;
  }
  closeSync(fd);
  return file;
}
