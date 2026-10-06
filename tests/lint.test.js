import { test } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Extensions the walk scans. Pinned by the 'file walk reaches .mjs sinks' test:
// dropping '.mjs' here fails the suite.
const SCAN_EXTENSIONS = ['.js', '.mjs'];

// The only `notice.innerHTML` form this allowance admits is a constant-only,
// single-quoted, no-interpolation assignment, and only inside image-lab.js.
const NOTICE_CONSTANT_LINE = /notice\.innerHTML\s*=\s*'[^']*'\s*;?\s*$/;

function findFiles(dir, exts) {
  let results = [];
  const list = fs.readdirSync(dir);
  for (const file of list) {
    const filePath = path.join(dir, file);
    const stat = fs.statSync(filePath);
    if (stat && stat.isDirectory()) {
      results = results.concat(findFiles(filePath, exts));
    } else if (exts.some(ext => file.endsWith(ext))) {
      results.push(filePath);
    }
  }
  return results;
}

// Blanks line comments, block comments and string/template literals while keeping
// newlines aligned, so per-line sink matching and `file:line` reporting stay exact.
// The only literal kept verbatim is a bare 'innerHTML'/"innerHTML"/`innerHTML`
// property name, because `el["innerHTML"] = x` is itself a real sink.
function stripCommentsAndStrings(source) {
  let out = '';
  let i = 0;
  const n = source.length;

  while (i < n) {
    const ch = source[i];

    if (ch === '/' && source[i + 1] === '/') {
      while (i < n && source[i] !== '\n') { out += ' '; i++; }
      continue;
    }

    if (ch === '/' && source[i + 1] === '*') {
      out += '  '; i += 2;
      while (i < n) {
        if (source[i] === '*' && source[i + 1] === '/') {
          out += '  '; i += 2; break;
        }
        out += (source[i] === '\n') ? '\n' : ' ';
        i++;
      }
      continue;
    }

    if (ch === '"' || ch === "'") {
      const start = i;
      const quote = ch;
      i++;
      let content = '';
      while (i < n && source[i] !== quote) {
        if (source[i] === '\\' && i + 1 < n) { i += 2; continue; }
        content += source[i];
        i++;
      }
      if (i < n) i++; // closing quote
      out += (content === 'innerHTML') ? source.slice(start, i) : ' '.repeat(i - start);
      continue;
    }

    if (ch === '`') {
      const start = i;
      i++;
      let content = '';
      let hasInterpolation = false;
      while (i < n && source[i] !== '`') {
        if (source[i] === '$' && source[i + 1] === '{') hasInterpolation = true;
        content += source[i];
        i++;
      }
      if (i < n) i++; // closing backtick
      if (!hasInterpolation && content === 'innerHTML') {
        out += source.slice(start, i);
      } else {
        for (let j = start; j < i; j++) out += (source[j] === '\n') ? '\n' : ' ';
      }
      continue;
    }

    out += ch;
    i++;
  }

  return out;
}

function isSinkLine(line) {
  return /\.innerHTML\s*\+?=/.test(line) ||
         /\[['"`]innerHTML['"`]\]\s*\+?=/.test(line) ||
         /\.insertAdjacentHTML\s*\(/.test(line) ||
         /document\.write\s*\(/.test(line);
}

function scanFile(file) {
  const failures = [];
  const raw = fs.readFileSync(file, 'utf8');
  const lines = raw.split('\n');
  const stripped = stripCommentsAndStrings(raw).split('\n');

  lines.forEach((line, index) => {
    // Allow clearing a container (empty string or a single space), with or
    // without whitespace around `=`.
    if (/\.innerHTML\s*=\s*["']\s*["']/.test(line)) return;

    // Allow the known constant-only notice in image-lab.js. Scoped by path, so a
    // `notice.innerHTML = userInput` sink in any other file still fails.
    if (file.endsWith('/decision-models/image-lab.js') && NOTICE_CONSTANT_LINE.test(line.trim())) return;

    if (isSinkLine(stripped[index])) {
      failures.push(`File ${file}:${index + 1} uses innerHTML sink: ${line.trim()}`);
    }
  });

  return failures;
}

function lintTree(root, exts = SCAN_EXTENSIONS) {
  const failures = [];
  for (const file of findFiles(root, exts)) {
    if (file.includes('/vendor/')) continue;
    failures.push(...scanFile(file));
  }
  return failures;
}

test('no innerHTML with template sinks', () => {
  const failures = lintTree('./site');
  if (failures.length > 0) {
    assert.fail('Found innerHTML sinks:\n' + failures.join('\n'));
  }
});

test('lint regression: walks fixtures and catches real sinks', () => {
  const failures = lintTree('./tests/fixtures');
  assert.strictEqual(failures.length, 7, 'Should catch exactly 7 sink occurrences in the fixtures');
});

test('comments and string literals mentioning sinks are ignored', () => {
  const failures = scanFile('./tests/fixtures/lint-clean.js');
  assert.strictEqual(failures.length, 0, 'Comment/string mentions must not fail the gate');
});

test('file walk reaches .mjs sinks', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'q17-lint-'));
  try {
    fs.writeFileSync(path.join(tmp, 'clean.js'), 'function ok() { return 1; }\n');
    fs.writeFileSync(path.join(tmp, 'sink.mjs'), 'function bad(el) { el.innerHTML = "<b>x</b>"; }\n');
    const failures = lintTree(tmp);
    assert.strictEqual(failures.length, 1, 'The real walk must scan and catch the .mjs sink');
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
