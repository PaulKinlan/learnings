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

// The only innerHTML assignment this allowance admits is a clear of the whole element: the
// statement must be NOTHING BUT the clear. Judged on the STRIPPED line, so a fake clear inside
// a trailing line or block comment cannot whitelist a real sink, and anchored at BOTH ends —
// end-anchoring alone matches the line tail, so a sink earlier on the same line returns before
// isSinkLine() ever runs. `[^;]*` cannot cross a statement separator, so `sink; x.innerHTML = ""`
// fails while every legitimate one-statement clear still passes (learnings-flk).
const EMPTY_CLEAR_LINE = /^[^;]*\.innerHTML\s*=\s*["']\s*["']\s*;?\s*$/;

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

// Blanks line comments and block comments while keeping newlines aligned, leaving string and
// template literals VERBATIM. The clear/notice allowances run on this form: a fake clear written
// inside a comment disappears (so it cannot whitelist a sink), while the RHS literal stays
// visible — the fully-blanked form cannot distinguish `= ""` from `= userHtml`.
function stripComments(source) {
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

    if (ch === '"' || ch === "'" || ch === '`') {
      const quote = ch;
      out += ch; i++;
      while (i < n && source[i] !== quote) {
        if (source[i] === '\\' && i + 1 < n) { out += source[i] + source[i + 1]; i += 2; continue; }
        out += source[i]; i++;
      }
      if (i < n) { out += quote; i++; }
      continue;
    }

    out += ch; i++;
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
  const noComments = stripComments(raw).split('\n');

  lines.forEach((line, index) => {
    const strippedLine = stripped[index];
    // Allow clearing a container (empty string or a single space), with or
    // without whitespace around `=`. Judged on the comment-stripped line and
    // anchored at BOTH ends: end-anchoring alone matches only the line tail, so a
    // sink earlier on the same line returned before isSinkLine() ran, and judging
    // the raw line let a fake clear inside a trailing comment through
    // (tm-lint-empty-clear-allowance-bypass, learnings-7wj, learnings-flk).
    // `[^;]*` cannot cross a statement separator.
    if (EMPTY_CLEAR_LINE.test(noComments[index])) return;

    // Allow the known constant-only notice in image-lab.js. Scoped by path, so a
    // `notice.innerHTML = userInput` sink in any other file still fails. Judged on the
    // comment-stripped line for the same reason: a comment cannot extend this allowance.
    if (file.endsWith('/decision-models/image-lab.js') && NOTICE_CONSTANT_LINE.test(noComments[index].trim())) return;

    if (isSinkLine(strippedLine)) {
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

test('the empty-string clearing allowance is anchored: a suffix after the literal is a sink (learnings-7wj)', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), '7wj-lint-'));
  try {
    // Legitimate clears keep passing, including with a trailing comment.
    fs.writeFileSync(path.join(tmp, 'clears.js'), [
      'function reset(el, panel) {',
      '  el.innerHTML = "";',
      "  el.innerHTML = ' ';",
      '  panel.innerHTML = "" ; // wipe the panel',
      '}',
      '',
    ].join('\n'));
    // Every one of these merely STARTS with an empty literal: the old unanchored allowance
    // returned before isSinkLine() ran and silently allowed the concatenation (or comma
    // expression) straight into the sink.
    fs.writeFileSync(path.join(tmp, 'bypass.js'), [
      'function inject(el, panel, userHtml) {',
      '  el.innerHTML = "" + userHtml;',
      '  panel.innerHTML = ""+ userHtml;',
      '  el.innerHTML = "" , userHtml;',
      '}',
      '',
    ].join('\n'));
    const failures = lintTree(tmp);
    assert.strictEqual(failures.length, 3, `every empty-literal-prefixed sink must fail, got:\n${failures.join('\n')}`);
    assert.ok(failures.every(f => f.includes('bypass.js')), 'only bypass.js fails; the anchored clears stay allowed');
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('the clearing allowance is anchored at both ends of the stripped line: no same-line or comment bypass (learnings-flk)', () => {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'flk-lint-'));
  try {
    // A real sink BEFORE a legit clear on the same line, or a fake clear inside a trailing
    // line/block comment: the old tail-only, raw-line allowance whitelisted all of these.
    fs.writeFileSync(path.join(tmp, 'bypass.js'), [
      'function inject(el, panel, userHtml) {',
      '  el.innerHTML = userHtml; panel.innerHTML = "";',
      '  el.innerHTML = userHtml; // previously: panel.innerHTML = ""',
      '  el.innerHTML = userHtml; /* was: panel.innerHTML = "" */',
      '}',
      '',
    ].join('\n'));
    // Legitimate one-statement clears — including with a trailing comment — stay allowed.
    fs.writeFileSync(path.join(tmp, 'clears.js'), [
      'function reset(el, panel) {',
      '  el.innerHTML = "";',
      '  panel.innerHTML = " " ; // wipe the panel',
      '}',
      '',
    ].join('\n'));
    const failures = lintTree(tmp);
    assert.strictEqual(failures.length, 3, `every sink-before-clear form must fail, got:\n${failures.join('\n')}`);
    assert.ok(failures.every(f => f.includes('bypass.js')), 'only bypass.js fails; the real clears stay allowed');
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});
