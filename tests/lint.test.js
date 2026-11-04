import { test } from 'node:test';
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

// Extensions the walk scans. Pinned by the 'file walk reaches .mjs sinks' test:
// dropping '.mjs' here fails the suite. '.html' is scanned through the inline-script masker
// below, not as raw markup (learnings-0ob).
const SCAN_EXTENSIONS = ['.js', '.mjs', '.html'];

// Property names whose COMPUTED form is a sink — `el['name'] = x` for the two HTML setters and for
// srcdoc, `globalThis['eval'](x)` for the call: kept verbatim by the blanking pass so the computed
// patterns below can see them (learnings-blv, learnings-wltf).
const COMPUTED_SINK_PROPERTIES = new Set(['innerHTML', 'outerHTML', 'srcdoc', 'eval']);

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
// The only literals kept verbatim are the bare names in COMPUTED_SINK_PROPERTIES, because
// `el["innerHTML"] = x`, `frame['srcdoc'] = x` and `globalThis['eval'](x)` are real sinks that a
// pattern over the blanked line could not see (learnings-blv, learnings-wltf).
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
      out += COMPUTED_SINK_PROPERTIES.has(content) ? source.slice(start, i) : ' '.repeat(i - start);
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
      if (!hasInterpolation && COMPUTED_SINK_PROPERTIES.has(content)) {
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

// Every DOM-injection construct the gate refuses, with the name reported on a hit.
//
// WHY A LIST (learnings-blv): matching only `.innerHTML =` and three friends left a scanner that a
// single `el.outerHTML = x`, `iframe.srcdoc = x` or `new Function(code)()` walked straight past. A
// gate whose bypass is "use the other property" is not a gate. Each pattern is matched against the
// string/comment-blanked line, so a construct inside a comment or a string is not a hit — except
// `setAttribute` with an event-handler name, whose attribute name IS a literal (see below).
//
// WHY EVERY ENTRY ALSO LISTS ITS ADJACENT SPELLING (learnings-wltf): the same objection applies one
// construct down. A gate that refuses `frame.srcdoc = x` but not `frame.srcdoc += x` or
// `frame['srcdoc'] = x`, that refuses `new Function(code)` but not the spec-identical bare
// `Function(code)()`, that refuses `eval(code)` but not `globalThis['eval'](code)`, and that refuses
// `setAttribute('onclick', x)` but not `setAttributeNS(null, 'onclick', x)`, is bypassed by a rename
// and not by a different technique. Aliasing (`const E = eval`) stays out of scope for a line
// scanner; that limit is stated here rather than hidden.
//
// DELIBERATELY NOT LISTED: `insertAdjacentText` and `textContent`/`innerText` take text and cannot
// parse HTML, so flagging them would be a false positive that pushes real code to work around the
// gate. Nothing legitimately needs the constructs below in this repository: the tree is clean
// against every pattern, which is what makes them enforceable.
const SINK_PATTERNS = [
  [/\.innerHTML\s*\+?=/, 'innerHTML assignment'],
  [/\[['"`]innerHTML['"`]\]\s*\+?=/, 'computed innerHTML assignment'],
  [/\[['"`]outerHTML['"`]\]\s*\+?=/, 'computed outerHTML assignment'],
  [/\.outerHTML\s*\+?=/, 'outerHTML assignment'],
  [/\.insertAdjacentHTML\s*\(/, 'insertAdjacentHTML'],
  [/\bdocument\.write(?:ln)?\s*\(/, 'document.write'],
  [/\.createContextualFragment\s*\(/, 'createContextualFragment'],
  [/\.srcdoc\s*\+?=/, 'iframe srcdoc assignment'],
  [/\[['"`]srcdoc['"`]\]\s*\+?=/, 'computed srcdoc assignment'],
  [/\beval\s*\(/, 'eval'],
  [/\[['"`]eval['"`]\]\s*\(/, 'computed eval call'],
  // No `new`: `Function(code)()` compiles and runs code exactly like `new Function(code)()`.
  [/\bFunction\s*\(/, 'Function constructor'],
];

// An inline event-handler attribute set from script is the same injection with a different name.
// Its attribute name is a string literal, which the blanking pass removes, so these are judged on
// the COMMENT-stripped line (like the empty-clear allowance). Two shapes, because the name is the
// FIRST argument of setAttribute and the SECOND of setAttributeNS — one optional-argument pattern
// would also read `setAttribute('class', 'only-if-loaded')` as an event handler (learnings-wltf).
// `setAttribute("on" + name, x)` is matched by neither and is covered only by review, which this
// comment states.
const EVENT_ATTR_LINES = [
  /\.setAttribute\s*\(\s*['"`]on[a-z]/i,
  /\.setAttributeNS\s*\(\s*[^,)]+,\s*['"`]on[a-z]/i,
];

function sinkConstruct(line, commentStrippedLine) {
  for (const [pattern, label] of SINK_PATTERNS) if (pattern.test(line)) return label;
  for (const pattern of EVENT_ATTR_LINES) if (pattern.test(commentStrippedLine)) return 'setAttribute of an event handler';
  return null;
}

// Script `type` values a browser executes as JavaScript. Anything else (speculationrules,
// importmap, application/ld+json, application/json, text/template, ...) is a data block, not
// code: the published pages ship such blocks (site/celld/index.html pins its speculationrules
// JSON by CSP sha256), and scanning them as JS would be meaningless. A missing or empty type
// is JavaScript. Deliberately a small explicit set: a type the browser will not execute cannot
// inject anything, and an exotic executable type appearing in the tree is a review-level event.
const JS_SCRIPT_TYPES = new Set(['', 'module', 'text/javascript', 'application/javascript',
  'text/ecmascript', 'application/ecmascript', 'text/jscript', 'text/livescript']);

// Returns a newline-aligned copy of an HTML source in which EVERYTHING is blanked to spaces
// except the bodies of executable inline <script> blocks. The same per-line scan used for JS
// files then runs over the result, so `file:line` reporting stays exact while markup, text
// nodes, attributes, HTML comments, external `src` scripts and non-JS data blocks can never
// produce a hit (learnings-0ob). Mirrors the HTML parser's own rules where they matter here:
// a script body ends at the first `</script`, and commented-out markup is inert. Regions are
// blanked with one native replace and joined once: a per-character loop over the published
// tree cost seconds of fast-gate time on a loaded box.
function maskHtmlKeepingInlineScripts(raw) {
  const blank = (s) => s.replace(/[^\n]/g, ' ');
  const parts = [];
  let i = 0;
  const n = raw.length;

  while (i < n) {
    const lt = raw.indexOf('<', i);
    if (lt === -1) { parts.push(blank(raw.slice(i))); break; }
    if (lt > i) { parts.push(blank(raw.slice(i, lt))); i = lt; }

    if (raw.startsWith('<!--', i)) {
      const end = raw.indexOf('-->', i + 4);
      const stop = (end === -1) ? n : end + 3;
      parts.push(blank(raw.slice(i, stop)));
      i = stop;
      continue;
    }

    if (/^<script(?=[\s>/])/i.test(raw.slice(i, i + 8))) {
      // Parse the open tag, skipping quoted attribute values so a `>` inside one cannot end it.
      let j = i + '<script'.length;
      let attrs = '';
      while (j < n) {
        const ch = raw[j];
        if (ch === '"' || ch === "'") {
          let k = j + 1;
          while (k < n && raw[k] !== ch) k++;
          attrs += raw.slice(j, Math.min(k + 1, n));
          j = k + 1;
          continue;
        }
        if (ch === '>') break;
        attrs += ch;
        j++;
      }
      const bodyStart = Math.min(j + 1, n);
      // No closing tag: the body runs to EOF — which is what a browser does with an unclosed
      // <script> (it consumes the rest of the document), so scanning it as JS is the
      // fail-closed, parser-faithful reading; a page with an unclosed script is broken anyway
      // (learnings-r7a7 reviewed this edge).
      const close = raw.slice(bodyStart).search(/<\/script[\s>]/i);
      const bodyEnd = (close === -1) ? n : bodyStart + close;
      const gt = (close === -1) ? -1 : raw.indexOf('>', bodyEnd);
      const closeTagEnd = (gt === -1) ? n : gt + 1;
      // (?:^|\s) anchor, not \b, for the same reason as src below: \btype also matched data-type=
      // and any other *-type=, and the leftmost match wins, so
      // <script data-type="text/plain" type="module"> was read as a text/plain DATA block and its
      // executable body blanked — fail-open (learnings-wcea).
      const typeMatch = attrs.match(/(?:^|\s)type\s*=\s*(?:"([^"]*)"|'([^']*)'|([^\s>]+))/i);
      const type = typeMatch ? (typeMatch[1] ?? typeMatch[2] ?? typeMatch[3] ?? '').trim().toLowerCase() : '';
      // (?:^|\s) anchor, not \b: \bsrc also matched data-src=, which would blank the body of
      // an executable inline script that merely carries a data-src attribute — fail-open
      // (learnings-r7a7).
      const isExecutable = !/(?:^|\s)src\s*=/i.test(attrs) && JS_SCRIPT_TYPES.has(type);
      parts.push(blank(raw.slice(i, bodyStart)));
      parts.push(isExecutable ? raw.slice(bodyStart, bodyEnd) : blank(raw.slice(bodyStart, bodyEnd)));
      parts.push(blank(raw.slice(bodyEnd, closeTagEnd)));
      i = closeTagEnd;
      continue;
    }

    parts.push(' ');
    i++;
  }

  return parts.join('');
}

function scanFile(file) {
  const failures = [];
  const raw = fs.readFileSync(file, 'utf8');
  // For HTML only executable inline <script> bodies survive the masker; every scan below runs
  // on that newline-aligned form so line numbers and the JS string/comment stripping stay exact.
  // A page with no executable inline script — the whole published tree today — returns before
  // the two strip passes ever see the blanked markup.
  const source = file.endsWith('.html') ? maskHtmlKeepingInlineScripts(raw) : raw;
  if (file.endsWith('.html') && !/\S/.test(source)) return failures;
  const lines = source.split('\n');
  const stripped = stripCommentsAndStrings(source).split('\n');
  const noComments = stripComments(source).split('\n');

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

    const construct = sinkConstruct(strippedLine, noComments[index]);
    if (construct) {
      failures.push(`File ${file}:${index + 1} uses DOM-injection sink (${construct}): ${line.trim()}`);
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

// learnings-19b8: the walk covered only ./site, so the committed research/ and scripts/ trees —
// code that runs on the VM and, for the probes, drives a browser — were outside every static gate
// in the repo (csp.test.js also walks site/ only). Both are scanned now, and a root that loses its
// coverage fails the explicit check below rather than silently scanning nothing.
const SOURCE_ROOTS = ['./site', './research', './scripts'];

test('no innerHTML with template sinks', () => {
  const failures = SOURCE_ROOTS.flatMap(root => lintTree(root));
  if (failures.length > 0) {
    assert.fail('Found innerHTML sinks:\n' + failures.join('\n'));
  }
});

test('the gate walks research/ and scripts/, not just site/', () => {
  // A root that quietly stopped matching would make the scan above pass without scanning these,
  // so the coverage itself is the assertion: real .mjs files from each tree are in the walk.
  const walked = SOURCE_ROOTS.flatMap(root => findFiles(root, SCAN_EXTENSIONS));
  for (const expected of ['research/nn-inp/trace2.mjs', 'scripts/serve.mjs', 'site/decision-models/image-lab.js']) {
    assert.ok(walked.includes(expected), `${expected} is scanned by the DOM-sink gate`);
  }
  // And a sink planted in a research-shaped path is still caught, so this is a scan and not a
  // filename allowlist.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), '19b8-lint-'));
  try {
    fs.mkdirSync(path.join(tmp, 'research'));
    fs.writeFileSync(path.join(tmp, 'research', 'probe.mjs'), 'function bad(el) { el.innerHTML = `<b>${x}</b>`; }\n');
    assert.strictEqual(lintTree(path.join(tmp, 'research')).length, 1);
  } finally {
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

test('lint regression: walks fixtures and catches real sinks', () => {
  // Per FILE, not a directory total: a total hides which fixture moved, and adding a fixture used to
  // silently change an unrelated expectation (learnings-blv).
  const perFile = (name) => scanFile(`./tests/fixtures/${name}`);
  assert.strictEqual(perFile('lint-sinks.js').length, 4, 'lint-sinks.js: the original innerHTML/write sinks');
  assert.strictEqual(perFile('lint-sinks.mjs').length, 1, 'lint-sinks.mjs: the planted .mjs sink');
  assert.strictEqual(perFile('lint-plant-app.js').length, 2, 'lint-plant-app.js: the two planted app sinks');
  assert.strictEqual(perFile('lint-clean.js').length, 0, 'comments and string literals mentioning a sink are not hits');
  assert.strictEqual(perFile('lint-sinks-extra.js').length, 14, 'lint-sinks-extra.js: one hit per construct');
  assert.strictEqual(perFile('lint-clean-extra.js').length, 0, 'text APIs and quoted constructs are not hits');
  assert.strictEqual(lintTree('./tests/fixtures').filter((f) => f.includes('lint-clean.js') || f.includes('lint-clean-extra.js')).length, 0);
});

test('every construct in lint-sinks-extra.js is reported BY NAME, one per line (learnings-blv, learnings-wltf)', () => {
  // A count cannot say which construct stopped being caught: a pattern firing twice on one line and
  // another going dead both leave it at 14. The fixture is one construct per line, so the ordered
  // label list is the assertion. The last five are the adjacent spellings a reviewer proved walked
  // past the widened gate (learnings-wltf): the computed and `+=` srcdoc forms, bare `Function(...)`,
  // `setAttributeNS` with a handler name, and the computed eval call.
  const labels = scanFile('./tests/fixtures/lint-sinks-extra.js')
    .map((f) => f.replace(/^.*DOM-injection sink \((.*)\):.*$/, '$1'));
  assert.deepStrictEqual(labels, [
    'outerHTML assignment', 'insertAdjacentHTML', 'document.write', 'createContextualFragment',
    'iframe srcdoc assignment', 'eval', 'Function constructor', 'setAttribute of an event handler',
    'computed outerHTML assignment',
    'computed srcdoc assignment', 'iframe srcdoc assignment', 'Function constructor',
    'setAttribute of an event handler', 'computed eval call',
  ], 'one label per fixture line, in file order');
});

test('comments and string literals mentioning sinks are ignored', () => {
  const failures = scanFile('./tests/fixtures/lint-clean.js');
  assert.strictEqual(failures.length, 0, 'Comment/string mentions must not fail the gate');
});

test('inline <script> bodies in HTML are scanned; markup and data blocks are not (learnings-0ob)', () => {
  // The five planted hits: a document.write in a script carrying data-src (line 11 — data-src
  // is NOT src, so the body is executable and must hit: pinning the attribute-boundary anchor,
  // learnings-r7a7), an interpolated innerHTML assignment in a classic inline script (line 18),
  // document.write in a module inline script (line 21), and the two data-type controls at lines
  // 30/31 — data-type= is NOT type= either (learnings-wcea), so the first script has no type at
  // all and the second's real type="module" must win over the data-type beside it; under \btype
  // the leftmost match read data-type="text/plain" and blanked both executable bodies.
  // The exclusion controls are DISCRIMINATING (learnings-r7a7, learnings-wcea): the external-src
  // script carries a sink-looking body, the text/template block carries unquoted sink text, and
  // line 32 pairs a data-type="text/javascript" with a REAL type="text/template" — so if the
  // masker's guards broke in either direction (isExecutable forced true, or data-type read as
  // type) the list would move. An empty-bodied src control only proved that string-blanking hides
  // things, not that the masker excludes. The commented-out sink, the prose/<pre> listing
  // mentions, the speculationrules JSON and the ld+json block must not produce hits either.
  const failures = scanFile('./tests/fixtures/lint-inline.html');
  assert.deepStrictEqual(
    failures.map((f) => f.replace(/ uses DOM-injection sink.*/, '')),
    ['File ./tests/fixtures/lint-inline.html:11', 'File ./tests/fixtures/lint-inline.html:18',
      'File ./tests/fixtures/lint-inline.html:21', 'File ./tests/fixtures/lint-inline.html:30',
      'File ./tests/fixtures/lint-inline.html:31'],
    `exactly the planted inline-script sinks, at their real line numbers, got:\n${failures.join('\n')}`);
  assert.ok(failures[0].includes('document.write'), 'the data-src script body is executable and hits');
  assert.ok(failures[1].includes('innerHTML assignment'), 'the innerHTML hit names its construct');
  assert.ok(failures[2].includes('document.write'), 'the document.write hit names its construct');
  assert.ok(failures[3].includes('document.write'), 'a data-type= script with no type= is executable JS and hits');
  assert.ok(failures[4].includes('insertAdjacentHTML'), 'a real type="module" next to data-type= is still read as the script type');
  // And the published pages that motivated the coverage are actually in the walk.
  const walked = SOURCE_ROOTS.flatMap((root) => findFiles(root, SCAN_EXTENSIONS));
  for (const expected of ['site/celld/index.html', 'site/neural-networks/backends/javascript.html']) {
    assert.ok(walked.includes(expected), `${expected} is scanned by the DOM-sink gate`);
  }
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
