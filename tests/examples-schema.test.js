import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { validateSpec, validateSpecIssues, requestFor } from '../site/decision-models/core.js';
import { examples, adaptationSpec } from '../site/decision-models/examples.js';

const html = readFileSync(fileURLToPath(new URL('../site/decision-models/examples.html', import.meta.url)), 'utf8');
const everySpec = [...Object.values(examples), adaptationSpec];

// The schema playground reports every problem; validateSpec throws the first. Both read the same
// walker, so the pair can never drift into two different rule sets.
const rejected = [
  null,
  undefined,
  {},
  { title: '', description: 'd', state: 's', questions: {} },
  { ...examples.tools, title: '   ' },
  { ...examples.tools, questions: {} },
  { ...examples.tools, questions: { a: { type: 'chat', instructions: 'x' } } },
  { ...examples.tools, questions: { a: { type: 'noul', instructions: '' } } },
  { ...examples.tools, questions: { a: { type: 'choice', instructions: 'x', criteria: { only: 'one option' } } } },
  { ...examples.tools, questions: { a: { type: 'choice', instructions: 'x', criteria: { a: 'A', b: '' } } } },
  { ...examples.tools, questions: { a: { type: 'score', instructions: 'x', criteria: ['only'] } } },
  { ...examples.tools, questions: { a: { type: 'noul', instructions: 'x', criteria: { yes: 'only one' } } } },
  { ...examples.tools, questions: { a: { type: 'noul', instructions: 'x', criteria: 'text' } } },
  { ...examples.tools, unexpected: true },
  { ...examples.tools, questions: { A: { type: 'noul', instructions: 'x' } } },
  { ...examples.tools, questions: { constructor: { type: 'noul', instructions: 'x' } } },
];

test('validateSpecIssues and validateSpec agree on the first problem', () => {
  for (const spec of rejected) {
    const issues = validateSpecIssues(spec);
    assert.ok(issues.length > 0, `expected at least one issue for ${JSON.stringify(spec)}`);
    assert.throws(() => validateSpec(spec), error => {
      assert.equal(error.message, issues[0].message);
      return true;
    });
  }
});

test('every checked example passes both readers with no issue', () => {
  for (const spec of everySpec) {
    assert.deepEqual(validateSpecIssues(spec), []);
    assert.deepEqual(validateSpec(spec), spec);
  }
});

test('the playground reports every problem in one pass, each with a JSON path', () => {
  const spec = {
    title: 'T',
    description: 'D',
    state: 'S',
    questions: {
      first: { type: 'choice', instructions: 'pick one', criteria: { one: 'only one option' } },
      SECOND: { type: 'noul', instructions: '' },
    },
  };
  const issues = validateSpecIssues(spec);
  assert.deepEqual(issues.map(issue => issue.path), [
    'questions.first.criteria',
    'questions.SECOND',
    'questions.SECOND.instructions',
  ]);
  assert.deepEqual(issues.map(issue => issue.message), [
    'Choice requires 2–255 named options in this workbench.',
    'Use simple lowercase question IDs.',
    'Instructions must be non-empty text.',
  ]);
});

test('a reported issue path points at the value that has to change', () => {
  const [issue] = validateSpecIssues({ ...examples.tools, questions: { action: { ...examples.tools.questions.action, criteria: {} } } });
  assert.equal(issue.path, 'questions.action.criteria');
  const [level] = validateSpecIssues({ ...examples.score, questions: { severity: { type: 'score', instructions: 'x', criteria: ['a', 2] } } });
  assert.equal(level.path, 'questions.severity.criteria[1]');
});

test('the request preview is exactly what a hosted call would send', () => {
  const request = requestFor(examples.routing, 'jev-1.13.0');
  assert.deepEqual(Object.keys(request), ['model', 'state', 'questions']);
  assert.equal(request.model, 'jev-1.13.0');
  assert.equal(request.state, examples.routing.state);
  assert.deepEqual(request.questions, examples.routing.questions);
});

test('examples page declares the shared CSP and loads one module', () => {
  assert.match(html, /<meta http-equiv="Content-Security-Policy" content="default-src 'none';/);
  assert.match(html, /<script type="module" src="examples-page\.js"><\/script>/);
  const inline = [...html.matchAll(/<script\b([^>]*)>/g)].filter(([, attributes]) => !/\bsrc=/.test(attributes));
  assert.deepEqual(inline.map(match => match[0]), [], 'no inline script may appear under this CSP');
  assert.doesNotMatch(html, /\son[a-z]+\s*=/i, 'no inline event handlers');
});

test('examples page ids are unique, so a lookup cannot land on the wrong element', () => {
  const ids = [...html.matchAll(/\sid="([^"]+)"/g)].map(match => match[1]);
  assert.deepEqual(ids.filter((id, index) => ids.indexOf(id) !== index), []);
});

test('every copy button targets a real example block, and every block has one', () => {
  const blocks = [...html.matchAll(/data-example="([^"]+)"/g)].map(match => match[1]);
  const buttons = [...html.matchAll(/data-copy="([^"]+)"/g)].map(match => match[1]);
  assert.equal(blocks.length, 4);
  assert.deepEqual(buttons.slice().sort(), blocks.slice().sort());
});

test('examples page ships the schema playground controls the module drives', () => {
  for (const id of ['schema', 'schema-example', 'schema-model', 'schema-check', 'schema-copy', 'schema-download', 'schema-open-lab', 'schema-status', 'schema-issues', 'schema-request', 'copy-status']) {
    assert.match(html, new RegExp(`id="${id}"`), `missing #${id}`);
  }
  assert.match(html, /<textarea id="schema" class="code"/, 'the editor must be a monospace textarea');
  assert.match(html, /id="schema-status" class="status" role="status"/, 'validation must be announced');
});

test('the examples page is reachable from the decision-models navigation', () => {
  for (const page of ['index.html', 'catalogue.html', 'lab.html', 'image-lab.html', 'on-device.html', 'playground.html', 'adaptation.html', 'architecture.html']) {
    const source = readFileSync(fileURLToPath(new URL(`../site/decision-models/${page}`, import.meta.url)), 'utf8');
    assert.match(source, /<a href="\.\.\/decision-models\/examples\.html">Examples<\/a>/, `${page} does not link the examples page`);
  }
  assert.match(html, /<a href="\.\.\/decision-models\/examples\.html" aria-current="page">Examples<\/a>/);
});
