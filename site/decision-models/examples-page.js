// Examples page + schema playground. No model is called here: the examples are text and the
// playground only validates JSON with the same walker the lab and the generator use.
import { validateSpec, validateSpecIssues, requestFor } from './core.js';
import { examples, adaptationSpec } from './examples.js';

const $ = id => document.getElementById(id);
// One-shot hand-off to lab.html. Session storage is tab-scoped and the lab deletes the entry as
// it reads it, so a draft cannot outlive the tab or be picked up twice.
const DRAFT_KEY = 'decision-spec-draft';
const SOURCES = Object.freeze({ ...examples, adaptation: adaptationSpec });
const DEBOUNCE_MS = 400;

let valid = null;
let timer = null;

const setCopyStatus = message => { $('copy-status').textContent = message; };
const setStatus = message => { $('schema-status').textContent = message; };
const count = (n, singular) => `${n} ${singular}${n === 1 ? '' : 's'}`;

function setActions(enabled) {
  $('schema-copy').disabled = !enabled;
  $('schema-download').disabled = !enabled;
  $('schema-open-lab').disabled = !enabled;
}

function renderIssues(issues) {
  const list = $('schema-issues');
  list.replaceChildren();
  for (const { path, message } of issues) {
    const item = document.createElement('li');
    const where = document.createElement('code');
    where.textContent = path || '(root)';
    item.append(where, document.createTextNode(` — ${message}`));
    list.append(item);
  }
  $('schema-report').hidden = issues.length === 0;
}

function refuse(message) {
  valid = null;
  setActions(false);
  $('schema-request').textContent = 'No valid schema yet.';
  setStatus(message);
}

function sync() {
  let parsed;
  try { parsed = JSON.parse($('schema').value); }
  catch (error) { renderIssues([]); refuse(`That is not valid JSON: ${error.message}`); return; }

  const issues = validateSpecIssues(parsed);
  renderIssues(issues);
  if (issues.length) { refuse(`${count(issues.length, 'problem')} found. Nothing would be sent.`); return; }

  valid = validateSpec(parsed);
  setActions(true);
  const model = $('schema-model').value.trim();
  $('schema-request').textContent = model
    ? JSON.stringify(requestFor(valid, model), null, 2)
    : 'Enter a model ID above to see the request body.';
  setStatus(`Valid. ${count(Object.keys(valid.questions).length, 'question')}; the lab and the generator accept this specification.`);
}

for (const [id, spec] of Object.entries(SOURCES)) {
  const option = document.createElement('option');
  option.value = id;
  option.textContent = spec.title;
  $('schema-example').append(option);
}

$('schema-example').addEventListener('change', () => {
  const chosen = SOURCES[$('schema-example').value];
  if (!chosen) return;
  $('schema').value = JSON.stringify(chosen, null, 2);
  sync();
  setCopyStatus(`Loaded the checked example “${chosen.title}”.`);
});

$('schema-check').addEventListener('click', () => { clearTimeout(timer); sync(); });
$('schema').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(sync, DEBOUNCE_MS); });
$('schema-model').addEventListener('input', () => { clearTimeout(timer); timer = setTimeout(sync, DEBOUNCE_MS); });

$('schema-copy').addEventListener('click', async () => {
  if (!valid) return;
  try {
    await navigator.clipboard.writeText(JSON.stringify(valid, null, 2));
    setCopyStatus('Validated JSON copied. It is still only in this page.');
  } catch {
    setCopyStatus('This browser refused clipboard access. Select the JSON above and copy it with Ctrl or Cmd + C.');
  }
});

$('schema-download').addEventListener('click', () => {
  if (!valid) return;
  const url = URL.createObjectURL(new Blob([JSON.stringify(valid, null, 2)], { type: 'application/json' }));
  const anchor = document.createElement('a');
  anchor.href = url;
  anchor.download = 'decision-specification.json';
  anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  setCopyStatus('Downloaded decision-specification.json.');
});

$('schema-open-lab').addEventListener('click', () => {
  if (!valid) return;
  try { sessionStorage.setItem(DRAFT_KEY, JSON.stringify(valid)); }
  catch {
    setCopyStatus('This tab blocked session storage, so the lab cannot pick the draft up. Use “Copy validated JSON” and paste it into the lab.');
    return;
  }
  location.assign('lab.html');
});

for (const button of document.querySelectorAll('[data-copy]')) {
  button.addEventListener('click', async () => {
    const block = document.querySelector(`[data-example="${button.dataset.copy}"] code`);
    if (!block) return;
    try {
      await navigator.clipboard.writeText(block.textContent);
      setCopyStatus(`Copied “${button.textContent}” — the same text is on the clipboard.`);
    } catch {
      // Clipboard access is denied on some origins and in some browsers. Selecting the block
      // leaves the reader one keystroke from the same result instead of a dead button.
      const range = document.createRange();
      range.selectNodeContents(block);
      const selection = getSelection();
      // A document without a selection object (detached frame, some embeds) still gets the
      // explanation, just not the pre-selected block.
      if (selection) {
        selection.removeAllRanges();
        selection.addRange(range);
        setCopyStatus('This browser refused clipboard access, so the block is selected: press Ctrl or Cmd + C.');
      } else {
        setCopyStatus('This browser refused clipboard access. Select the block above and copy it with Ctrl or Cmd + C.');
      }
    }
  });
}

$('schema').value = JSON.stringify(examples.tools, null, 2);
sync();
