import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { PRESETS, decideImage } from '../site/decision-models/image-lab.js';

// The image lab's "api" engine used to advertise "Multimodal API Adapter (Wity-1 / OpenAI / Gemini)"
// while hardcoding every request — the reader's API key and full image payload — to
// https://api.typesafe.ai/v1/systemone. A reader who selected that mode believing they were calling
// OpenAI or Gemini would hand that credential to TypeSafe. The fix relabels the engine and its
// fields so the UI states plainly that the key and image go to api.typesafe.ai, and the notice names
// the recipient. These assertions pin that disclosure; reverting the relabel fails them.

const html = readFileSync(new URL('../site/decision-models/image-lab.html', import.meta.url), 'utf8');
const js = readFileSync(new URL('../site/decision-models/image-lab.js', import.meta.url), 'utf8');

test('image lab api engine names its recipient instead of advertising other providers (learnings-mem)', () => {
  const option = html.match(/<option value="api">([^<]+)<\/option>/);
  assert.ok(option, 'api engine option must exist');
  const label = option[1];
  assert.match(label, /TypeSafe/, `api engine option must name TypeSafe, got "${label}"`);
  assert.doesNotMatch(
    label,
    /OpenAI|Gemini|Wity-1/,
    `api engine option must not advertise OpenAI/Gemini/Wity-1 as selectable recipients, got "${label}"`
  );
});

test('image lab api key and model fields state the TypeSafe recipient (learnings-mem)', () => {
  const apiField = html.match(/<div id="api-field"[^>]*>([\s\S]*?)<\/div>/);
  assert.ok(apiField, 'api-field must exist');
  const fields = apiField[1];
  assert.match(fields, /TypeSafe API key/, 'api key label must name TypeSafe');
  assert.match(fields, /Model ID \(hosted by TypeSafe\)/, 'model ID label must state it is hosted by TypeSafe');
  assert.doesNotMatch(fields, /Wity-1|OpenAI|Gemini/, 'api fields must not advertise a different provider');
});

test('image lab api engine notice states the key and image go to api.typesafe.ai (learnings-mem)', () => {
  const notice = js.match(/notice\.innerHTML = '([^']*api\.typesafe\.ai[^']*)'/);
  assert.ok(notice, 'api engine notice must name the recipient endpoint');
  assert.match(notice[1], /API key and image/i, `notice must state the key and image are sent, got "${notice[1]}"`);
  assert.match(notice[1], /api\.typesafe\.ai\/v1\/systemone/, `notice must name the endpoint, got "${notice[1]}"`);
});

test('image lab api engine sends the key and image to api.typesafe.ai and says so in its source (learnings-mem)', async (t) => {
  const spec = PRESETS.ui.spec;
  const dummyImage = 'data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==';
  const payload = {
    model: 'stub-vision-server',
    answers: {
      is_blocked: { type: 'noul', noul: 0.93, confidence: 0.93 },
      action: {
        type: 'choice',
        choice: 'retry',
        confidence: 0.88,
        probabilities: { retry: 0.88, cancel: 0.06, dismiss: 0.04, unclear: 0.02 }
      },
      severity: {
        type: 'score',
        score: 1.9,
        confidence: 0.9,
        probabilities: { '0': 0.02, '1': 0.08, '2': 0.9 }
      }
    }
  };

  const original = globalThis.fetch;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    return { ok: true, status: 200, json: async () => structuredClone(payload) };
  };
  t.after(() => { globalThis.fetch = original; });

  const key = 'synthetic-test-not-a-real-key';
  const res = await decideImage({
    spec,
    imagePayload: dummyImage,
    engine: 'api',
    endpoint: '',
    key,
    model: 'Imajev-4B'
  });

  assert.equal(calls.length, 1, 'api engine must make exactly one request');
  assert.equal(calls[0].url, 'https://api.typesafe.ai/v1/systemone', 'the only recipient must be api.typesafe.ai');
  assert.equal(calls[0].options.headers.Authorization, `Bearer ${key}`, 'the key must go only to the TypeSafe recipient');
  assert.equal(JSON.parse(calls[0].options.body).image, dummyImage, 'the image must go only to the TypeSafe recipient');
  assert.equal(res.source, 'TypeSafe API (Imajev-4B)', 'the reported source must name TypeSafe');
});
