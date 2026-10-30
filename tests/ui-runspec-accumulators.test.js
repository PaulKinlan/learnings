// tests/ui-runspec-accumulators.test.js — runSpec's answer/question accumulators are keyed by
// caller-supplied question ids. runSpec is exported and does not validate on entry, so a spec
// carrying a `__proto__` question id must keep it as an own key instead of triggering the
// prototype setter and silently dropping the row (tm-plain-object-answer-accumulators;
// the learnings-e7p/learnings-5tf class). DOM and engine imports are stubbed; no browser,
// no network, no model.
import test from "node:test";
import assert from "node:assert/strict";
import { registerHooks } from "node:module";

registerHooks({
  resolve(specifier, context, nextResolve) {
    const stubs = {
      "./core.js":
        "export const decide = async () => { throw new Error('unexpected jev path'); };" +
        "export const gate = () => 'n/a';" +
        // Mirror the real helper's non-browser behaviour: no location realm -> harness owns the seam.
        "export const mockDecide = () => globalThis.__MOCK_DECIDE__ ?? null;" +
        "export const onLoopback = () => true;",
      "./laya-engine.js":
        "export const loadLaya = async () => ({ checkpoint: 'fixture-laya'," +
        " decideAll: async (state, questions) => { globalThis.__layaQuestions = questions;" +
        " const answers = Object.create(null);" +
        " for (const id of Object.keys(questions)) answers[id] = { type: 'noul', noul: 1 };" +
        " return { answers }; } });",
      "./laya-pack.js": "export const choiceConfidence = () => 1;",
      "./kev-engine.js":
        "export const loadKev = async () => ({ checkpoint: 'fixture-kev'," +
        " decidePacked: async (state, questions) => ({ distributions: questions.map(() => [0.4, 0.6]), tokens: 0, ms: 0, flat: [] }) });",
      "./kev-pack.js":
        "export const choose = () => 0;" +
        "export const expectedLevel = () => 0;" +
        "export const noulProbability = (dist) => dist[1];",
    };
    if (stubs[specifier] && context.parentURL?.endsWith("/site/decision-models/ui.js")) {
      return { url: `data:text/javascript,${stubs[specifier]}`, shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

// settings() reads the provider select through $ -> document.getElementById; a null element
// defaults to 'laya', a {value} element selects the path under test.
let providerValue = null;
globalThis.document = {
  getElementById: (id) => (id === "provider" && providerValue ? { value: providerValue } : null),
};

const { runSpec } = await import("../site/decision-models/ui.js");

// JSON.parse, not an object literal: parse yields `__proto__` as an ordinary own key, which is
// how a caller-supplied spec would actually carry it.
const SPEC = JSON.parse(
  '{"state":"the state","questions":{' +
    '"tool":{"type":"noul","instructions":"Is the tool right?"},' +
    '"__proto__":{"type":"noul","instructions":"Is the path clear?"}' +
  "}}",
);

test("runSpec (laya path) keeps a __proto__ question id as an own key through to the answers (learnings-7h7)", async () => {
  providerValue = null; // no provider element -> settings() defaults to 'laya'
  const result = await runSpec(structuredClone(SPEC));
  assert.ok(
    Object.hasOwn(globalThis.__layaQuestions, "__proto__"),
    "the __proto__ question must reach decideAll as an own key, not be dropped by the prototype setter",
  );
  assert.deepEqual(Object.keys(globalThis.__layaQuestions).sort(), ["__proto__", "tool"]);
  assert.ok(
    Object.hasOwn(result.data.answers, "__proto__"),
    "the __proto__ answer must be an own key of the returned answers",
  );
  assert.equal(Object.getPrototypeOf(globalThis.__layaQuestions), null);
});

test("runSpec (kev path) accumulates a __proto__ answer as an own key without polluting the prototype (learnings-7h7)", async () => {
  providerValue = "kev";
  const result = await runSpec(structuredClone(SPEC));
  assert.ok(
    Object.hasOwn(result.data.answers, "__proto__"),
    "the __proto__ answer must be an own key, not a reassignment of the accumulator's prototype",
  );
  assert.deepEqual(Object.keys(result.data.answers).sort(), ["__proto__", "tool"]);
  assert.equal(Object.getPrototypeOf(result.data.answers), null);
  assert.equal(result.data.answers["__proto__"].type, "noul");
});
