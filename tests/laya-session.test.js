// tests/laya-session.test.js — the LayaSession answer map. decideAll() runs one graph pass per
// question, so the inference itself is stubbed here; what is under test is the map it returns,
// which callers read with Object.entries() and render row by row (playground.js).
import test from "node:test";
import assert from "node:assert/strict";
import { LayaSession } from "../site/decision-models/laya-engine.js";

// The constructor only stores its inputs — nothing is compiled until loadLaya() — so a
// spec-shaped object plus a stubbed decide() is enough to drive the real decideAll() loop.
function stubbedSession() {
  const session = new LayaSession({
    spec: { repo: "test/laya", label: "stub", window: 8, pooledDim: 4 },
    checkpoint: "stub",
    main: null,
    act: null,
    tokenizer: null,
    encode: null,
    config: {},
    embeddings: null,
  });
  session.decide = async (state, question) => ({
    answer: { type: question.t, ins: question.ins },
    tokens: 3,
  });
  return session;
}

test("decideAll keeps a __proto__ question id as an own answer key instead of dropping the row (learnings-5tf)", async () => {
  // The playground hands decideAll the question textarea's JSON with no id validation in front of
  // it, and JSON.parse yields `__proto__` as an ordinary own key.
  const questions = JSON.parse(
    '{"tool":{"t":"choice","ins":"Which tool?"},"__proto__":{"t":"noul","ins":"Is the path clear?"}}',
  );

  const report = await stubbedSession().decideAll("state", questions);

  assert.deepEqual(
    Object.entries(report.answers).map(([id]) => id).sort(),
    ["__proto__", "tool"],
    "every question id the caller passed in comes back as a key of the returned answers map",
  );
  assert.ok(
    Object.hasOwn(report.answers, "__proto__"),
    "a __proto__ question id must be an own property, not a reassignment of the map's prototype",
  );
  assert.equal(
    Object.getPrototypeOf(report.answers),
    null,
    "the accumulator is prototype-less, so no question id can reach an inherited accessor",
  );
  assert.deepEqual(report.answers["__proto__"], { type: "noul", ins: "Is the path clear?" });
  assert.deepEqual(
    JSON.parse(JSON.stringify(report.answers)),
    JSON.parse('{"tool":{"type":"choice","ins":"Which tool?"},"__proto__":{"type":"noul","ins":"Is the path clear?"}}'),
    "callers serialize the whole run (app.js raw readout and download), so both keys survive the round trip",
  );
  assert.equal(report.usage.input_tokens, 6, "both rows were decided and their tokens counted");
});
