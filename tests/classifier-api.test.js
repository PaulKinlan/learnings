import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Classifier, toLayaQuestions } from "../site/decision-models/classifier-api.js";
import { renderOptions } from "../site/decision-models/laya-pack.js";

// Mock Laya session for testing Classifier wrapper and mapping
class MockLayaSession {
  constructor() {
    this.window = 512;
  }
  encode(text) {
    return text.split(/\s+/);
  }
  async decide(state, question) {
    if (question.t === "noul") {
      return {
        answer: {
          type: "noul",
          noul: 0.85,
          confidence: 0.70,
        },
        tokens: 35,
        markers: 1,
      };
    }
    if (question.t === "choice") {
      const keys = Object.keys(question.crit);
      const probabilities = {};
      keys.forEach((k, idx) => {
        probabilities[k] = idx === 0 ? 0.75 : 0.25 / (keys.length - 1);
      });
      return {
        answer: {
          type: "choice",
          choice: keys[0],
          probabilities,
          confidence: 0.75,
        },
        tokens: 45,
        markers: keys.length,
      };
    }
    if (question.t === "score") {
      const count = question.crit.length;
      const probabilities = {};
      for (let i = 0; i < count; i++) {
        probabilities[String(i)] = i === count - 1 ? 0.8 : 0.2 / (count - 1);
      }
      return {
        answer: {
          type: "score",
          score: count - 1,
          probabilities,
          confidence: 0.8,
        },
        tokens: 40,
        markers: count,
      };
    }
  }
  delete() {}
}

test("Classifier API: availability reports status", async () => {
  const avail = await Classifier.availability();
  assert.ok(["available", "downloadable"].includes(avail));
});

test("Classifier API: create validates dictionary and questions", async () => {
  await assert.rejects(() => Classifier.create(null), /parameter 1 is not a dictionary/);
  await assert.rejects(() => Classifier.create({}), /'questions' must be a non-empty array/);
  await assert.rejects(() => Classifier.create({ questions: [] }), /'questions' must be a non-empty array/);
  await assert.rejects(
    () => Classifier.create({ questions: [{ type: "binary" }] }),
    /Each question must have a string 'id'/,
  );
});

test("Classifier API: classify formats decisions for binary, categorical and ordinal questions", async () => {
  const mockSession = new MockLayaSession();
  const schema = {
    context: "Customer triage",
    questions: [
      {
        id: "is_urgent",
        type: "binary",
        prompt: "Is this ticket urgent?",
      },
      {
        id: "dept",
        type: "categorical",
        prompt: "Which team?",
        options: [
          { label: "billing", description: "Payments" },
          { label: "tech", description: "Bugs" },
        ],
      },
      {
        id: "severity",
        type: "ordinal",
        prompt: "Severity rating",
        options: [
          { label: "1", description: "Low" },
          { label: "2", description: "Medium" },
          { label: "3", description: "High" },
        ],
      },
    ],
  };

  const classifier = new Classifier(mockSession, schema, {
    is_urgent: { t: "noul", ins: "Customer triage\nIs this ticket urgent?", crit: {}, meta: { originalType: "binary" } },
    dept: {
      t: "choice",
      ins: "Customer triage\nWhich team?",
      crit: { billing: "Payments", tech: "Bugs" },
      meta: { originalType: "categorical", optionsList: schema.questions[1].options },
    },
    severity: {
      t: "score",
      ins: "Customer triage\nSeverity rating",
      crit: ["Low", "Medium", "High"],
      meta: { originalType: "ordinal", optionsList: schema.questions[2].options },
    },
  });

  assert.equal(classifier.contextWindow, 512);

  const result = await classifier.classify("Payment failed for invoice #4421");

  // Binary check
  assert.equal(result.is_urgent.id, "is_urgent");
  assert.equal(result.is_urgent.label, "true");
  assert.equal(result.is_urgent.probability, 0.85);
  assert.equal(result.is_urgent.probabilities.length, 2);
  assert.equal(result.is_urgent.probabilities[0].label, "true");
  assert.equal(result.is_urgent.probabilities[1].label, "false");

  // Categorical check
  assert.equal(result.dept.id, "dept");
  assert.equal(result.dept.label, "billing");
  assert.equal(result.dept.confidence, 0.75);
  assert.equal(result.dept.probabilities.length, 2);
  assert.equal(result.dept.probabilities[0].label, "billing");
  assert.equal(result.dept.probabilities[1].label, "tech");

  // Ordinal check
  assert.equal(result.severity.id, "severity");
  assert.equal(result.severity.label, "3");
  assert.equal(typeof result.severity.expectedScore, "number");
  assert.equal(result.severity.probabilities.length, 3);

  // measureContextUsage
  const usage = await classifier.measureContextUsage("Hello world test");
  assert.equal(usage, 3);

  // destroy
  classifier.destroy();
  await assert.rejects(() => classifier.classify("test"), /The classifier session has been destroyed/);
});

// learnings-aak: both accumulators in classifier-api.js are keyed by caller-supplied question ids,
// so both are null-prototype maps. `__proto__` is a valid id as far as create()'s validation goes
// (it checks only that the id is a non-empty string), and the playground reaches create() with the
// demo-5 schema textarea's JSON.parse output and no id validation in front of it.
const PROTO_ID = "__proto__";

// The caller's own path: the schema row is parsed from JSON that names the id as
// `{"id":"__proto__"}`, so JSON.parse yields an ordinary own key `id` whose string *value* is
// `__proto__` — the id reaches this loop as `q.id`, a legal non-empty string id, not as an own
// `__proto__` key of the parsed row. The trap is the plain `{}` map written by assignment with that
// untrusted id or label as the key —
// `layaQuestions[q.id] = ...` or `crit[label] = ...`: the key resolves to the accessor inherited
// from Object.prototype, so the setter runs — an object value replaces the map's prototype, a
// string value is ignored and no own property appears. A computed key in an object literal never
// does this: `{ [q.id]: q }` defines an own property and never touches the prototype.
function playgroundSchema() {
  return JSON.parse(
    '{"questions":[' +
      '{"id":"tool","type":"categorical","prompt":"Which tool?","options":[{"label":"list_tabs","description":"List them"},{"label":"other","description":"Something else"}]},' +
      '{"id":"__proto__","type":"binary","prompt":"Escalate?"}]}',
  ).questions;
}

test("Classifier API: toLayaQuestions keys a __proto__ question id as an own property (learnings-aak)", () => {
  const questions = playgroundSchema();
  const map = toLayaQuestions(questions, null);

  assert.deepEqual(Object.keys(map), ["tool", PROTO_ID]);
  assert.equal(Object.getPrototypeOf(map), null, "the map must be prototype-safe");
  assert.ok(Object.hasOwn(map, PROTO_ID), "the caller's __proto__ question must survive as an own key");
  assert.equal(map[PROTO_ID].t, "noul");
  assert.equal(map[PROTO_ID].ins, "Escalate?");

  // The context prefix is part of the mapping, so it is pinned through the same helper.
  const withContext = toLayaQuestions(playgroundSchema(), "Triage");
  assert.deepEqual(Object.keys(withContext), ["tool", PROTO_ID]);
  assert.equal(withContext[PROTO_ID].ins, "Triage\nEscalate?");
});

// toLayaQuestions() is only worth anything if the live API still routes through it. create()
// cannot run in node (getLaya() downloads and verifies the LiteRT checkpoints), so the
// delegation is pinned by reading the method body: an inline `{}` accumulator is a failure.
test("Classifier API: create() delegates its question mapping to toLayaQuestions (learnings-aak)", () => {
  const source = readFileSync(new URL("../site/decision-models/classifier-api.js", import.meta.url), "utf8");
  const create = source.slice(source.indexOf("static async create("), source.indexOf("get contextWindow"));
  assert.match(
    create,
    /new Classifier\(\s*session,\s*options,\s*toLayaQuestions\(/,
    "create() must map its questions through the helper under test",
  );
  assert.doesNotMatch(create, /const layaQuestions = \{\}/, "create() must not rebuild the map with a plain {} accumulator");
});

test("Classifier API: classify() returns a __proto__ question id as an own decision key (learnings-aak)", async () => {
  const questions = playgroundSchema();
  const classifier = new Classifier(new MockLayaSession(), { questions }, toLayaQuestions(questions, null));

  const result = await classifier.classify("Payment failed for invoice #4421");

  assert.deepEqual(Object.keys(result), ["tool", PROTO_ID]);
  assert.equal(Object.getPrototypeOf(result), null, "the returned map must be prototype-safe");
  assert.ok(Object.hasOwn(result, PROTO_ID), "the __proto__ decision must survive as an own key");
  assert.equal(result[PROTO_ID].id, PROTO_ID);
  assert.equal(result[PROTO_ID].label, "true");
  assert.equal(result[PROTO_ID].probability, 0.85);
  assert.equal(result[PROTO_ID].probabilities.length, 2);
  assert.equal(result.tool.id, "tool");
  assert.equal(result.tool.label, "list_tabs");

  classifier.destroy();
});

// learnings-e7p: the categorical criteria map inside toLayaQuestions() is keyed by the caller's
// option label, and `__proto__` is a legal label on the same textarea-JSON-to-create() path as the
// ids above. On a plain {} accumulator the assignment lands on the inherited accessor, which
// ignores a non-object value, so the option is dropped from the criteria map and renderOptions()
// names one option fewer than the caller did. With a single survivor left, laya-pack's
// buildSequence() rejects the whole sequence — "the inference contract needs at least two
// options"; with two or more survivors the run simply proceeds against the truncated question.
// decodeQuestion()'s "N logits for M choice criteria" guard does not fire on this drop — it and
// renderOptions() read the same criteria map, so their counts agree — it only catches a genuine
// disagreement between the gathered logits and the number of criteria.
test("Classifier API: toLayaQuestions keeps a __proto__ option label as an own criterion (learnings-e7p)", () => {
  // Mirrors the playground caller: the demo-5 schema textarea's text, JSON.parsed — `__proto__`
  // arrives as an ordinary string in the first option's `label` field, not as an own key of the
  // parsed rows — and handed to create().
  const questions = JSON.parse(
    '{"questions":[{"id":"tool","type":"categorical","prompt":"Which tool?","options":[' +
      '{"label":"__proto__","description":"Escalate?"},{"label":"list_tabs","description":"List them"}]}]}',
  ).questions;

  const crit = toLayaQuestions(questions, null).tool.crit;

  assert.deepEqual(
    Object.keys(crit),
    ["__proto__", "list_tabs"],
    "every option label the caller named comes back as a criterion key, in the caller's order",
  );
  assert.ok(
    Object.hasOwn(crit, "__proto__"),
    "a __proto__ option label must be an own property, not a reassignment of the map's prototype",
  );
  assert.equal(crit["__proto__"], "Escalate?");
  assert.equal(crit.list_tabs, "List them");
  assert.equal(
    Object.getPrototypeOf(crit),
    null,
    "the criteria accumulator is prototype-less, so no option label can reach an inherited accessor",
  );

  // The engine's own prompt builder reads this map with Object.entries(crit), so the label has to
  // survive as far as the rendered option text.
  assert.deepEqual(
    renderOptions({ t: "choice", crit }),
    ["__proto__: Escalate?", "list_tabs: List them"],
    "a __proto__ option is rendered into the prompt like any other option",
  );
});

test("Classifier API: toLayaQuestions maps ordinary categorical labels unchanged (learnings-e7p)", () => {
  const questions = [
    {
      id: "dept",
      type: "categorical",
      prompt: "Which team?",
      options: [
        { label: "billing", description: "Payments" },
        { label: "tech", description: "Bugs" },
        { label: "plain" },
        "ask_human",
      ],
    },
  ];

  const crit = toLayaQuestions(questions, null).dept.crit;

  // Spread into a plain object: the criteria are still content-identical to the pre-fix map, only
  // the prototype is (deliberately) different.
  assert.deepEqual({ ...crit }, {
    billing: "Payments",
    tech: "Bugs",
    plain: "plain",
    ask_human: "ask_human",
  });
  assert.deepEqual(Object.keys(crit), ["billing", "tech", "plain", "ask_human"], "label-index order is unchanged");
});
