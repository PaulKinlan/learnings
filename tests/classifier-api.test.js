import { test } from "node:test";
import assert from "node:assert/strict";
import { Classifier } from "../site/decision-models/classifier-api.js";

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
