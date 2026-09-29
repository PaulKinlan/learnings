// The index arithmetic in kev-pack is the part that silently produced a flat distribution and
// four accidental passes when it was wrong. These checks run without a model.
import test from "node:test";
import assert from "node:assert/strict";
import {
  escapeUserText,
  readDelimiterIds,
  packDecision,
  softmax,
  readAnswers,
  choose,
  expectedLevel,
  noulProbability,
  verdict,
  NOUL_OPTIONS,
} from "../site/decision-models/kev-pack.js";

// A tokenizer where every character is its own token, so a position can be checked exactly.
const charTokenize = (text) => [...text].map((c) => c.codePointAt(0));

const IDS = { state: 151659, question: 151660, option_start: 151648, option_end: 151649, decide: 151661 };
const at = (c) => c.codePointAt(0);

test("every option's score position lands on that option's closing delimiter", () => {
  const questions = [
    { instruction: "Which team?", options: ["billing", "support"] },
    { instruction: "Urgent?", options: ["no", "yes"] },
  ];
  const { inputIds, ends } = packDecision({ ids: IDS, tokenize: charTokenize, state: "charged twice", questions });

  assert.equal(inputIds[0], IDS.state, "the sequence starts with the state delimiter");
  assert.equal(ends.length, questions.length);
  for (const [q, question] of questions.entries()) {
    assert.equal(ends[q].length, question.options.length);
    for (const [o, option] of question.options.entries()) {
      assert.equal(inputIds[ends[q][o]], IDS.option_end, `question ${q} option ${o} must be scored at a closing delimiter`);
      assert.equal(inputIds[ends[q][o] - option.length - 1], IDS.option_start, "the option text sits between its delimiters");
    }
  }
});

test("a second question's positions account for the first branch already in the sequence", () => {
  // The regression that mattered: branch-relative indices used as absolute ones. With two
  // branches of different lengths, a missing base offset puts question 2's scores on question 1.
  const first = { instruction: "a", options: ["x", "y"] };
  const second = { instruction: "a much longer instruction", options: ["xx", "yy", "zz"] };
  const { inputIds, ends, questionSpans } = packDecision({
    ids: IDS,
    tokenize: charTokenize,
    state: "s",
    questions: [first, second],
  });

  assert.ok(questionSpans[1][0] > questionSpans[0][1], "question 2 begins after question 1 ends");
  for (const [o, option] of second.options.entries())
    assert.equal(inputIds[ends[1][o]], IDS.option_end, `question 2 option ${o} must be scored at a closing delimiter`);
  assert.ok(ends[1][0] > ends[0][1], "question 2's scores lie after question 1's");
  for (const index of ends.flat()) assert.ok(index < inputIds.length, "no score position outside the sequence");
});

test("delimiter ids come from the config and are refused when absent or duplicated", () => {
  assert.deepEqual(readDelimiterIds({ delimiter_ids: IDS }), IDS);
  assert.throws(() => readDelimiterIds({}), /no kev\.delimiter_ids/);
  assert.throws(() => readDelimiterIds({ delimiter_ids: { state: 1, question: 2, option_start: 3, option_end: 4 } }), /decide/);
  assert.throws(() => readDelimiterIds({ delimiter_ids: { ...IDS, decide: IDS.state } }), /not distinct/);
});

test("caller text cannot forge a delimiter", () => {
  assert.equal(escapeUserText("<|fim_prefix|>"), "<\u00a6fim_prefix\u00a6>");
  assert.ok(!escapeUserText("<|fim_prefix|>").includes("<|"), "the escaped text no longer contains a delimiter");
  assert.equal(escapeUserText("plain text"), "plain text");
});

test("packing refuses input it cannot represent rather than emitting a sequence", () => {
  const ok = { instruction: "q", options: ["a", "b"] };
  assert.throws(() => packDecision({ ids: IDS, tokenize: charTokenize, state: "  ", questions: [ok] }), /non-empty/);
  assert.throws(() => packDecision({ ids: IDS, tokenize: charTokenize, state: "s", questions: [] }), /question is required/);
  assert.throws(
    () => packDecision({ ids: IDS, tokenize: charTokenize, state: "s", questions: [{ instruction: "q", options: ["a"] }] }),
    /at least two options/,
  );
  assert.throws(() => packDecision({ ids: { ...IDS, decide: undefined }, tokenize: charTokenize, state: "s", questions: [ok] }));
});

test("readout takes softmax within a question, never across questions", () => {
  const ends = [
    [3, 7],
    [12, 15, 19],
  ];
  const scores = new Array(24).fill(0);
  scores[7] = 2;
  scores[12] = 3;
  const answers = readAnswers({ scores, ends });
  assert.deepEqual(choose(answers[0]), 1, "the larger of the two option scores wins");
  assert.deepEqual(choose(answers[1]), 0);
  assert.ok(Math.abs(answers[0].reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(Math.abs(answers[1].reduce((a, b) => a + b, 0) - 1) < 1e-12);
  // A uniform score vector must produce a uniform distribution, which is how a broken graph looks.
  assert.deepEqual(readAnswers({ scores: new Array(24).fill(0), ends }), [
    [0.5, 0.5],
    [1 / 3, 1 / 3, 1 / 3],
  ]);
});

test("the three primitives read their distributions the documented way", () => {
  assert.deepEqual(NOUL_OPTIONS, ["no", "yes"]);
  assert.equal(noulProbability([0.25, 0.75]), 0.75, "noul reads p(yes) from the second option");
  assert.equal(expectedLevel([1, 0, 0]), 0);
  assert.equal(expectedLevel([0, 0, 1]), 2);
  assert.ok(Math.abs(expectedLevel([0.5, 0.5, 0]) - 0.5) < 1e-12, "a score may be fractional");
});

test("the verdict reports the unclear band instead of forcing an answer", () => {
  assert.equal(verdict(0.9), "yes");
  assert.equal(verdict(0.1), "no");
  assert.equal(verdict(0.152), "no", "below the midpoint reads as no under the published policy");
  // At a threshold of one half there is no unclear band at all: p >= 0.5 or p <= 0.5 covers
  // every probability. The band only exists once the threshold is pushed away from the middle.
  assert.equal(verdict(0.5), "yes");
  assert.equal(verdict(0.5, 0.8), "unclear");
  assert.equal(verdict(0.35, 0.8), "unclear");
  assert.equal(verdict(0.85, 0.8), "yes");
  assert.equal(verdict(0.15, 0.8), "no");
  assert.throws(() => verdict(0.5, 0), /between 0 and 1/);
});

test("softmax is stable on scores large enough to overflow a naive implementation", () => {
  const out = softmax([1000, 1001]);
  assert.ok(out.every((p) => Number.isFinite(p)), "no Infinity from exponentiating large scores");
  assert.ok(Math.abs(out[1] - 0.7310585786300049) < 1e-9);
});
