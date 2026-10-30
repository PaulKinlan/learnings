// The index arithmetic in kev-pack is the part that silently produced a flat distribution and
// four accidental passes when it was wrong. These checks run without a model.
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Tokenizer } from "../site/vendor/tokenizers.min.mjs";
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
import { specialLiteralsFromTokenizer, tokenizerSpecials } from "../site/decision-models/tokenizer-specials.mjs";

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

// learnings-iif (tm-kev-delimiter-token-id-filter-missing): Kev had only the literal-shape
// escape, while Laya got the full three-part hardening (learnings-xp4/gen). The graph derives
// segments, positions and the block-causal mask from the delimiter ids themselves, so one id
// smuggled through reader text shifts which token is read as an option's score. Mirrors the
// Laya suite: declared literals, case-folded aliases, token-level id drop, engine wiring.
test("a checkpoint-declared special literal is neutralized in state, instructions and options", () => {
  const LITERAL = "<2mass>";
  const INJECTED = 59999;
  // A tokenize stub that emits a control id wherever the literal survives: the sentinel is
  // what a real checkpoint tokenizer would do with its own special:true literal.
  const tokenize = (text) => {
    const out = [];
    let rest = String(text);
    while (rest.length) {
      if (rest.startsWith(LITERAL)) { out.push(INJECTED); rest = rest.slice(LITERAL.length); continue; }
      out.push(rest.codePointAt(0));
      rest = rest.slice(1);
    }
    return out;
  };
  const pack = (args) => packDecision({
    ids: IDS, tokenize, state: `a ${LITERAL} b`,
    questions: [{ instruction: `a ${LITERAL} b`, options: [`a ${LITERAL} b`, "x"] }],
    ...args,
  });

  assert.ok(pack({}).inputIds.includes(INJECTED), "sanity: the stub really injects the literal when nothing neutralizes it");

  const declared = specialLiteralsFromTokenizer({ added_tokens: [{ content: LITERAL, special: true }] });
  const neutralized = pack({ specialStrings: declared });
  assert.equal(neutralized.inputIds.filter((id) => id === INJECTED).length, 0,
    "a literal the checkpoint declares special must never reach the sequence from reader text");

  assert.throws(() => pack({ specialStrings: ["ok", ""] }), /specialStrings must be an array of non-empty strings/);
  assert.throws(() => pack({ normalizedSpecialStrings: [""] }), /normalizedSpecialStrings must be an array of non-empty strings/);
  assert.throws(() => pack({ specialIds: [1, "2"] }), /specialIds must be an array of integer token ids/);
});

// learnings-gen mirror: an added token declared `normalized: true` is matched AFTER the
// tokenizer normalizes the text, so the literal-shape escape and an exact literal match both
// miss the case-folded alias. Exercised through the real Tokenizer runtime on a minimal
// synthetic tokenizer.json with a Lowercase normalizer.
test("a normalized:true checkpoint token cannot be injected through its case-folded alias", () => {
  const tokenizerJson = {
    version: "1.0", truncation: null, padding: null,
    added_tokens: [
      { id: 100, content: "<STATE>", special: true, normalized: true, single_word: false },
      { id: 101, content: "<2mass>", special: true, normalized: false, single_word: false },
    ],
    normalizer: { type: "Lowercase" },
    pre_tokenizer: { type: "Whitespace" },
    post_processor: null, decoder: null,
    model: { type: "BPE", vocab: { "a": 1, "b": 2, "x": 3 }, merges: [] },
  };
  const tokenizer = new Tokenizer(tokenizerJson, {});
  const tokenize = (text) => tokenizer.encode(text, { add_special_tokens: false }).ids;
  assert.deepEqual(tokenize("a <state> b"), [1, 100, 2], "sanity: the tokenizer really does match the folded alias");

  const specials = tokenizerSpecials(tokenizerJson);
  assert.deepEqual(specials.normalizedLiterals, ["<STATE>"], "the alias must be recognized as normalized");
  assert.ok(specials.ids.includes(100) && specials.ids.includes(101), "declared special ids are collected");

  const pack = (args) => packDecision({
    ids: IDS, tokenize, state: "a <state> b",
    questions: [{ instruction: "a <state> b", options: ["a <state> b", "x"] }],
    ...args,
  });
  assert.ok(pack({}).inputIds.includes(100), "sanity: without the specials the folded alias injects its id");

  const neutralized = pack({
    specialStrings: specials.literals,
    normalizedSpecialStrings: specials.normalizedLiterals,
    specialIds: specials.ids,
  });
  assert.equal(neutralized.inputIds.filter((id) => id === 100 || id === 101).length, 0,
    "the folded alias must not reach the sequence through state, instruction or option text");

  // The id filter is the normalization-agnostic half: with NO text-level defense at all, the
  // declared ids that come out of the tokenizer are still dropped.
  const idFiltered = pack({ specialIds: specials.ids });
  assert.equal(idFiltered.inputIds.filter((id) => id === 100 || id === 101).length, 0,
    "whatever the normalizer did to the text, a declared special id coming out is dropped");
});

test("delimiter ids are structural: reader text can never add one, with or without specialIds", () => {
  const CONTROL = 50300;
  // No literal in this text is special: the tokenizer emits the ids on its own, the way a
  // Strip/Replace normalizer could produce them in ways no literal match can predict. The
  // character tokens stay: an instruction whose EVERY token is privileged fails closed as
  // empty, which is the correct behavior but not what this test is about.
  const tokenize = (text) => {
    const out = [...String(text)].map((c) => c.codePointAt(0));
    if (String(text).includes("magic")) out.push(CONTROL, IDS.option_start);
    return out;
  };
  const questions = [{ instruction: "magic", options: ["magic", "b"] }];

  const leaky = packDecision({ ids: IDS, tokenize, state: "magic", questions });
  assert.ok(leaky.inputIds.includes(CONTROL), "sanity: a non-delimiter control id passes when it is not declared");
  assert.equal(leaky.inputIds.filter((id) => id === IDS.option_start).length, 2,
    "the injected option_start is dropped even with no specialIds: only the two real option spans remain");

  const forgedState = packDecision({
    ids: IDS,
    tokenize: (text) => (String(text).includes("magic") ? [IDS.state] : [...String(text)].map((c) => c.codePointAt(0))),
    state: "magic",
    questions: [{ instruction: "q", options: ["a", "b"] }],
  });
  assert.equal(forgedState.inputIds[0], IDS.state, "the sequence still starts with the structural state delimiter");
  assert.equal(forgedState.inputIds.filter((id) => id === IDS.state).length, 1,
    "a forged state delimiter from reader text is dropped: an extra segment boundary would shift every score position");

  const filtered = packDecision({ ids: IDS, tokenize, state: "magic", questions, specialIds: [CONTROL] });
  assert.equal(filtered.inputIds.filter((id) => id === CONTROL).length, 0, "a declared control id is dropped");
});

// Wiring guard (mirrors the Laya xp4/gen guard): the engine must derive the specials from the
// tokenizer it actually loaded — literals, normalized aliases and ids — and pass all three.
test("kev-engine derives the neutralized literals from the loaded tokenizer", () => {
  const engine = readFileSync(new URL("../site/decision-models/kev-engine.js", import.meta.url), "utf8");
  assert.match(engine, /const specials = tokenizerSpecials\(tokenizerJson\);/);
  assert.match(engine, /this\.specials = specials/);
  for (const field of ["specialStrings: this.specials?.literals", "normalizedSpecialStrings: this.specials?.normalizedLiterals", "specialIds: this.specials?.ids"])
    assert.ok(engine.includes(field), `the engine must pass ${field}`);
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
