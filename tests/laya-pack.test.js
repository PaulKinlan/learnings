// tests/laya-pack.test.js — the builder arithmetic and the decoder math, checked without a
// model, plus one end-to-end pin against the vendored tokenizer and the worked reference row
// published in site/decision-models/laya/HOST_CONTRACT.md §E (A01/department).
import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { Tokenizer } from "@huggingface/tokenizers";
import {
  ENGLISH_SPECIAL_IDS,
  buildSequence,
  renderOptions,
  serializeState,
  renderCriterion,
  softmax,
  tempBucket,
  selectTemperature,
  actFeatures,
  pyRound,
  decodeQuestion,
} from "../site/decision-models/laya-pack.js";

const IDS = ENGLISH_SPECIAL_IDS; // { cls: 50281, sep: 50282, pad: 50283, mask: 50284, unk: 50280 }
const MASK_STR = "[MASK]";

// A tokenizer where every character is its own token, so a position can be checked exactly.
const charTokenize = (text) => [...text].map((c) => c.codePointAt(0));

const ENGLISH_CONFIG = {
  temperature: [1.6369030475616455, 1.2514300346374512, 1.983399510383606],
  temperature_by_options: {
    "choice:3-5": 1.7601518630981445,
    "choice:6-10": 1.0000158548355103,
    "score:3-5": 1.2514300346374512,
    "noul:2": 1.983399510383606,
    "choice:11+": 0.10058280825614929,
    "choice:2": 1.9063563346862793,
  },
};

// ── rendering ────────────────────────────────────────────────────────────────

test("renderOptions: choice keeps label-only options and renders described ones", () => {
  const q = { t: "choice", ins: "?", crit: { billing: "invoices, refunds", other: "", zero: 0, off: false } };
  assert.deepEqual(renderOptions(q), [
    "billing: invoices, refunds",
    "other",
    "zero: 0",
    "off: false",
  ]);
});

test("renderOptions: score options are zero-based levels; noul is always [false, true]", () => {
  assert.deepEqual(renderOptions({ t: "score", ins: "?", crit: ["bad", "okay", "great"] }),
    ["level 0: bad", "level 1: okay", "level 2: great"]);
  assert.deepEqual(renderOptions({ t: "noul", ins: "?" }), [
    "false: no, the statement does not hold",
    "true: yes, the statement holds",
  ]);
  assert.deepEqual(renderOptions({ t: "noul", ins: "?", crit: { true: "it does" } }), [
    "false: no, the statement does not hold",
    "true: it does",
  ]);
});

test("serializeState uses Python's json.dumps spacing, never compact JS JSON", () => {
  // The graph was trained on Python-rendered text: {"a": 1, "b": [1, 2]} — with the spaces.
  const state = { from: "Velqun", subject: "Duplicate charge", body: "Please refund." };
  assert.equal(
    serializeState(state),
    '{"from": "Velqun", "subject": "Duplicate charge", "body": "Please refund."}',
  );
  assert.equal(serializeState([1, true, null, "x"]), '[1, true, null, "x"]');
  assert.equal(serializeState("a plain string"), "a plain string");
  assert.equal(renderCriterion({ a: [1, 2] }), '{"a": [1, 2]}');
});

// ── buildSequence with the exact char tokenizer ─────────────────────────────

test("every option's marker position lands on the slot immediately before its mask id", async () => {
  const { ids, markers } = await buildSequence({
    tokenize: charTokenize,
    state: "charged twice",
    question: { t: "choice", ins: "Which team?", crit: { billing: "", support: "" } },
    maxLen: 512,
    headMaxLen: 192,
    ids: IDS,
    maskString: MASK_STR,
  });
  assert.equal(ids[0], IDS.cls);
  for (const m of markers) {
    assert.equal(ids[m], IDS.mask, "the marker position holds the option's mask id");
  }
  assert.equal(markers.length, 2);
  // [CLS] + head + [SEP] + opt0 + opt1 + [SEP] + state + [SEP]
  assert.equal(ids[ids.length - 1], IDS.sep);
  const sepAfterOptions = ids.indexOf(IDS.sep, markers[1]);
  assert.equal(ids.slice(sepAfterOptions + 1).join(""), charTokenize("charged twice").concat([IDS.sep]).join(""));
});

test("literal [MASK] strings in caller text become one space and never a marker id", async () => {
  const { ids, markers } = await buildSequence({
    tokenize: charTokenize,
    state: "s",
    question: { t: "choice", ins: "x [MASK] y", crit: { "a [MASK] b": "", c: "" } },
    maxLen: 512,
    headMaxLen: 192,
    ids: IDS,
    maskString: MASK_STR,
  });
  assert.equal(markers.length, 2);
  // The instruction contained one literal mask string, replaced by a space — no stray 50284.
  assert.equal(ids.filter((id) => id === IDS.mask).length, 2, "only the two option markers remain");
});

test("the state is right-truncated to the room left by head and options", async () => {
  const longState = "s".repeat(600);
  const { ids, markers } = await buildSequence({
    tokenize: charTokenize,
    state: longState,
    question: { t: "choice", ins: "q", crit: { a: "", b: "" } },
    maxLen: 512,
    headMaxLen: 192,
    ids: IDS,
    maskString: MASK_STR,
  });
  assert.equal(ids.length, 512, "the sequence slices to the window");
  assert.equal(markers.length, 2, "both markers survive a full state");
  // The retained state is the FRONT of the text (truncate_left=false).
  const firstSep = ids.indexOf(IDS.sep);
  const lastSep = ids.lastIndexOf(IDS.sep);
  const statePart = ids.slice(ids.indexOf(IDS.sep, markers[1]) + 1, lastSep);
  assert.ok(statePart.length < longState.length);
  assert.equal(statePart[0], "s".codePointAt(0));
  assert.ok(firstSep > 0);
});

test("a question whose options cannot all fit the window is rejected before any graph call", async () => {
  await assert.rejects(() =>
    buildSequence({
      tokenize: charTokenize,
      state: "s",
      question: { t: "choice", ins: "q", crit: Object.fromEntries(Array.from({ length: 20 }, (_, i) => [`option-${i}`, "a description that is not short"])) },
      maxLen: 64,
      headMaxLen: 192,
      ids: IDS,
      maskString: MASK_STR,
    }),
    /options exceed head_max_len/,
  );
});

test("option squeezing keeps every marker when the head budget is blown", async () => {
  const crit = Object.fromEntries(Array.from({ length: 12 }, (_, i) => [`opt${i}`, "d".repeat(60)]));
  const { markers } = await buildSequence({
    tokenize: charTokenize,
    state: "s",
    question: { t: "choice", ins: "q", crit },
    maxLen: 512,
    headMaxLen: 192,
    ids: IDS,
    maskString: MASK_STR,
  });
  assert.equal(markers.length, 12, "§B.5: squeezing per-option slices rather than dropping markers");
});

// ── decoder math ─────────────────────────────────────────────────────────────

test("temperature bucketing and the bucket-first lookup order", () => {
  assert.equal(tempBucket(2), "2");
  assert.equal(tempBucket(5), "3-5");
  assert.equal(tempBucket(10), "6-10");
  assert.equal(tempBucket(11), "11+");
  // A01 is choice with K=4: the bucket 3-5 wins over the choice default.
  assert.equal(selectTemperature(ENGLISH_CONFIG, "choice", 4), 1.7601518630981445);
  // noul K=2 has a bucket entry; score K=7 falls back to the score default.
  assert.equal(selectTemperature(ENGLISH_CONFIG, "noul", 2), ENGLISH_CONFIG.temperature_by_options["noul:2"]);
  assert.equal(selectTemperature(ENGLISH_CONFIG, "score", 7), ENGLISH_CONFIG.temperature[1]);
});

test("pyRound ties to even on the represented value", () => {
  assert.equal(pyRound(0.6750863194465637, 4), 0.6751);
  // Exact binary ties: 2.5 and 7.5 are exactly representable, so the even rule is exercised.
  assert.equal(pyRound(0.25, 1), 0.2);  // 2.5 -> even is 2
  assert.equal(pyRound(0.75, 1), 0.8);  // 7.5 -> even is 8
  assert.equal(pyRound(0.00005, 4), 0.0001); // the stored double sits just above the tie —
  // and scaling the double would wrongly call it a tie and round to even (0), unlike Python
  assert.equal(pyRound(1.0, 4), 1.0);
});

test("act features come from raw probabilities, never calibrated ones", () => {
  const pRaw = softmax([1.0703915357589722, -1.8068784475326538, -2.2441372871398926, -2.46530818939209]);
  const [top1, gap, entropy, kScaled] = actFeatures(pRaw, 4);
  const sorted = [...pRaw].sort((a, b) => b - a);
  assert.ok(Math.abs(top1 - sorted[0]) < 1e-12);
  assert.ok(Math.abs(gap - (sorted[0] - sorted[1])) < 1e-12);
  assert.ok(entropy >= 0 && entropy <= 1);
  assert.equal(kScaled, 4 / 255);
});

// The worked row, HOST_CONTRACT.md §E A01/department: the decoder must reproduce the
// captured probabilities and the captured answer exactly from the captured logits.
test("A01/department: captured logits decode to the captured answer", () => {
  const rawLogits = [1.0703915357589722, -1.8068784475326538, -2.2441372871398926, -2.46530818939209];
  const actProbability = softmax([3710.220947265625, -3030.86669921875])[0];
  const q = {
    t: "choice",
    ins: "Which department should handle this request?",
    crit: {
      billing: "invoices, payments, refunds",
      technical: "bugs, outages, system errors",
      sales: "pricing, new contracts",
      other: "everything else",
    },
  };
  const answer = decodeQuestion({ q, rawLogits, actProbability, config: ENGLISH_CONFIG });
  assert.equal(answer.type, "choice");
  assert.equal(answer.choice, "billing");
  assert.deepEqual(answer.probabilities, {
    billing: 0.6751,
    technical: 0.1317,
    sales: 0.1027,
    other: 0.0906,
  });
  assert.equal(answer.action.act_probability, 1.0);
});

// ── the end-to-end sequence pin with the vendored real tokenizer ────────────

const A01_SEQUENCE = [
  50281, 22122, 1953, 27, 6758, 7811, 943, 6016, 436, 2748, 32, 50282,
  50284, 33484, 27, 29838, 1271, 13, 10762, 13, 1275, 41748,
  50284, 7681, 27, 19775, 13, 562, 1131, 13, 985, 6332,
  50284, 6224, 27, 20910, 13, 747, 12712,
  50284, 643, 27, 3253, 2010,
  50282,
  9819, 4064, 1381, 346, 37599, 82, 328, 995, 346, 19091, 1381, 346, 24900, 21821, 4179, 995, 346,
  2915, 1381, 346, 7845, 23005, 253, 21036, 4179, 449, 94,
  50282,
];
const A01_MARKERS = [12, 22, 32, 39];

test("A01/department: the built sequence matches the gate-0 capture id for id", async () => {
  const tok = new Tokenizer(
    JSON.parse(readFileSync(new URL("../site/decision-models/laya/en-tokenizer.json", import.meta.url), "utf8")),
    {},
  );
  const tokenize = (text) => tok.encode(text, { add_special_tokens: false }).ids;

  const { ids, markers } = await buildSequence({
    tokenize,
    state: { from: "Velqun", subject: "Duplicate charge", body: "Please refund the duplicate charge." },
    question: {
      t: "choice",
      ins: "Which department should handle this request?",
      crit: {
        billing: "invoices, payments, refunds",
        technical: "bugs, outages, system errors",
        sales: "pricing, new contracts",
        other: "everything else",
      },
    },
    maxLen: 512,
    headMaxLen: 192,
    ids: IDS,
    maskString: MASK_STR,
  });
  assert.deepEqual(ids, A01_SEQUENCE, "the byte-for-byte sequence of the worked row");
  assert.deepEqual(markers, A01_MARKERS);
  assert.equal(ids.length, 73, "the worked row's sequence_length");
});

// ── the multilingual gate rows: sequence and markers must match the publisher's ──

const ML_IDS = { cls: 2, sep: 1, pad: 0, unk: 3, mask: 4 };
const ML_MASK_STR = "<mask>";
const QTYPE_NAME = { 0: "choice", 1: "score", 2: "noul" };

test("multilingual gate rows: the built sequences match the publisher's captures exactly", async () => {
  const tok = new Tokenizer(
    JSON.parse(readFileSync(new URL("../site/decision-models/laya/ml-tokenizer.json", import.meta.url), "utf8")),
    {},
  );
  const tokenize = (text) => tok.encode(text, { add_special_tokens: false }).ids;
  const rows = JSON.parse(
    readFileSync(new URL("../site/decision-models/laya/gate_rows_s256.json", import.meta.url), "utf8"),
  ).rows;

  // One row per (language, question-type) combination — EN, JA and mixed across all three types.
  const seen = new Map();
  for (const row of rows) {
    const key = `${row.language}:${row.qtype}`;
    if (!seen.has(key)) seen.set(key, row);
  }
  assert.equal(seen.size, 9, "the sample covers every language×type combination in the fixture");

  for (const [key, row] of seen) {
    const q = row.question;
    const question = { t: QTYPE_NAME[row.qtype], ins: q.instructions, crit: q.criteria ?? {} };
    const { ids, markers } = await buildSequence({
      tokenize,
      state: row.state,
      question,
      maxLen: row.window,
      headMaxLen: 256,
      ids: ML_IDS,
      maskString: ML_MASK_STR,
    });
    assert.deepEqual(ids, row.sequence_ids, `${key} (${row.row_id}): sequence ids differ`);
    assert.deepEqual(markers, row.marker_positions, `${key} (${row.row_id}): marker positions differ`);
  }
});
