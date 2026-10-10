// Packing and decoding for Laya's single-row decision format.
//
// A JS port of the host contract vendored beside it (laya/ML-HOST_CONTRACT.md §B and §D),
// which itself copies laya 0.3.4's common.py and agent.py verbatim. Like kev-pack.js this
// module is pure: a tokenize function and the special-token ids come in as data, so every
// index and every probability can be checked without loading a graph. The worked reference
// rows at the end of ML-HOST_CONTRACT.md are the anchors for tests/laya-pack.test.js.

export const QTYPE_INDEX = { choice: 0, score: 1, noul: 2 };
export const QTYPE_ONEHOT = { choice: [1, 0, 0], score: [0, 1, 0], noul: [0, 0, 1] };

// The literal mask-token string of the English (ModernBERT) checkpoint. Section B step 3/4/7:
// each occurrence in caller text becomes one ASCII space, never a marker.
export const ENGLISH_MASK_STRING = "[MASK]";
export const ENGLISH_SPECIAL_IDS = { cls: 50281, sep: 50282, pad: 50283, mask: 50284, unk: 50280 };

// All special:true literals in the vendored en-tokenizer.json and ml-tokenizer.json
// added_tokens tables. The selected checkpoint's mask is replaced first, as before.
const READER_SPECIAL_STRINGS = [
  "<|padding|>", "<|endoftext|>", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]",
  "<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<start_of_turn>", "<end_of_turn>",
];

function neutralizeReaderText(text, maskString) {
  let safe = String(text).replaceAll(maskString, " ");
  for (const special of READER_SPECIAL_STRINGS) safe = safe.replaceAll(special, " ");
  return safe;
}

/** Python's json.dumps(value, ensure_ascii=False, separators=(", ", ": ")) — the spacing
 * matters: the graph was trained on Python-rendered text, and a compact JS stringify is a
 * different string. Float formatting follows JS Number, not Python repr (noted limitation:
 * integral floats render "1" rather than Python's "1.0"). */
function pyJson(value) {
  if (value === null || value === undefined) return "null";
  const t = typeof value;
  if (t === "string") return JSON.stringify(value);
  if (t === "number") {
    if (!Number.isFinite(value)) throw new Error("non-finite number cannot be serialized into a Laya sequence");
    return String(value);
  }
  if (t === "boolean") return value ? "true" : "false";
  if (Array.isArray(value)) return "[" + value.map(pyJson).join(", ") + "]";
  if (t === "object") {
    return "{" + Object.entries(value).map(([k, v]) => `${JSON.stringify(k)}: ${pyJson(v)}`).join(", ") + "}";
  }
  return JSON.stringify(String(value)); // Python's default=str
}

/** render_criterion: strings pass through; structured values become compact JSON with
 * Python separators (contract §B step 2). */
export function renderCriterion(value) {
  return typeof value === "string" ? value : pyJson(value);
}

/** serialize_state: a string state is used unchanged; dict/list state becomes
 * json.dumps(state, ensure_ascii=False) — Python default separators, i.e. ", " and ": ". */
export function serializeState(state) {
  return typeof state === "string" ? state : pyJson(state);
}

/** render_options, label-index order, noul always [false, true] (contract §B step 2).
 * Only null/"" mean "no description"; 0 and false are real descriptions. */
export function renderOptions(q) {
  const crit = q.crit;
  if (q.t === "choice") {
    if (!crit || typeof crit !== "object" || Array.isArray(crit))
      throw new Error("a choice question needs an insertion-ordered criteria object");
    return Object.entries(crit).map(([k, v]) =>
      v === null || v === undefined || v === "" ? k : `${k}: ${renderCriterion(v)}`);
  }
  if (q.t === "score") {
    if (!Array.isArray(crit)) throw new Error("a score question needs a criteria array");
    return crit.map((c, i) => `level ${i}: ${renderCriterion(c)}`);
  }
  if (q.t === "noul") {
    const c = crit && typeof crit === "object" ? crit : {};
    const absent = (v) => v === null || v === undefined || v === "";
    return [
      "false: " + (absent(c.false) ? "no, the statement does not hold" : renderCriterion(c.false)),
      "true: " + (absent(c.true) ? "yes, the statement holds" : renderCriterion(c.true)),
    ];
  }
  throw new Error(`unknown question type '${q.t}' — expected choice, score or noul`);
}

/**
 * build_sequence (contract §B, verbatim port):
 *   [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 … [SEP] <state> [SEP]
 * with right padding applied by the engine, not here.
 *
 * @param {object} arg
 * @param {(text: string) => number[]} arg.tokenize  encode WITHOUT special tokens
 * @param {string|object|Array} arg.state
 * @param {{t: string, ins: any, crit: any}} arg.question
 * @param {number} arg.maxLen       the graph window N (256 or 512)
 * @param {number} arg.headMaxLen   the builder head budget (192 for English)
 * @param {{cls: number, sep: number, mask: number}} arg.ids
 * @param {string} arg.maskString   the checkpoint's literal mask token ("[MASK]" for English)
 * @param {number[]} [arg.optionOrder]  identity when omitted; duplicate orders are rejected
 * @returns {{ids: number[], markers: number[]}} marker positions are zero-based, pre-padding
 */
export async function buildSequence({ tokenize, state, question, maxLen, headMaxLen, ids, maskString, optionOrder = null }) {
  if (typeof tokenize !== "function") throw new Error("buildSequence needs a tokenize function");
  if (!Number.isInteger(maxLen) || !Number.isInteger(headMaxLen)) throw new Error("maxLen and headMaxLen are required integers");
  const q = question;
  const opts = renderOptions(q);
  if (opts.length < 2) throw new Error("the inference contract needs at least two options");
  const order = optionOrder ?? opts.map((_, i) => i);
  if (order.length !== opts.length || new Set(order).size !== order.length)
    throw new Error("option_order must be a permutation of the options");

  // §B.3: literal mask strings become one space; all other special literals are also inert.
  const ins = neutralizeReaderText(q.ins, maskString);
  let headIds = await tokenize(`${q.t} question: ${ins}`);

  // §B.4: special literals -> space, ONE leading ASCII space, at most 48 text tokens per option,
  // with the integer mask id prepended (so an unsqueezed option occupies at most 49 slots).
  let optIds = [];
  for (const i of order) {
    optIds.push([ids.mask, ...(await tokenize(" " + neutralizeReaderText(opts[i], maskString))).slice(0, 48)]);
  }

  // §B.5: the squeezing rule — only when the head budget is blown.
  let optBudget = headMaxLen - optIds.reduce((acc, o) => acc + o.length, 0);
  if (optBudget < 16) {
    const per = Math.max(4, Math.floor((headMaxLen - 16) / Math.max(1, optIds.length)));
    optIds = optIds.map((o) => o.slice(0, per));
    optBudget = headMaxLen - optIds.reduce((acc, o) => acc + o.length, 0);
  }
  headIds = headIds.slice(0, Math.max(8, optBudget));

  // §B.6: [CLS] head [SEP], then each option with its zero-based marker position recorded
  // BEFORE it, then the option-ending [SEP].
  const out = [ids.cls, ...headIds, ids.sep];
  const markers = [];
  for (const o of optIds) {
    markers.push(out.length);
    out.push(...o);
  }
  out.push(ids.sep);

  // §B.7: the state is serialized, special literals spaced, tokenized without special tokens.
  const stateIds = await tokenize(neutralizeReaderText(serializeState(state), maskString));

  // §B.8: right-truncate the state to the remaining room, append the final [SEP], slice to N.
  const room = Math.max(0, maxLen - out.length - 1);
  const finalIds = [...out, ...stateIds.slice(0, room), ids.sep];
  const sliced = finalIds.slice(0, maxLen);
  const keptMarkers = markers.filter((m) => m < maxLen);

  // §B.9: a retained marker whose option was squeezed is valid; fewer markers than options
  // is the only rejection, and it happens before either graph is called.
  if (keptMarkers.length !== opts.length) {
    throw new Error(
      `question options exceed head_max_len=${headMaxLen}: ` +
        `only ${keptMarkers.length} of ${opts.length} marker positions fit in window ${maxLen}`,
    );
  }
  return { ids: sliced, markers: keptMarkers };
}

// ── decoding (contract §D) ─────────────────────────────────────────────────────

export function softmax(logits) {
  if (!logits.length) throw new Error("softmax of an empty logit list");
  const m = Math.max(...logits);
  const ex = logits.map((z) => Math.exp(z - m));
  const sum = ex.reduce((a, b) => a + b, 0);
  return ex.map((v) => v / sum);
}

/** temp_bucket: 2 / 3-5 / 6-10 / 11+ (§D step 4). */
export function tempBucket(K) {
  return K <= 2 ? "2" : K <= 5 ? "3-5" : K <= 10 ? "6-10" : "11+";
}

/** Bucket first, qtype default second — the contract's exact lookup order. */
export function selectTemperature(config, qtype, K) {
  const bucket = `${qtype}:${tempBucket(K)}`;
  const byOptions = config.temperature_by_options ?? {};
  if (bucket in byOptions) return byOptions[bucket];
  const index = QTYPE_INDEX[qtype];
  if (index === undefined) throw new Error(`unknown question type '${qtype}'`);
  return config.temperature[index];
}

/** The four act-head features, from RAW probabilities — never calibrated ones (§D step 2). */
export function actFeatures(pRaw, K) {
  const k = Math.max(K, 2);
  const sorted = [...pRaw].sort((a, b) => b - a);
  const top1 = sorted[0];
  const top2 = sorted.length > 1 ? sorted[1] : 0;
  const entropy = -pRaw.reduce((acc, p) => acc + p * Math.log(Math.max(p, 1e-9)), 0) / Math.log(k);
  return [top1, top1 - top2, entropy, k / 255];
}

/** Python round(value, 4) — ties to even on the true decimal expansion of the binary
 * value, which is how CPython rounds (via correctly-rounded decimal conversion). Scaling a
 * double instead would invent ties that do not exist: 0.00005 * 1e4 === 0.5 exactly in IEEE
 * arithmetic, but the stored value of 0.00005 sits just ABOVE the tie (its expansion is
 * 0.0000500000000000000239…) and Python rounds it UP. A double's expansion deviates from any
 * shortest-decimal tie within 17 significant digits; 18 guard digits from a correctly-rounded
 * toFixed cover that. */
export function pyRound(value, ndigits = 4) {
  if (!Number.isFinite(value)) return value;
  const neg = value < 0 || Object.is(value, -0);
  const GUARD = 18;
  const s = Math.abs(value).toFixed(ndigits + GUARD);
  const kept = s.slice(0, s.length - GUARD);
  const guard = s.slice(s.length - GUARD);
  const tie = "5" + "0".repeat(GUARD - 1);
  let roundUp;
  if (guard > tie) roundUp = true;
  else if (guard < tie) roundUp = false;
  else roundUp = (kept.charCodeAt(kept.length - 1) - 48) % 2 === 1; // exact tie: to even
  const step = 10 ** -ndigits;
  let result = parseFloat(kept);
  if (roundUp) result = parseFloat((result + step).toFixed(ndigits));
  return neg ? -result : result;
}

export function choiceConfidence(p, K) {
  if (K < 2) return 1;
  const entropy = -p.reduce((acc, v) => acc + v * Math.log(Math.min(Math.max(v, 1e-12), 1)), 0) / Math.log(K);
  return Math.min(Math.max(1 - entropy, 0), 1);
}

/**
 * Decode one question row (§D steps 4–6). Inputs are the gathered marker logits in ORIGINAL
 * option order, the act-head's softmax class-0 probability, the temperature config, and the
 * question itself (for labels and legend). Intermediate probabilities are never rounded.
 */
export function decodeQuestion({ q, rawLogits, actProbability, config }) {
  const K = rawLogits.length;
  if (K < 2) throw new Error("decoding needs at least two options");
  const pRaw = softmax(rawLogits);
  const T = selectTemperature(config, q.t, K);
  const p = softmax(rawLogits.map((z) => z / Math.max(1e-3, T)));
  const action = { act_probability: pyRound(actProbability, 4) };

  if (q.t === "choice") {
    const labels = Object.keys(q.crit);
    if (labels.length !== K) throw new Error(`${K} logits for ${labels.length} choice criteria`);
    let best = 0;
    for (let i = 1; i < K; i++) if (p[i] > p[best]) best = i; // first argmax on ties
    return {
      type: "choice",
      choice: labels[best],
      probabilities: Object.fromEntries(labels.map((l, i) => [l, pyRound(p[i], 4)])),
      confidence: pyRound(choiceConfidence(p, K), 4),
      action,
    };
  }
  if (q.t === "score") {
    const expectation = p.reduce((acc, v, i) => acc + i * v, 0);
    return {
      type: "score",
      score: pyRound(expectation, 4),
      legend: Object.fromEntries(q.crit.map((c, i) => [String(i), c])),
      probabilities: Object.fromEntries(q.crit.map((_, i) => [String(i), pyRound(p[i], 4)])),
      confidence: pyRound(choiceConfidence(p, K), 4),
      action,
    };
  }
  if (q.t === "noul") {
    const p1 = p[1];
    return {
      type: "noul",
      noul: pyRound(p1, 4),
      confidence: pyRound(Math.max(p1, 1 - p1), 4),
      action,
    };
  }
  throw new Error(`unknown question type '${q.t}'`);
}
