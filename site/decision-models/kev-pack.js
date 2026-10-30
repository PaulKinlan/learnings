// Packing and readout for Kev's single-sequence decision format.
//
// Kev is not a generator. One state and any number of typed questions are packed into one
// sequence, and the readout is a comparison between hidden states at particular tokens. This
// module is pure: it takes a tokenize function and the delimiter ids as data, so the index
// arithmetic can be checked without loading a model.
import { neutralizeReaderText } from "./tokenizer-specials.mjs";

/** Caller text can never produce a delimiter: <|name|> becomes <¦name¦> before tokenizing.
 * One layer of three (learnings-iif): packDecision additionally neutralizes the loaded
 * tokenizer's declared special literals — including the case-folded/NFKC aliases of
 * `normalized: true` tokens that this shape-regex cannot see — and drops every privileged id
 * that still survives tokenization. */
export const escapeUserText = (text) => String(text).replace(/<\|([A-Za-z0-9_]+)\|>/g, "<\u00a6$1\u00a6>");

/** The five delimiter ids. From the model's own config.json under `kev.delimiter_ids`. */
export const DELIMITER_KEYS = ["state", "question", "option_start", "option_end", "decide"];

/**
 * Read and check the delimiter ids published with the model.
 *
 * These must not be obtained by tokenizing the delimiter strings. `<|fim_prefix|>` and its four
 * siblings tokenize to one ordinary token when passed through the caller-text escape, and a
 * graph given four copies of that token finds no option spans at all: it returns a vector of
 * zeros rather than raising, and every question then reads as a uniform distribution.
 */
export function readDelimiterIds(kevConfig) {
  const ids = kevConfig && kevConfig.delimiter_ids;
  if (!ids || typeof ids !== "object")
    throw new Error(
      "This model's config.json has no kev.delimiter_ids, so the packed decision format cannot be built. " +
        "Use a checkpoint that publishes delimiter_ids, or add them to the config beside the delimiters.",
    );
  for (const key of DELIMITER_KEYS)
    if (!Number.isInteger(ids[key]))
      throw new Error(`kev.delimiter_ids.${key} is missing or not an integer; all five delimiters are required.`);
  if (new Set(DELIMITER_KEYS.map((k) => ids[k])).size !== DELIMITER_KEYS.length)
    throw new Error("kev.delimiter_ids are not distinct; the five delimiters must be five different tokens.");
  return ids;
}

/**
 * Build one sequence carrying the state and every question.
 *
 * Layout: state, then one branch per question. A branch is the question token, the instruction
 * text, then each option wrapped in option_start/option_end, then the decide token. The graph
 * derives the segments, positions and block-causal mask from the delimiter ids itself, so the
 * block-causal rule — each question sees the state and its own branch but not the others — is
 * not something the caller implements here. All the caller supplies is ids and a mask of ones.
 *
 * @param {string[]} [arg.specialStrings] special:true literals from the loaded tokenizer
 *   (tokenizerSpecials().literals). No floor list is applied when omitted: Kev's structural
 *   guarantee is the id drop below, and the engine always passes its tokenizer's own set.
 * @param {string[]} [arg.normalizedSpecialStrings] the subset the tokenizer matches AFTER
 *   normalization, matched case-insensitively and in NFKC form (tokenizerSpecials().normalizedLiterals)
 * @param {number[]} [arg.specialIds] every declared special id; any that survives tokenization
 *   is dropped from the sequence (tokenizerSpecials().ids)
 * @returns {{inputIds: number[], ends: number[][], questionSpans: Array<[number, number]>}}
 *   `ends[q][o]` is the index in inputIds of the token scored for option o of question q.
 */
export function packDecision({ ids, tokenize, state, questions, specialStrings = [], normalizedSpecialStrings = [], specialIds = [] }) {
  if (!ids || !tokenize) throw new Error("packDecision needs delimiter ids and a tokenize function.");
  if (typeof state !== "string" || !state.trim()) throw new Error("The state must be non-empty text.");
  if (!questions.length) throw new Error("At least one question is required.");
  for (const key of DELIMITER_KEYS) if (!Number.isInteger(ids[key])) throw new Error(`Delimiter id ${key} is missing.`);
  if (!Array.isArray(specialStrings) || specialStrings.some((s) => typeof s !== "string" || !s))
    throw new Error("specialStrings must be an array of non-empty strings");
  if (!Array.isArray(normalizedSpecialStrings) || normalizedSpecialStrings.some((s) => typeof s !== "string" || !s))
    throw new Error("normalizedSpecialStrings must be an array of non-empty strings");
  if (!Array.isArray(specialIds) || specialIds.some((id) => !Number.isInteger(id)))
    throw new Error("specialIds must be an array of integer token ids");
  const specials = {
    literals: [...new Set(specialStrings)].sort((a, b) => b.length - a.length),
    normalizedLiterals: [...new Set(normalizedSpecialStrings)],
  };
  // The three-part hardening Laya already has (learnings-xp4, learnings-gen), ported for
  // tm-kev-delimiter-token-id-filter-missing (learnings-iif). The five delimiter ids are ALWAYS
  // privileged: the graph derives segments, positions and the block-causal mask from the
  // delimiter ids themselves, so one delimiter id arriving from reader text inserts an extra
  // segment boundary and shifts which token is read as an option's score. Unlike Laya there is
  // no UNK exemption: Kev's byte-fallback vocabulary has no information-bearing UNK id, so
  // every declared special id is structure, not content.
  const privileged = new Set([...specialIds, ...DELIMITER_KEYS.map((key) => ids[key])]);
  const dropInjected = (tokens) => tokens.filter((id) => !privileged.has(id));
  // Text-level half: declared specials — including case-folded/NFKC aliases the tokenizer
  // matches after normalizing, which escapeUserText's shape-regex cannot see — become one
  // space. The id filter above is the normalization-agnostic half: whatever a normalizer did
  // to the text, a privileged id coming out is dropped.
  const neutralize = (text) => neutralizeReaderText(text, null, specials);

  const inputIds = [ids.state, ...dropInjected(tokenize(neutralize(state)))];
  const ends = [];
  const questionSpans = [];

  for (const q of questions) {
    const instruction = dropInjected(tokenize(neutralize(q.instruction)));
    if (!instruction.length) throw new Error("A question instruction must be non-empty text.");
    if (!q.options || q.options.length < 2)
      throw new Error("A question needs at least two options, or there is nothing to choose between.");
    const spans = q.options.map((option) => [ids.option_start, ...dropInjected(tokenize(neutralize(option))), ids.option_end]);

    const base = inputIds.length;
    const questionAt = base;
    // Within the branch: the question token, then the instruction, then the option spans.
    let cursor = 1 + instruction.length;
    const branchEnds = spans.map((span) => {
      cursor += span.length;
      return base + cursor - 1; // the option_end token is the last token of each span
    });

    inputIds.push(ids.question, ...instruction, ...spans.flat(), ids.decide);
    questionSpans.push([questionAt, inputIds.length - 1]);
    ends.push(branchEnds);
  }

  for (const [q, branchEnds] of ends.entries())
    for (const index of branchEnds)
      if (!(index > 0 && index < inputIds.length))
        throw new Error(`Option score position ${index} for question ${q} falls outside the sequence.`);

  return { inputIds, ends, questionSpans };
}

/**
 * Calibrated temperature for jaredpalmer/kev-0.6b (trial v7-06b/02-trial-2),
 * fitted on development rows to minimize NLL and ECE (raw ECE 0.0857 -> calibrated ECE 0.0317).
 * The ONNX export outputs uncalibrated logits (T = 1.0); applying this temperature rescales logits
 * to calibrated probabilities while preserving the argmax ranking exactly.
 */
export const KEV_CALIBRATED_TEMPERATURE = 1.9318726578496908;

/**
 * Softmax, subtracting the maximum so large scores cannot overflow to Infinity.
 * Supports optional temperature scaling (default T = 1.0 for raw logits).
 */
export function softmax(values, temperature = 1.0) {
  if (!values.length) throw new Error("softmax needs at least one value.");
  if (typeof temperature !== "number" || !Number.isFinite(temperature) || temperature <= 0) {
    throw new Error(`temperature must be a finite positive number, got ${String(temperature)}`);
  }
  const max = Math.max(...values);
  const exps = values.map((v) => Math.exp((v - max) / temperature));
  const total = exps.reduce((a, b) => a + b, 0);
  if (!Number.isFinite(total) || total <= 0) throw new Error("softmax produced no usable total.");
  return exps.map((e) => e / total);
}

/**
 * Turn the score at each option's final token into a distribution per question.
 *
 * The graph returns one score per token. Only the positions in `ends` matter; softmax is taken
 * within a question, never across questions. Optional temperature scaling defaults to 1.0 (raw).
 */
export function readAnswers({ scores, ends, temperature = 1.0 }) {
  return ends.map((branchEnds) => softmax(branchEnds.map((index) => scores[index]), temperature));
}

/** Choice: the option with the highest probability. Ties keep the earlier option. */
export function choose(distribution) {
  let best = 0;
  for (let i = 1; i < distribution.length; i++) if (distribution[i] > distribution[best]) best = i;
  return best;
}

/** Noul: p(second option), where options are ["no", "yes"]. */
export const NOUL_OPTIONS = ["no", "yes"];
export const noulProbability = (distribution) => distribution[1];

/** Score: the expected level over an ordered scale, which may be fractional. */
export function expectedLevel(distribution) {
  return distribution.reduce((sum, p, level) => sum + p * level, 0);
}

/**
 * A verdict for the published examples, and only that.
 *
 * Naming this a "threshold policy" rather than a calibrated probability is not hedging. On
 * question shapes unlike its training data this model ranks cases correctly while returning
 * probabilities well below the midpoint — a real credential-grab scored 0.142 and a harmless
 * message 0.055 on the same instruction. Both below 0.5, so a midpoint threshold answers "no"
 * to both and the correct ranking is thrown away. Judge the ranking first; move the threshold
 * only against labelled data.
 */
export function verdict(probability, threshold = 0.5) {
  if (!(threshold > 0 && threshold < 1)) throw new Error("The threshold must lie between 0 and 1.");
  if (probability >= threshold) return "yes";
  if (probability <= 1 - threshold) return "no";
  return "unclear";
}
