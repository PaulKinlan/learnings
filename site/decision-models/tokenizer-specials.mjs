// What a tokenizer.json declares special, and the text-level neutralizer for it — shared by
// BOTH packers (laya-pack.js, kev-pack.js) so the hardening cannot drift between the engines
// (learnings-iif; the one-copy rule from learnings-0ef). Three parts:
//   literals           special:true contents plus a built-in floor, longest-first
//   normalizedLiterals the subset the tokenizer matches AFTER normalizing (case-folded, NFKC)
//   ids                every declared special id, for the token-level drop that catches what
//                      no literal match can see (learnings-xp4, learnings-gen)
// Pure module: no engine, no network, no tokenizer runtime.

// Every special:true literal that has been seen in a tokenizer.json added_tokens table. This
// is only a floor: it can never be the full set for a checkpoint that adds its own control
// literals (<2mass>, [@BOS@], <unusedN>, [unusedN], …), so loaders pass what their own tokenizer
// declares (tokenizerSpecials) and untrusted reader text cannot smuggle a control token past the
// mask (learnings-xp4). The selected checkpoint's mask is replaced first, as section B of the host
// contract requires.
export const READER_SPECIAL_STRINGS = [
  "<|padding|>", "<|endoftext|>", "[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]",
  "<pad>", "<eos>", "<bos>", "<unk>", "<mask>", "<start_of_turn>", "<end_of_turn>",
];

/**
 * What a tokenizer.json declares must never be reachable from reader text.
 *
 * - `literals`: special:true contents (plus the built-in floor), longest-first.
 * - `normalizedLiterals`: the subset whose added token carries `normalized: true`. The tokenizer
 *   normalizes input before matching those, so a literal match is not enough — `<CTRL>` declared
 *   this way matches `<ctrl>` (learnings-gen).
 * - `ids`: every declared special id, so buildSequence can drop one that survived tokenization.
 *   This is the normalization-agnostic half of the defence: whatever a normalizer does to the
 *   text (case folding, NFKC, Strip, Replace) the id it produces is the same one the checkpoint
 *   treats as control.
 *
 * @param {unknown} tokenizerJson parsed tokenizer.json (added_tokens is optional)
 * @returns {{literals: string[], normalizedLiterals: string[], ids: number[]}}
 */
export function tokenizerSpecials(tokenizerJson) {
  const literals = new Set(READER_SPECIAL_STRINGS);
  const normalized = new Set();
  const ids = new Set();
  const added = Array.isArray(tokenizerJson?.added_tokens) ? tokenizerJson.added_tokens : [];
  for (const token of added) {
    if (token?.special !== true) continue;
    if (Number.isInteger(token.id)) ids.add(token.id);
    if (typeof token.content !== "string" || !token.content) continue;
    literals.add(token.content);
    if (token.normalized === true) normalized.add(token.content);
  }
  const byLength = (a, b) => b.length - a.length;
  return {
    literals: [...literals].sort(byLength),
    normalizedLiterals: [...normalized].sort(byLength),
    ids: [...ids].sort((a, b) => a - b),
  };
}

/** The literal list alone, for callers that only neutralize text. */
export function specialLiteralsFromTokenizer(tokenizerJson) {
  return tokenizerSpecials(tokenizerJson).literals;
}

const escapeLiteral = (value) => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");

/** Replace literal special tokens in reader text with one space (Laya
 * contract §B step 3; Kev packDecision applies it before its delimiter escape).
 * Literals a tokenizer marks `normalized: true` are matched case-insensitively and in their NFKC
 * form as well, because the tokenizer normalizes the text before matching them. Normalizers that
 * rewrite text in ways no literal match can predict (Strip, Replace) are caught downstream by the
 * token-id filter, which does not care how the text changed. */
export function neutralizeReaderText(text, maskString, { literals = READER_SPECIAL_STRINGS, normalizedLiterals = [] } = {}) {
  const foldedSet = new Set(normalizedLiterals);
  const exact = [];
  const folded = [];
  const add = (value) => {
    if (typeof value !== "string" || !value) return;
    if (foldedSet.has(value)) folded.push(value, value.normalize("NFKC"));
    else exact.push(value);
  };
  add(maskString);
  for (const literal of literals) add(literal);
  let safe = String(text);
  for (const literal of [...new Set(exact)].sort((a, b) => b.length - a.length)) safe = safe.replaceAll(literal, " ");
  for (const literal of [...new Set(folded)].filter(Boolean).sort((a, b) => b.length - a.length))
    safe = safe.replace(new RegExp(escapeLiteral(literal), "giu"), " ");
  return safe;
}
