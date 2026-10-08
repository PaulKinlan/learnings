// Pinned identity of the Laya hub repositories, checked in next to laya-engine.js.
//
// Unlike Kev, Laya's publishers ship their own SHA256SUMS, and this site vendors a copy of it
// under laya/ (ml-SHA256SUMS, en-SHA256SUMS) — so the per-artifact digests are not repeated
// here. What this file records is WHERE those bytes come from: each value is a commit sha, so
// every resolve/ URL laya-engine.js builds is immutable and an upstream push to a mutable main
// branch can change neither the artifacts this engine runs nor the hashes it checks them
// against. Re-pinning means editing a revision and re-vendoring that revision's SHA256SUMS
// together, then re-checking LAYA_SHA256SUMS_SHA256.
//
// Provenance, verified against huggingface.co on 2026-10-05:
//
//   multilingual — litert-community/Laya-Multilingual-LiteRT at 058cbf34… is the revision whose
//   SHA256SUMS is byte-identical to the vendored laya/ml-SHA256SUMS (sha256 4cb3d664…, 7780
//   bytes), and at that revision the LFS oids of laya_ml_s512_embeds_wfp16.tflite (dae207fa…),
//   token_embeddings_fp16.bin (58608f7b…), laya_ml_act_head_fp32.tflite (30da532a…) and
//   tokenizer.json (609d8f4c…) equal the digests the vendored file records, as does the
//   downloaded laya_ml_calibration.json (80e148c6…). main has since moved: at fd97044e (HEAD,
//   2026-10-05) that main graph is 2023399e…, so a main-ref fetch no longer matches the
//   vendored hashes at all.
//
//   multilingual, s256 — quant-smoke.mjs substitutes two graphs the list above does not name,
//   laya_ml_s256_embeds_fp32.tflite and laya_ml_s256_embeds_wfp16.tflite, for the main graph.
//   Their digests come from that same vendored sums file instead of being repeated here: each
//   has its own row in laya/ml-SHA256SUMS, and laya-engine.js selects the row by the fetched
//   file's own name, so a graph swapped into the main slot is checked against that graph's
//   digest, and fails closed when the row is absent or the bytes differ
//   (tests/laya-engine-integrity.test.js exercises both against the real file). The
//   per-artifact comparison above is narrower than the sums file on purpose — it covers only
//   the five artifacts it names — and every other row, the two s256 graphs included, rests on
//   the vendored file being the pinned revision's own, which is what the 4cb3d664… pin and its
//   re-derivation in tests/laya-engine.test.js establish.
//
//   english — litert-community/laya-LiteRT at 9baf244b… was that repo's HEAD when the vendored
//   laya/laya_act_head_fp32.tflite and laya/en-tokenizer.json were captured (this repo's commit
//   9f0d548, 2026-09-29); its SHA256SUMS lists c6f8de9b… and 6c8aaa9a… for those two files,
//   which is what the vendored copies hash to. The checkpoint stays disabled (wasm32 heap), and
//   the vendored laya/en-SHA256SUMS is NOT that repo's sums file — it is a copy of the
//   multilingual list, so it holds no digest for any laya_en_* graph. That gap predates this
//   pin and only matters if the english checkpoint is ever enabled.
export const LAYA_REVISION = Object.freeze({
  multilingual: "058cbf34ed2c6a2854ec60e9534c229d08740387",
  english: "9baf244b399a74d4a96a7846a1198b21b64cfdae",
});

// The sha256 of the vendored SHA256SUMS itself, per checkpoint, where that file really is the
// pinned revision's sums. tests/laya-engine.test.js re-derives it from the checked-in file, so
// editing one without the other fails offline rather than at a reader's download.
export const LAYA_SHA256SUMS_SHA256 = Object.freeze({
  multilingual: "4cb3d664b7483d57c319254f96f8dec0d1c168f6c6ec88748b61e703f0968dad",
});

// Maximum expected download size in bytes per artifact (derived from known publisher artifact sizes
// with reasonable tolerance, ~20%-50% headroom). Enforced during streaming by fetchBytes using
// AbortController to defend against unbounded download buffering / memory exhaustion
// (threat model tm-unbounded-download-buffer).
export const LAYA_MAX_BYTES = Object.freeze({
  "tokenizer.json": 50 * 1024 * 1024, // multilingual: 34,363,188 bytes (~34.4 MB) -> 50 MB cap
  "ml-tokenizer.json": 50 * 1024 * 1024,
  "en-tokenizer.json": 10 * 1024 * 1024, // english: 3,583,228 bytes (~3.6 MB) -> 10 MB cap
  "laya_ml_act_head_fp32.tflite": 4 * 1024 * 1024, // actual: 795,816 bytes (~796 KB) -> 4 MB cap
  "laya_act_head_fp32.tflite": 4 * 1024 * 1024, // actual: 1,057,960 bytes (~1.06 MB) -> 4 MB cap
  "laya_ml_calibration.json": 64 * 1024, // actual: 9,156 bytes (~9.2 KB) -> 64 KB cap
  "rl_agent_config.json": 64 * 1024, // actual: 745 bytes (~0.7 KB) -> 64 KB cap
  "token_embeddings_fp16.bin": 450 * 1024 * 1024, // actual: 393,216,000 bytes (~393.2 MB) -> 450 MB cap
  "laya_ml_s512_embeds_wfp16.tflite": 320 * 1024 * 1024, // actual: 251,806,912 bytes (~251.8 MB) -> 320 MB cap
  "laya_ml_s256_embeds_wfp16.tflite": 320 * 1024 * 1024, // actual: 250,889,408 bytes (~250.9 MB) -> 320 MB cap
  "laya_ml_s256_embeds_fp32.tflite": 600 * 1024 * 1024, // actual: 500,969,948 bytes (~501.0 MB) -> 600 MB cap
  "laya_en_s512_wfp16.tflite": 1000 * 1024 * 1024, // actual: 843,929,120 bytes (~843.9 MB) -> 1000 MB cap
  "SHA256SUMS": 1024 * 1024, // actual: 7,780 bytes (~7.8 KB) -> 1 MB cap
  "ml-SHA256SUMS": 1024 * 1024,
  "en-SHA256SUMS": 1024 * 1024,
});

export const LAYA_DEFAULT_MAX_BYTES = 1000 * 1024 * 1024; // 1 GB fallback cap

