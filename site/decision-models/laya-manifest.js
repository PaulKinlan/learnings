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
