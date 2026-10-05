// Pinned identity of the Kev weights, checked in next to kev-engine.js.
//
// onnx-community/kev-0.6b-ONNX publishes no SHA256SUMS file, so unlike Laya there is no
// publisher manifest to fetch. KEV_REVISION is a commit sha — every resolve/ URL is therefore
// immutable, and an upstream push to the mutable main branch cannot change the bytes this
// engine runs. KEV_SHA256 was computed once from the files at that revision (2026-10-05) and
// is what kev-engine.js checks each artifact against before it reaches ONNX Runtime;
// re-pinning means editing both the revision and the digests together.
export const KEV_REVISION = "5a827091b3d58fe4fdfc02a5a5b2ce364047d678";

export const KEV_SHA256 = Object.freeze({
  "tokenizer.json": "c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539",
  "config.json": "40ce1c875acf016ab7742ecdc45d3bad0fe8c51d3ead0ca2263ec5eb4e96f42d",
  "onnx/model_q4.onnx": "db4657bb2f14f073cb4d289260dfc50af31adc1260a04cb2fe17c4f03da0881a",
  "onnx/model_q4.onnx_data": "20668ed757f5de294c254cab111910fbc37fb11addca8189e3f58b3c695f90f1",
});
