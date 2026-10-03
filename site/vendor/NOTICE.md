Vendored browser runtime
========================

This directory holds the JavaScript and WebAssembly needed to run a decision model in the
browser without loading a script from a third-party CDN. Scripts are served from this origin, so
the site's Content-Security-Policy can keep `script-src 'self'`.

| file | package | version | licence |
|---|---|---|---|
| ort.bundle.min.mjs | onnxruntime-web (JS API bundle) | 1.31.0-dev.20260914 | MIT |
| ort-wasm-simd-threaded.jsep.mjs | onnxruntime-web (JSEP glue) | 1.31.0-dev.20260914 | MIT |
| ort-wasm-simd-threaded.jsep.wasm | onnxruntime-web (JSEP wasm binary, 28 MB) | 1.31.0-dev.20260914 | MIT |

Notes that matter:

- The onnxruntime-web version is a **development build pinned by a date** rather than a release.
  It is what `@huggingface/transformers@4.3.0` depends on. Treat it as pinned, not as chosen.
- The **JSEP WebAssembly** binary (`ort-wasm-simd-threaded.jsep.wasm`, 28 MB) is the one vendored
  build because `GatherBlockQuantized(1)` registers only through `jsepInit`; the plain CPU and
  `asyncify` builds fail to deserialize the q4 Kev graph and have been removed.
- onnxruntime-web ships no licence file of its own; the MIT licence is declared in its
  package.json, which is where the licence column above comes from.

The model weights are **not** vendored. They are fetched from huggingface.co when a reader asks
for them.

LiteRT.js (@litertjs/core 2.5.3 + @litertjs/wasm-utils, Apache-2.0) — site/vendor/litert/.
The LiteRT.js web runtime (WebGPU/WebNN/WASM XNNPack) and its WASM payloads
(litert_wasm_internal / litert_wasm_compat_internal). One surgical patch: the
bare-specifier import of "@litertjs/wasm-utils" in index.js is rewritten to the
sibling ./wasm-utils.js so the module graph resolves without an import map.

@huggingface/tokenizers (Apache-2.0) — site/vendor/tokenizers.min.mjs (pure JS, no WASM),
license at site/vendor/TOKENIZERS-LICENSE.
