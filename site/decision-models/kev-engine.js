// The Kev decision engine, shared by on-device.html and playground.html.
//
// Kev is a 0.6 B decision model from onnx-community/kev-0.6b-ONNX: one state and any number
// of typed questions packed into a single sequence (kev-pack.js does the index arithmetic),
// read as a comparison between hidden states at delimiter tokens.
//
// The runtime here is ONNX Runtime Web's own JS API (vendor/ort.bundle.min.mjs) driving the
// JSEP wasm build (vendor/ort-wasm-simd-threaded.jsep.{mjs,wasm}): the q4 weights use
// GatherBlockQuantized, which registers ONLY through the jsep bridge — the asyncify and
// plain wasm builds lack jsepInit entirely, and transformers.js's embedded ORT never calls
// it either (all three measured failing with "Could not find an implementation for
// GatherBlockQuantized(1)"). The jsep loader + the ort bundle's wrapper is the one working
// CPU path for this model, so that is what this engine uses — no transformers.js involved.
//
// The tokenizer is @huggingface/tokenizers (pure JS, no ORT inside). The q4 weights
// download once from huggingface.co — the only network request.

import * as ort from "../vendor/ort.bundle.min.mjs";
import { Tokenizer } from "../vendor/tokenizers.min.mjs";
import {
  escapeUserText,
  readDelimiterIds,
  packDecision,
  readAnswers,
} from "./kev-pack.js";

const REPO = "onnx-community/kev-0.6b-ONNX";

const VENDOR = new URL("../vendor/", import.meta.url).href;
ort.env.wasm.wasmPaths = {
  mjs: new URL("ort-wasm-simd-threaded.jsep.mjs", VENDOR).href,
  wasm: new URL("ort-wasm-simd-threaded.jsep.wasm", VENDOR).href,
};
ort.env.logLevel = "warning";

const SPECIAL_TOKEN_FILES = ["special_tokens_map.json", "added_tokens.json"];

/**
 * Load tokenizer and weights. Nothing downloads until this is called — never on page load.
 * `onProgress(stage, received, total)` reports the weight download. `urls` overrides asset
 * URLs (tests serve local copies instead of re-downloading).
 */
export async function loadKev({ onProgress = null, urls = {} } = {}) {
  const urlFor = (file) => urls[file] ?? `https://huggingface.co/${REPO}/resolve/main/${file}`;
  const notify = (stage, loaded = 0, total = 0, file = stage) => {
    onProgress?.({ stage, file, loaded, total, received: loaded }, loaded, total, file);
  };

  notify("tokenizer", 0, 0, "tokenizer.json");
  const [tokenizerJson, config] = await Promise.all([
    fetch(urlFor("tokenizer.json")).then((r) => r.text()),
    fetch(urlFor("config.json")).then((r) => r.json()),
  ]);
  const tokenizer = new Tokenizer(JSON.parse(tokenizerJson), {});
  const ids = readDelimiterIds(config?.kev);

  notify("weights", 0, 0, "onnx/model_q4.onnx_data");
  const onnxUrl = urls["onnx/model_q4.onnx"] ?? `https://huggingface.co/${REPO}/resolve/main/onnx/model_q4.onnx`;
  const onnxDataUrl = urls["onnx/model_q4.onnx_data"] ?? `https://huggingface.co/${REPO}/resolve/main/onnx/model_q4.onnx_data`;
  const t0 = performance.now();
  const session = await ort.InferenceSession.create(onnxUrl, {
    executionProviders: ["wasm"],
    // The q4 weights live in an external-data file; mount it by the exact name the graph
    // references, or deserialization dies with "Module.MountedFiles is not available".
    externalData: [{ path: "model_q4.onnx_data", data: onnxDataUrl }],
  }).catch((error) => {
    throw new Error(`the q4 weights did not create a session (${error.message})`);
  });

  return new KevSession({
    tokenizer,
    session,
    ids,
    checkpoint: `${REPO} (q4 weights, CPU/ONNX Runtime Web jsep)`,
    loadMs: performance.now() - t0,
  });
}

export class KevSession {
  constructor({ tokenizer, session, ids, checkpoint, loadMs }) {
    this.tokenizer = tokenizer;
    this.session = session;
    this.ids = ids;
    this.checkpoint = checkpoint;
    this.loadMs = loadMs;
    this.cache = new Map();
  }

  tokenize(text) {
    const escaped = escapeUserText(text);
    if (!this.cache.has(escaped)) {
      const encoded = this.tokenizer.encode(escaped, { add_special_tokens: false });
      this.cache.set(escaped, encoded.ids);
    }
    return this.cache.get(escaped);
  }

  /**
   * One packed pass over a state and any number of typed questions.
   * @returns {{distributions: Array, tokens: number, ms: number, flat: boolean}}
   */
  async decidePacked(state, questions) {
    const texts = [state];
    for (const q of questions) texts.push(q.instruction, ...q.options);
    const tokenize = (text) => this.tokenize(text);

    const { inputIds, ends } = packDecision({
      ids: this.ids,
      tokenize,
      state,
      questions,
    });

    const started = performance.now();
    const feeds = {
      input_ids: new ort.Tensor("int64", BigInt64Array.from(inputIds, BigInt), [1, inputIds.length]),
      attention_mask: new ort.Tensor("int64", new BigInt64Array(inputIds.length).fill(1n), [1, inputIds.length]),
    };
    const results = await this.session.run(feeds);
    const logits = results.logits ?? results[Object.keys(results)[0]];
    const scores = Array.from(logits.data, Number);
    const ms = performance.now() - started;

    // A graph that found no delimiters returns zeros rather than raising; a uniform
    // distribution is that failure speaking, not a confident tie.
    const flat = new Set(scores.map((v) => v.toFixed(6))).size === 1;

    return { distributions: readAnswers({ scores, ends }), tokens: inputIds.length, ms, flat };
  }

  /** What actually ran, for honest backend reporting. */
  backendReport() {
    return {
      requested: "wasm (ONNX Runtime Web, jsep build)",
      providers: ["wasm"],
      webgpuPresent: typeof navigator !== "undefined" && "gpu" in navigator,
      crossOriginIsolated: typeof crossOriginIsolated === "boolean" && crossOriginIsolated,
    };
  }

  async delete() {
    try {
      await this.session?.release?.();
    } catch {
      /* releasing twice is fine */
    }
  }
}
