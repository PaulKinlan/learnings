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
// The tokenizer is @huggingface/tokenizers (pure JS, no ORT inside). The artifacts download
// once from huggingface.co — the only network requests — each from an immutable revision and
// each verified against the digests in kev-manifest.js before it is handed to ORT.

import * as ort from "../vendor/ort.bundle.min.mjs";
import { Tokenizer } from "../vendor/tokenizers.min.mjs";
import {
  escapeUserText,
  readDelimiterIds,
  packDecision,
  readAnswers,
} from "./kev-pack.js";
import { KEV_REVISION, KEV_SHA256, KEV_MAX_BYTES, KEV_DEFAULT_MAX_BYTES } from "./kev-manifest.js";

const REPO = "onnx-community/kev-0.6b-ONNX";
// The pin: a commit sha, not a branch, so the bytes behind every URL are immutable.
const HUB_BASE = `https://huggingface.co/${REPO}/resolve/${KEV_REVISION}/`;

/** The pinned URL for one artifact, by its path inside the repo. */
export function kevArtifactUrl(file) {
  return `${HUB_BASE}${file}`;
}

const VENDOR = new URL("../vendor/", import.meta.url).href;
ort.env.wasm.wasmPaths = {
  mjs: new URL("ort-wasm-simd-threaded.jsep.mjs", VENDOR).href,
  wasm: new URL("ort-wasm-simd-threaded.jsep.wasm", VENDOR).href,
};
ort.env.logLevel = "warning";

const SPECIAL_TOKEN_FILES = ["special_tokens_map.json", "added_tokens.json"];

/**
 * Fetch one artifact into bytes, reporting progress, and refuse bytes whose SHA-256 does not
 * match the pinned digest. Downloads are strictly bounded by a maximum byte cap enforced
 * via AbortController while streaming (and checked against Content-Length before streaming begins)
 * to prevent memory exhaustion from unbounded responses (tm-unbounded-download-buffer).
 * There is deliberately no path that skips the check: the trust decision never depends on
 * the caller, so a substituted or corrupted artifact fails here rather than executing.
 * The digest is required — an absent one throws instead of passing.
 */
export async function fetchVerified(url, { onProgress = null, sha256, label = url, maxBytes = null } = {}) {
  if (!sha256) throw new Error(`no pinned sha256 for ${label}`);
  const basename = String(url).split("/").pop()?.split("?")[0] ?? "";
  const labelBasename = String(label).split("/").pop()?.split("?")[0] ?? "";
  const limit = maxBytes ?? KEV_MAX_BYTES[labelBasename] ?? KEV_MAX_BYTES[basename] ?? KEV_DEFAULT_MAX_BYTES;

  const controller = new AbortController();
  const res = await fetch(url, { signal: controller.signal });
  if (!res.ok) throw new Error(`download failed for ${label}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  if (limit != null && total > limit) {
    controller.abort();
    throw new Error(
      `download failed for ${label}: Content-Length ${total} exceeds maximum allowed size ${limit} bytes`,
    );
  }
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    received += value.length;
    if (limit != null && received > limit) {
      controller.abort();
      try { await reader.cancel(); } catch {}
      throw new Error(
        `download failed for ${label}: stream exceeded maximum allowed size ${limit} bytes (received ${received} bytes)`,
      );
    }
    chunks.push(value);
    onProgress?.(received, total);
  }
  const bytes = chunks.length === 1 ? chunks[0] : (() => {
    const b = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      b.set(chunk, offset);
      offset += chunk.length;
    }
    return b;
  })();
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== sha256.toLowerCase()) {
    throw new Error(
      `integrity check failed for ${label}: sha256 ${hex.slice(0, 16)}… does not match the pinned ${sha256.slice(0, 16)}…`,
    );
  }
  return bytes;
}

/**
 * Load tokenizer and weights. Nothing downloads until this is called — never on page load.
 * `onProgress(stage, received, total)` reports the weight download. `urls` overrides asset
 * URLs (tests serve local copies instead of re-downloading); it does not change the check, so
 * a local copy is still verified against the pinned digest of the artifact it stands in for.
 */
export async function loadKev({ onProgress = null, urls = {} } = {}) {
  const urlFor = (file) => urls[file] ?? kevArtifactUrl(file);
  const notify = (stage, loaded = 0, total = 0, file = stage) => {
    onProgress?.({ stage, file, loaded, total, received: loaded }, loaded, total, file);
  };

  notify("tokenizer", 0, 0, "tokenizer.json");
  const [tokenizerBytes, configBytes] = await Promise.all([
    fetchVerified(urlFor("tokenizer.json"), {
      sha256: KEV_SHA256["tokenizer.json"],
      label: `${REPO}/tokenizer.json`,
      maxBytes: KEV_MAX_BYTES["tokenizer.json"],
    }),
    fetchVerified(urlFor("config.json"), {
      sha256: KEV_SHA256["config.json"],
      label: `${REPO}/config.json`,
      maxBytes: KEV_MAX_BYTES["config.json"],
    }),
  ]);
  const tokenizer = new Tokenizer(JSON.parse(new TextDecoder().decode(tokenizerBytes)), {});
  const config = JSON.parse(new TextDecoder().decode(configBytes));
  const ids = readDelimiterIds(config?.kev);

  notify("weights", 0, 0, "onnx/model_q4.onnx_data");
  const t0 = performance.now();
  const [onnxBytes, onnxDataBytes] = await Promise.all([
    fetchVerified(urlFor("onnx/model_q4.onnx"), {
      sha256: KEV_SHA256["onnx/model_q4.onnx"],
      label: `${REPO}/onnx/model_q4.onnx`,
      maxBytes: KEV_MAX_BYTES["onnx/model_q4.onnx"],
    }),
    fetchVerified(urlFor("onnx/model_q4.onnx_data"), {
      sha256: KEV_SHA256["onnx/model_q4.onnx_data"],
      label: `${REPO}/onnx/model_q4.onnx_data`,
      maxBytes: KEV_MAX_BYTES["onnx/model_q4.onnx_data"],
      onProgress: (received, total) => notify("weights", received, total, "onnx/model_q4.onnx_data"),
    }),
  ]);
  const session = await ort.InferenceSession.create(onnxBytes, {
    executionProviders: ["wasm"],
    // The q4 weights live in an external-data file; mount it by the exact name the graph
    // references, or deserialization dies with "Module.MountedFiles is not available".
    externalData: [{ path: "model_q4.onnx_data", data: onnxDataBytes }],
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
