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
// from huggingface.co on demand (reloads may fetch them again). Model requests use immutable
// revisions, and each artifact is verified against kev-manifest.js before ORT receives it.

import * as ort from "../vendor/ort.bundle.min.mjs";
import { Tokenizer } from "../vendor/tokenizers.min.mjs";
import {
  escapeUserText,
  readDelimiterIds,
  packDecision,
  readAnswers,
} from "./kev-pack.js";
import { tokenizerSpecials } from "./tokenizer-specials.mjs";
import { KEV_REVISION, KEV_SHA256, KEV_MAX_BYTES, KEV_DEFAULT_MAX_BYTES } from "./kev-manifest.js";
import { ARTIFACT_DOWNLOAD_TIMEOUT_MS, fetchArtifactBytes } from "./fetch-policy.mjs";

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
 * match the pinned digest. The policy, the byte caps, the idle deadline and the verification all
 * live in fetch-policy.mjs (learnings-0ef) so kev and laya cannot drift; this wrapper pins the
 * Kev manifest as the cap chain and keeps the historical name for callers and tests.
 * `timeoutMs` bounds a STALL (idle with no progress), not the total transfer time, and must be a
 * finite positive number — anything else throws instead of silently aborting every download.
 */
export function fetchVerified(url, options = {}) {
  return fetchArtifactBytes(url, {
    timeoutMs: ARTIFACT_DOWNLOAD_TIMEOUT_MS,
    caps: KEV_MAX_BYTES,
    defaultMaxBytes: KEV_DEFAULT_MAX_BYTES,
    ...options,
  });
}

/**
 * Load tokenizer and weights. Nothing downloads until this is called — never on page load.
 * `onProgress(stage, received, total)` reports the weight download. `urls` overrides asset
 * URLs (tests serve local copies instead of re-downloading); it does not change the check, so
 * a local copy is still verified against the pinned digest of the artifact it stands in for.
 * `timeoutMs` is the idle deadline shared by every artifact download (tm-download-no-timeout):
 * it bounds a stalled hub, not the total transfer time, and must be finite and positive.
 */
export async function loadKev({ onProgress = null, urls = {}, timeoutMs = ARTIFACT_DOWNLOAD_TIMEOUT_MS } = {}) {
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
      timeoutMs,
    }),
    fetchVerified(urlFor("config.json"), {
      sha256: KEV_SHA256["config.json"],
      label: `${REPO}/config.json`,
      maxBytes: KEV_MAX_BYTES["config.json"],
      timeoutMs,
    }),
  ]);
  const tokenizerJson = JSON.parse(new TextDecoder().decode(tokenizerBytes));
  const tokenizer = new Tokenizer(tokenizerJson, {});
  const config = JSON.parse(new TextDecoder().decode(configBytes));
  const ids = readDelimiterIds(config?.kev);
  // What this checkpoint declares special is what must be neutralized in untrusted reader text:
  // a fixed list cannot know a checkpoint's own control tokens, and a literal match alone cannot
  // see a `normalized: true` alias (the tokenizer folds the text before matching it) — so
  // packDecision also drops any declared special id, and every delimiter id, that still comes
  // out of the tokenizer (learnings-iif, mirroring the Laya loader since learnings-xp4/gen).
  const specials = tokenizerSpecials(tokenizerJson);

  notify("weights", 0, 0, "onnx/model_q4.onnx_data");
  const t0 = performance.now();
  const [onnxBytes, onnxDataBytes] = await Promise.all([
    fetchVerified(urlFor("onnx/model_q4.onnx"), {
      sha256: KEV_SHA256["onnx/model_q4.onnx"],
      label: `${REPO}/onnx/model_q4.onnx`,
      maxBytes: KEV_MAX_BYTES["onnx/model_q4.onnx"],
      timeoutMs,
    }),
    fetchVerified(urlFor("onnx/model_q4.onnx_data"), {
      sha256: KEV_SHA256["onnx/model_q4.onnx_data"],
      label: `${REPO}/onnx/model_q4.onnx_data`,
      maxBytes: KEV_MAX_BYTES["onnx/model_q4.onnx_data"],
      timeoutMs,
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
    specials,
    checkpoint: `${REPO} (q4 weights, CPU/ONNX Runtime Web jsep)`,
    loadMs: performance.now() - t0,
  });
}

export class KevSession {
  constructor({ tokenizer, session, ids, checkpoint, loadMs, specials = null }) {
    this.tokenizer = tokenizer;
    this.session = session;
    this.ids = ids;
    this.specials = specials; // this checkpoint's special literals, normalized aliases and ids
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
   * Optional temperature scaling (default T = 1.0 for raw model logits).
   * @returns {{distributions: Array, tokens: number, ms: number, flat: boolean, temperature: number}}
   */
  async decidePacked(state, questions, { temperature = 1.0 } = {}) {
    const texts = [state];
    for (const q of questions) texts.push(q.instruction, ...q.options);
    const tokenize = (text) => this.tokenize(text);

    const { inputIds, ends } = packDecision({
      ids: this.ids,
      tokenize,
      state,
      questions,
      specialStrings: this.specials?.literals,
      normalizedSpecialStrings: this.specials?.normalizedLiterals,
      specialIds: this.specials?.ids,
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

    return {
      distributions: readAnswers({ scores, ends, temperature }),
      tokens: inputIds.length,
      ms,
      flat,
      temperature,
    };
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
    } finally {
      this.cache?.clear?.();
      this.session = null;
    }
  }
}
