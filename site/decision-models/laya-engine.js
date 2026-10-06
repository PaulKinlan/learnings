// The Laya decision engine: real local inference with the litert-community Laya graphs on
// Google's LiteRT.js runtime, in this tab, with nothing sent anywhere.
//
// Two checkpoints are described here, one used:
//
//   MULTILINGUAL (the default and the only browser-viable one today): the
//   external-embedding form from litert-community/Laya-Multilingual-LiteRT. The 393 MB
//   token table lives in JS memory as float16 and is looked up here per row; the graph
//   itself is 251 MB, which is what compiles in this runtime. The host contract is
//   laya/ML-HOST_CONTRACT.md; the tokenizer is Gemma-style BPE, 256,000 ids.
//
//   ENGLISH (documented, disabled): the token-id form's fp32-weight graphs are
//   844 MB–1.68 GB and blow the wasm32 heap at compile time in LiteRT.js today
//   (measured: litert_wasm_internal, RuntimeError: memory access out of bounds). The
//   work is correct — the act head compiles, and laya-pack.js pins the gate-0 captures
//   id-for-id — the memory ceiling is the runtime's, not the model's.
//
// The main graph's weights download once from huggingface.co (the ONLY network requests, each
// from an immutable commit revision and each verified byte-for-byte against the publisher's
// SHA256SUMS — an integrity check and not a root of trust, see the note in loadLaya); the act
// head, the contract texts and this module are served from this site. Packing and decoding live
// in laya-pack.js, pinned id-for-id against the publisher's gate captures.

import { loadLiteRt, loadAndCompile, Tensor } from "../vendor/litert/index.js";
import { Tokenizer } from "../vendor/tokenizers.min.mjs";
import {
  ENGLISH_MASK_STRING,
  ENGLISH_SPECIAL_IDS,
  QTYPE_ONEHOT,
  actFeatures,
  buildSequence,
  decodeQuestion,
  softmax,
} from "./laya-pack.js";
import { LAYA_REVISION, LAYA_SHA256SUMS_SHA256 } from "./laya-manifest.js";

export const MULTILINGUAL_SPECIAL_IDS = Object.freeze({ cls: 2, sep: 1, pad: 0, unk: 3, mask: 4 });
export const MULTILINGUAL_MASK_STRING = "<mask>";

const ML_REPO = "litert-community/Laya-Multilingual-LiteRT";
const EN_REPO = "litert-community/laya-LiteRT";

const CHECKPOINTS = {
  multilingual: {
    repo: ML_REPO,
    revision: LAYA_REVISION.multilingual,
    label: "multilingual (mmBERT-base, 100+ languages)",
    window: 512,
    headMaxLen: 256,
    ids: MULTILINGUAL_SPECIAL_IDS,
    maskString: MULTILINGUAL_MASK_STRING,
    pooledDim: 768,
    embedding: "external-fp16-table",
    files: {
      main: "laya_ml_s512_embeds_wfp16.tflite", // 251,806,912 bytes
      act: "laya_ml_act_head_fp32.tflite", // 795,816 bytes — the ML head, never the English one
      embeddings: "token_embeddings_fp16.bin", // 393,216,000 bytes, [256000,768] fp16
      tokenizer: "tokenizer.json", // 34,363,188 bytes
      calibration: "laya_ml_calibration.json",
    },
    localFiles: {
      SHA256SUMS: "ml-SHA256SUMS",
      act: "laya_ml_act_head_fp32.tflite",
      tokenizer: "ml-tokenizer.json",
      calibration: "laya_ml_calibration.json",
    },
  },
  english: {
    repo: EN_REPO,
    revision: LAYA_REVISION.english,
    label: "english (ModernBERT-large) — disabled: fp32-weight graphs exceed the wasm32 heap",
    window: 512,
    headMaxLen: 192,
    ids: ENGLISH_SPECIAL_IDS,
    maskString: ENGLISH_MASK_STRING,
    pooledDim: 1024,
    embedding: "in-graph",
    files: {
      main: "laya_en_s512_wfp16.tflite",
      act: "laya_act_head_fp32.tflite",
      tokenizer: "tokenizer.json",
      calibration: null, // temperatures are the upstream config's (see laya/rl_agent_config.json)
    },
    localFiles: {
      SHA256SUMS: "en-SHA256SUMS",
      act: "laya_act_head_fp32.tflite",
      tokenizer: "en-tokenizer.json",
    },
    disabled: "LiteRT.js wasm32 compile OOM measured 2026-09-28 (see module comment)",
  },
};

const LAYA_DIR = new URL("./laya/", import.meta.url).href;

/**
 * The pinned hub URL for one artifact, by checkpoint name and its path inside that repo. The
 * revision is a commit sha (laya-manifest.js), never a branch or a tag, so the bytes behind the
 * URL cannot change after the fact — and neither can the SHA256SUMS fetched from the same place.
 */
export function layaArtifactUrl(checkpoint, file) {
  const spec = CHECKPOINTS[checkpoint];
  if (!spec) throw new Error(`unknown checkpoint '${checkpoint}' (known: ${Object.keys(CHECKPOINTS).join(", ")})`);
  return `https://huggingface.co/${spec.repo}/resolve/${spec.revision}/${file}`;
}

let litertPromise = null;
function litert() {
  // One global runtime load. The directory path lets LiteRT.js pick the best WASM variant
  // the browser supports (relaxed SIMD when present, the compat build otherwise).
  litertPromise ??= loadLiteRt(new URL("../vendor/litert/", import.meta.url).href);
  return litertPromise;
}

async function verifySha256(bytes, sha256, label) {
  if (!sha256) throw new Error(`no pinned sha256 for ${label}`);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== sha256.toLowerCase()) {
    throw new Error(
      `integrity check failed for ${label}: sha256 ${hex.slice(0, 16)}… does not match the publisher's ${sha256.slice(0, 16)}…`,
    );
  }
}

/** Fetch a URL into a Uint8Array with progress callbacks and a required sha256 check. */
async function fetchBytes(url, { onProgress = null, sha256, label = url } = {}) {
  if (!sha256) throw new Error(`no pinned sha256 for ${label}`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`download failed for ${label}: HTTP ${res.status}`);
  const total = Number(res.headers.get("content-length")) || 0;
  const reader = res.body.getReader();
  const chunks = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress?.(received, total);
  }
  const bytes = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    bytes.set(chunk, offset);
    offset += chunk.length;
  }
  await verifySha256(bytes, sha256, label);
  return bytes;
}

function parseSha256Sums(text) {
  const out = {};
  for (const line of text.split("\n")) {
    const match = line.match(/^([0-9a-f]{64})[ *]+(.+)$/);
    if (match) out[match[2].trim()] = match[1];
  }
  return out;
}

// float16 -> float32 conversion. A one-time 65536-entry lookup table (256 KB) so the
// per-token gather is two memory reads and no bit math per value.
let fp16Table = null;
function fp16Lut() {
  if (fp16Table) return fp16Table;
  const lut = new Float32Array(65536);
  for (let u = 0; u < 65536; u++) {
    const sign = u & 0x8000 ? -1 : 1;
    const exp = (u & 0x7c00) >> 10;
    const frac = u & 0x03ff;
    lut[u] =
      exp === 0
        ? sign * Math.pow(2, -14) * (frac / 1024)
        : exp === 0x1f
          ? frac
            ? NaN
            : sign * Infinity
          : sign * Math.pow(2, exp - 15) * (1 + frac / 1024);
  }
  fp16Table = lut;
  return lut;
}

/**
 * Load the engine. Nothing downloads until this is called — never on page load.
 * `onProgress(stage, received, total)` reports the weight downloads so a slow connection
 * shows progress. `urls` overrides individual asset URLs (used by tests to serve local
 * copies instead of re-downloading from the hub).
 */
export async function loadLaya({ checkpoint = "multilingual", calibration = "fitted", window = null, onProgress = null, urls = {} } = {}) {
  const baseSpec = CHECKPOINTS[checkpoint];
  const spec = window ? { ...baseSpec, window } : baseSpec;
  if (!spec) throw new Error(`unknown checkpoint '${checkpoint}' (known: ${Object.keys(CHECKPOINTS).join(", ")})`);
  if (spec.disabled) throw new Error(`checkpoint '${checkpoint}' is disabled: ${spec.disabled}`);

  onProgress?.("runtime", 0, 0);
  await litert();

  const localMap = spec.localFiles ?? {};
  const urlFor = (key, file) =>
    urls[file] ??
    (localMap[key] ? new URL(localMap[key], LAYA_DIR).href : layaArtifactUrl(checkpoint, file));
  const sumsUrl = urls.SHA256SUMS ??
    (localMap.SHA256SUMS ? new URL(localMap.SHA256SUMS, LAYA_DIR).href : layaArtifactUrl(checkpoint, "SHA256SUMS"));

  // SHA256SUMS is an INTEGRITY check only. It is NOT a root of trust. It catches corruption in
  // transit and an accidental mismatch against the digests recorded here, but the sums are the
  // publisher's own — the copy vendored under laya/, or the same file at the pinned revision,
  // both authored by whoever published the artifacts. A publisher who replaces a graph can
  // replace its hash in the same push, and this check will pass. The revision pin in
  // laya-manifest.js limits which published bytes and which published hashes are in play; it
  // does not make them trustworthy. Reviewers must not read this as signature verification.
  //
  // A sums file that does not arrive is a hard failure, on purpose: an empty digest table would
  // turn every check below into no check at all, which is the silent fallback this refuses.
  // The multilingual sums file itself is checked against the digest in laya-manifest.js.
  const sumsResponse = await fetch(sumsUrl);
  if (!sumsResponse.ok) {
    throw new Error(
      `SHA256SUMS unavailable for ${spec.repo}@${spec.revision}: HTTP ${sumsResponse.status} (${sumsUrl})`,
    );
  }
  const sumsBytes = new Uint8Array(await sumsResponse.arrayBuffer());
  await verifySha256(sumsBytes, LAYA_SHA256SUMS_SHA256[checkpoint], `${spec.repo}/SHA256SUMS`);
  const sums = parseSha256Sums(new TextDecoder().decode(sumsBytes));

  // Overrides use their final URL's filename (and hence their own published hash).
  // Vendored filenames may differ from the upstream names, but are byte-identical to
  // the pinned revision's artifacts; look those up by their upstream names instead.
  const basename = (url) => String(url).split("/").pop().split("?")[0];
  const digestFor = (key, file, url) => sums[
    urls[file] == null && localMap[key] ? file : basename(url)
  ];

  onProgress?.("tokenizer", 0, 0);
  const tokenizerUrl = urlFor("tokenizer", spec.files.tokenizer);
  const tokenizer = new Tokenizer(
    JSON.parse(new TextDecoder().decode(await fetchBytes(tokenizerUrl, {
      sha256: digestFor("tokenizer", spec.files.tokenizer, tokenizerUrl),
      label: `${spec.repo}/${spec.files.tokenizer}`,
    }))),
    {},
  );
  const mainUrl = urlFor("main", spec.files.main);
  const actUrl = urlFor("act", spec.files.act);
  const fetchJobs = {
    main: fetchBytes(mainUrl, {
      onProgress: (received, total) => onProgress?.("weights", received, total),
      sha256: digestFor("main", spec.files.main, mainUrl),
      label: `${spec.repo}/${spec.files.main}`,
    }),
    act: fetchBytes(actUrl, {
      sha256: digestFor("act", spec.files.act, actUrl),
      label: `${spec.repo}/${spec.files.act}`,
    }),
  };
  if (spec.files.embeddings) {
    const embUrl = urlFor("embeddings", spec.files.embeddings);
    fetchJobs.embeddings = fetchBytes(embUrl, {
      onProgress: (received, total) => onProgress?.("embedding-table", received, total),
      sha256: digestFor("embeddings", spec.files.embeddings, embUrl),
      label: `${spec.repo}/${spec.files.embeddings} (host-side token table)`,
    });
  }
  onProgress?.("weights", 0, 0);
  const jobEntries = Object.entries(fetchJobs);
  const resolved = await Promise.all(jobEntries.map(([, job]) => job));
  const { main: mainBytes, act: actBytes, embeddings: embeddingBytes } = Object.fromEntries(
    jobEntries.map(([name], i) => [name, resolved[i]]),
  );

  onProgress?.("compile", 0, 0);
  const [main, act] = await Promise.all([
    loadAndCompile(mainBytes, { accelerator: "wasm" }),
    loadAndCompile(actBytes, { accelerator: "wasm" }),
  ]);

  let embeddings = null;
  if (embeddingBytes) {
    if (embeddingBytes.byteLength % 2) throw new Error("the fp16 token table has an odd byte length");
    embeddings = new Uint16Array(
      embeddingBytes.buffer,
      embeddingBytes.byteOffset,
      embeddingBytes.byteLength / 2,
    );
  }

  const calibrationUrl = spec.files.calibration
    ? urlFor("calibration", spec.files.calibration)
    : new URL("rl_agent_config.json", LAYA_DIR).href;
  const calibrationFile = spec.files.calibration ?? "rl_agent_config.json";
  const calibrationJson = JSON.parse(new TextDecoder().decode(await fetchBytes(calibrationUrl, {
    sha256: digestFor("calibration", calibrationFile, calibrationUrl),
    label: `${spec.repo}/${calibrationFile}`,
  })));
  // The fitted calibration is the contract's production default. The publisher's saved
  // conversion GATES used the source config — all T=1, no buckets (ML-HOST_CONTRACT.md §D
  // note) — so "source" exists for gate-comparable answers and honest parity claims.
  const config = calibration === "source"
    ? { temperature: [1, 1, 1], temperature_by_options: {} }
    : {
        temperature: calibrationJson.temperature,
        temperature_by_options: calibrationJson.temperature_by_options,
      };

  const encode = (text) => tokenizer.encode(text, { add_special_tokens: false }).ids;

  return new LayaSession({ spec, checkpoint, main, act, tokenizer, encode, config, embeddings });
}

export class LayaSession {
  constructor({ spec, checkpoint, main, act, tokenizer, encode, config, embeddings }) {
    this.spec = spec;
    this.checkpointName = checkpoint;
    this.checkpoint = `${spec.repo} — ${spec.label}`;
    this.main = main;
    this.act = act;
    this.tokenizer = tokenizer;
    this.encode = encode;
    this.config = config;
    this.embeddings = embeddings; // Uint16Array fp16, [256000,768] row-major, when external
    this.lut = embeddings ? fp16Lut() : null;
    this.window = spec.window;
    this.pooledDim = spec.pooledDim;
  }

  /** [1,N,768] float32 inputs_embeds: the fp16 table's rows for the sequence ids,
   * padding with the actual PAD row (contract §C), converted to float32. */
  gatherEmbeddings(ids) {
    const N = this.window;
    const dim = this.pooledDim;
    const lut = this.lut;
    const table = this.embeddings;
    const out = new Float32Array(N * dim);
    for (let i = 0; i < N; i++) {
      const row = i < ids.length ? ids[i] : this.spec.ids.pad;
      const src = row * dim;
      const dst = i * dim;
      for (let d = 0; d < dim; d++) out[dst + d] = lut[table[src + d]];
    }
    return out;
  }

  /** One decision row: state + one typed question -> the contract's answer dictionary. */
  async decide(state, question) {
    const { ids, markers } = await buildSequence({
      tokenize: this.encode,
      state,
      question,
      maxLen: this.window,
      headMaxLen: this.spec.headMaxLen,
      ids: this.spec.ids,
      maskString: this.spec.maskString,
    });

    const N = this.window;
    const attentionMask = new Float32Array(N);
    attentionMask.fill(1, 0, ids.length);
    const qtypeOnehot = Float32Array.from(QTYPE_ONEHOT[question.t]);

    const inputs = {
      attention_mask: Tensor.fromTypedArray(attentionMask, [1, N]),
      qtype_onehot: Tensor.fromTypedArray(qtypeOnehot, [1, 3]),
    };
    if (this.spec.embedding === "external-fp16-table") {
      inputs.inputs_embeds = Tensor.fromTypedArray(this.gatherEmbeddings(ids), [1, N, this.pooledDim]);
    } else {
      const inputIds = new Int32Array(N).fill(this.spec.ids.pad);
      inputIds.set(ids);
      inputs.input_ids = Tensor.fromTypedArray(inputIds, [1, N]);
    }

    const outputs = await this.main.run(inputs);
    const tokenLogits = await outputs.token_logits.data();
    const pooledCls = await outputs.pooled_cls.data();
    const rawLogits = markers.map((m) => tokenLogits[m]);

    const pRaw = softmax(rawLogits);
    const feats = Float32Array.from(actFeatures(pRaw, rawLogits.length));
    const actOutputs = await this.act.run({
      pooled_cls: Tensor.fromTypedArray(pooledCls, [1, pooledCls.length]),
      feats: Tensor.fromTypedArray(feats, [1, 4]),
    });
    const actProbability = softmax(await actOutputs.act_logits.data())[0];

    const answer = decodeQuestion({
      q: question,
      rawLogits,
      actProbability,
      config: this.config,
    });
    return { answer, tokens: ids.length, markers: markers.length };
  }

  /** Several questions over one state, each its own row (the contract's single-row rule). */
  async decideAll(state, questions) {
    const answers = {};
    let inputTokens = 0;
    for (const [id, question] of Object.entries(questions)) {
      const { answer, tokens } = await this.decide(state, question);
      answers[id] = answer;
      inputTokens += tokens;
    }
    return {
      model: "laya-rl-agent",
      checkpoint: this.checkpoint,
      answers,
      usage: { input_tokens: inputTokens, output_tokens: 0 },
    };
  }

  delete() {
    for (const m of [this.main, this.act]) {
      try {
        m?.delete?.();
      } catch {
        /* deleting twice is fine */
      }
    }
    // The host-side token table is a JS-side 393 MB Uint16Array — drop the reference so
    // delete() actually reclaims it; deleting only the compiled graphs keeps it alive.
    this.embeddings = null;
    this.lut = null;
  }
}
