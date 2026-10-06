// Stub network and LiteRT so integrity failures are observable without downloading weights.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash, webcrypto } from "node:crypto";
import { registerHooks } from "node:module";
import { LAYA_SHA256SUMS_SHA256 } from "../site/decision-models/laya-manifest.js";

registerHooks({
  resolve(specifier, context, nextResolve) {
    if (specifier === "../vendor/litert/index.js") {
      return { url: "data:text/javascript,export const loadLiteRt = async () => ({}); export const loadAndCompile = async () => ({}); export class Tensor {}", shortCircuit: true };
    }
    if (specifier === "../vendor/tokenizers.min.mjs") {
      return { url: "data:text/javascript,export class Tokenizer { constructor(json) { globalThis.__layaTokenizerConstructions++; this.json = json; } }", shortCircuit: true };
    }
    return nextResolve(specifier, context);
  },
});

// The alternate module is a scratch copy of the pre-fix engine for the falsification run.
const { loadLaya } = await import(process.env.LAYA_ENGINE_MODULE ?? "../site/decision-models/laya-engine.js");
const names = {
  main: "laya_ml_s512_embeds_wfp16.tflite",
  act: "laya_ml_act_head_fp32.tflite",
  embeddings: "token_embeddings_fp16.bin",
  tokenizer: "tokenizer.json",
  calibration: "laya_ml_calibration.json",
};
const encoder = new TextEncoder();
const payloads = Object.fromEntries(Object.entries(names).map(([key]) => [key,
  encoder.encode(key === "tokenizer" ? '{"fixture":"tokenizer"}' :
    key === "calibration" ? '{"temperature":[2,3,4],"temperature_by_options":{}}' : key),
]));
const sha = (bytes) => createHash("sha256").update(bytes).digest("hex");

async function withFixture({ missing, tampered, local = false, explicitVendoredTokenizer = false, alternateGraph = null, sumsUnavailable = false, sumsTampered = false } = {}, check) {
  const sums = Object.entries(names)
    .filter(([key]) => key !== missing)
    .map(([key, file]) => `${sha(payloads[key])}  ${file}\n`)
    .join("");
  const sumsBytes = encoder.encode(sums);
  const urls = Object.fromEntries(Object.entries(names).map(([key, file]) => [file,
    `https://fixture.invalid/${file}`,
  ]));
  urls.SHA256SUMS = explicitVendoredTokenizer ? "./laya/ml-SHA256SUMS" : "https://fixture.invalid/SHA256SUMS";
  if (explicitVendoredTokenizer) urls[names.tokenizer] = "./laya/ml-tokenizer.json";
  if (alternateGraph) urls[names.main] = `./laya/laya_ml_s256_${alternateGraph}.tflite`;
  const originalFetch = globalThis.fetch;
  const originalCrypto = Object.getOwnPropertyDescriptor(globalThis, "crypto");
  globalThis.__layaTokenizerConstructions = 0;
  globalThis.fetch = async (url) => {
    if (url === urls.SHA256SUMS) return sumsUnavailable ? new Response(null, { status: 404 }) :
      new Response(sumsTampered ? encoder.encode(`${sums}# changed\n`) : sumsBytes);
    const key = Object.keys(names).find((candidate) => url === (local && ["act", "tokenizer", "calibration"].includes(candidate)
      ? new URL(`../site/decision-models/laya/${{ act: names.act, tokenizer: "ml-tokenizer.json", calibration: names.calibration }[candidate]}`, import.meta.url).href
      : urls[names[candidate]]));
    if (!key) throw new Error(`unexpected fixture fetch: ${url}`);
    const changed = key === "tokenizer" ? '{"fixture":"changed"}' :
      key === "calibration" ? '{"temperature":[9,9,9],"temperature_by_options":{}}' : `tampered ${key}`;
    return new Response(tampered === key ? encoder.encode(changed) :
      alternateGraph && key === "main" ? encoder.encode("s256 graph") : payloads[key]);
  };
  // The real manifest pins the real sums. For these small synthetic sums only, supply the
  // fixture's pin; every artifact still goes through the real SHA-256 implementation.
  Object.defineProperty(globalThis, "crypto", { configurable: true, value: { subtle: { digest: (algorithm, bytes) =>
    Buffer.from(bytes).equals(Buffer.from(sumsBytes))
      ? Promise.resolve(Buffer.from(LAYA_SHA256SUMS_SHA256.multilingual, "hex"))
      : webcrypto.subtle.digest(algorithm, bytes) } } });
  try {
    await check({ urls: local ? { ...urls, [names.act]: undefined, [names.tokenizer]: undefined,
      [names.calibration]: undefined } : urls });
  } finally {
    globalThis.fetch = originalFetch;
    Object.defineProperty(globalThis, "crypto", originalCrypto);
    delete globalThis.__layaTokenizerConstructions;
  }
}

async function expectFailure(load, pattern, description) {
  await assert.rejects(load, (error) => {
    console.log(`${description}: ${error.message}`);
    assert.match(error.message, pattern);
    return true;
  });
}

test("missing or changed SHA256SUMS fails before artifact loading", async () => {
  await withFixture({ sumsUnavailable: true }, async ({ urls }) => {
    await expectFailure(loadLaya({ urls }), /SHA256SUMS unavailable for .*: HTTP 404/, "missing sums");
  });
  await withFixture({ sumsTampered: true }, async ({ urls }) => {
    await expectFailure(loadLaya({ urls }), /integrity check failed for .*\/SHA256SUMS/, "tampered sums");
  });
});

test("missing artifact digest fails closed and names the file", async () => {
  for (const key of ["main", "act", "embeddings"]) {
    await withFixture({ missing: key }, async ({ urls }) => {
      await expectFailure(loadLaya({ urls }), new RegExp(`no pinned sha256 for .*/${names[key]}`), `missing ${key}`);
    });
  }
});

for (const key of ["tokenizer", "calibration"]) {
  test(`missing ${key} digest fails closed`, async () => {
    await withFixture({ missing: key }, async ({ urls }) => {
      await expectFailure(loadLaya({ urls }), new RegExp(`no pinned sha256 for .*/${names[key]}`), `missing ${key}`);
      if (key === "tokenizer") assert.equal(globalThis.__layaTokenizerConstructions, 0);
    });
  });

  test(`tampered ${key} fails before a session is returned`, async () => {
    await withFixture({ tampered: key }, async ({ urls }) => {
      await expectFailure(loadLaya({ urls }), new RegExp(`integrity check failed for .*/${names[key]}`), `tampered ${key}`);
      if (key === "tokenizer") assert.equal(globalThis.__layaTokenizerConstructions, 0);
    });
  });
}

test("matching fixture digests load all artifacts and support JSON", async () => {
  await withFixture({}, async ({ urls }) => {
    const session = await loadLaya({ urls });
    assert.equal(session.tokenizer.json.fixture, "tokenizer");
    assert.deepEqual(session.config.temperature, [2, 3, 4]);
    assert.equal(globalThis.__layaTokenizerConstructions, 1);
    assert.equal(session.embeddings.length, payloads.embeddings.length / 2);
    console.log(`honest fixture: tokenizer=${session.tokenizer.json.fixture}, calibration=${session.config.temperature.join(",")}, embeddings=${session.embeddings.length}`);
  });
});

test("vendored filenames resolve to upstream digest keys", async () => {
  await withFixture({ local: true }, async ({ urls }) => {
    const session = await loadLaya({ urls });
    assert.equal(session.tokenizer.json.fixture, "tokenizer");
    assert.deepEqual(session.config.temperature, [2, 3, 4]);
  });
  await withFixture({ local: true, tampered: "tokenizer" }, async ({ urls }) => {
    await expectFailure(loadLaya({ urls }), /integrity check failed for .*\/tokenizer\.json/, "tampered vendored tokenizer");
  });
});

test("explicit vendored tokenizer override loads and verifies like measure and quant-smoke", async () => {
  await withFixture({ explicitVendoredTokenizer: true }, async ({ urls }) => {
    assert.equal(urls[names.tokenizer], "./laya/ml-tokenizer.json");
    const session = await loadLaya({ checkpoint: "multilingual", calibration: "source", urls });
    assert.equal(session.tokenizer.json.fixture, "tokenizer");
    assert.equal(globalThis.__layaTokenizerConstructions, 1);
  });
  await withFixture({ explicitVendoredTokenizer: true, tampered: "tokenizer" }, async ({ urls }) => {
    await expectFailure(loadLaya({ checkpoint: "multilingual", calibration: "source", urls }),
      /integrity check failed for .*\/tokenizer\.json/, "tampered explicitly overridden tokenizer");
    assert.equal(globalThis.__layaTokenizerConstructions, 0);
  });
  await withFixture({ explicitVendoredTokenizer: true, missing: "tokenizer" }, async ({ urls }) => {
    await expectFailure(loadLaya({ checkpoint: "multilingual", calibration: "source", urls }),
      /no pinned sha256 for \.\/laya\/ml-tokenizer\.json/, "missing explicitly overridden tokenizer digest");
  });
});

test("s256 graph requires explicit unverified declaration and reports the skip", async () => {
  for (const variant of ["embeds_fp32", "embeds_wfp16"]) {
    await withFixture({ alternateGraph: variant, explicitVendoredTokenizer: true }, async ({ urls }) => {
      const graphUrl = urls[names.main];
      const warnings = [];
      const originalWarn = console.warn;
      console.warn = (message) => warnings.push(message);
      try {
        const session = await loadLaya({ checkpoint: "multilingual", calibration: "source", window: 256,
          urls, unverified: [graphUrl] });
        assert.equal(session.tokenizer.json.fixture, "tokenizer");
        assert.deepEqual(session.unverifiedArtifacts, [graphUrl]);
        assert.ok(warnings.some((message) => message.includes(`Laya loaded unverified artifact: ${graphUrl}`)));
        assert.equal(session.window, 256);
      } finally {
        console.warn = originalWarn;
      }
    });
  }
  await withFixture({ alternateGraph: "embeds_wfp16", explicitVendoredTokenizer: true }, async ({ urls }) => {
    await expectFailure(loadLaya({ checkpoint: "multilingual", window: 256, urls }),
      /integrity check failed for .*\/laya_ml_s512_embeds_wfp16\.tflite/, "undeclared s256 graph with s512 digest");
  });
  await withFixture({ alternateGraph: "embeds_wfp16", explicitVendoredTokenizer: true, missing: "main" }, async ({ urls }) => {
    await expectFailure(loadLaya({ checkpoint: "multilingual", window: 256, urls }),
      /no pinned sha256 for \.\/laya\/laya_ml_s256_embeds_wfp16\.tflite/, "undeclared s256 graph with missing main digest");
  });
});
