// tests/laya-engine.test.js — the Laya weight provenance. The engine must fetch its graphs, its
// token table and the SHA256SUMS it checks them against from an immutable revision, and the
// digests it checks against must be the ones that revision published. No model, no browser, no
// network: everything here is derived from the checked-in manifest and the vendored sums file.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { LAYA_REVISION, LAYA_SHA256SUMS_SHA256 } from "../site/decision-models/laya-manifest.js";
import { layaArtifactUrl } from "../site/decision-models/laya-engine.js";

const REPOS = {
  multilingual: "litert-community/Laya-Multilingual-LiteRT",
  english: "litert-community/laya-LiteRT",
};

// Every path loadLaya can resolve per checkpoint: the artifacts it fetches plus the sums file it
// verifies them against. The sums file is in the list because a hash list fetched from a mutable
// ref is as replaceable as the bytes it describes.
const FETCHED = {
  multilingual: [
    "laya_ml_s512_embeds_wfp16.tflite",
    "laya_ml_act_head_fp32.tflite",
    "token_embeddings_fp16.bin",
    "tokenizer.json",
    "laya_ml_calibration.json",
    "SHA256SUMS",
  ],
  english: ["laya_en_s512_wfp16.tflite", "laya_act_head_fp32.tflite", "tokenizer.json", "SHA256SUMS"],
};

test("every artifact and the sums file are pinned to an immutable revision, not a branch", () => {
  for (const [checkpoint, files] of Object.entries(FETCHED)) {
    assert.match(LAYA_REVISION[checkpoint], /^[0-9a-f]{40}$/, `${checkpoint}: the pin is a commit sha`);
    for (const file of files) {
      const url = layaArtifactUrl(checkpoint, file);
      assert.equal(
        url,
        `https://huggingface.co/${REPOS[checkpoint]}/resolve/${LAYA_REVISION[checkpoint]}/${file}`,
      );
      assert.match(url, /\/resolve\/[0-9a-f]{40}\//, `${file} resolves through a commit sha`);
      assert.ok(!url.includes("/main/"), `${file} has no path through the mutable main branch`);
    }
  }
  assert.throws(() => layaArtifactUrl("nope", "tokenizer.json"), /unknown checkpoint 'nope'/);
});

test("the vendored multilingual sums are the pinned revision's own file, and cover what it fetches", () => {
  const bytes = readFileSync(new URL("../site/decision-models/laya/ml-SHA256SUMS", import.meta.url));
  const text = bytes.toString("utf8");
  assert.equal(
    createHash("sha256").update(bytes).digest("hex"),
    LAYA_SHA256SUMS_SHA256.multilingual,
    "laya/ml-SHA256SUMS is byte-identical to SHA256SUMS at the pinned revision: re-pin both together",
  );
  for (const file of FETCHED.multilingual) {
    if (file === "SHA256SUMS") continue;
    const escaped = file.replaceAll(".", "\\.");
    assert.match(
      text,
      new RegExp(`^[0-9a-f]{64} {1,2}\\*?${escaped}$`, "m"),
      `${file} has a recorded digest, so its fetch is actually checked`,
    );
  }
  for (const [local, upstream] of [
    ["ml-tokenizer.json", "tokenizer.json"],
    ["laya_ml_act_head_fp32.tflite", "laya_ml_act_head_fp32.tflite"],
    ["laya_ml_calibration.json", "laya_ml_calibration.json"],
  ]) {
    const digest = createHash("sha256")
      .update(readFileSync(new URL(`../site/decision-models/laya/${local}`, import.meta.url)))
      .digest("hex");
    assert.ok(text.includes(`${digest}  ${upstream}\n`), `${local} matches the pinned ${upstream} digest`);
  }
});
