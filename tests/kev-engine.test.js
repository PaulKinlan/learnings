// The Kev weight identity. The engine must run only bytes fetched from an immutable revision
// and matching the checked-in digests; these checks run without a model or a browser.
import test from "node:test";
import assert from "node:assert/strict";
import { createHash } from "node:crypto";
import { KEV_REVISION, KEV_SHA256 } from "../site/decision-models/kev-manifest.js";
import { fetchVerified, kevArtifactUrl } from "../site/decision-models/kev-engine.js";

const ARTIFACTS = ["tokenizer.json", "config.json", "onnx/model_q4.onnx", "onnx/model_q4.onnx_data"];

test("every artifact is pinned to an immutable revision, not a branch", () => {
  assert.match(KEV_REVISION, /^[0-9a-f]{40}$/, "the revision is a commit sha");
  assert.deepEqual(
    Object.keys(KEV_SHA256).sort(),
    [...ARTIFACTS].sort(),
    "the manifest covers exactly the four artifacts the engine fetches",
  );
  for (const file of ARTIFACTS) {
    assert.match(KEV_SHA256[file], /^[0-9a-f]{64}$/, `${file} has a sha256`);
    const url = kevArtifactUrl(file);
    assert.equal(url, `https://huggingface.co/onnx-community/kev-0.6b-ONNX/resolve/${KEV_REVISION}/${file}`);
    assert.ok(!url.includes("/resolve/main/"), "no URL resolves through the mutable main branch");
  }
});

test("a digest mismatch aborts before the bytes are returned", async () => {
  const bytes = Buffer.from("not the pinned kev weights");
  const url = `data:application/octet-stream;base64,${bytes.toString("base64")}`;
  const good = createHash("sha256").update(bytes).digest("hex");

  const verified = await fetchVerified(url, { sha256: good, label: "fixture" });
  assert.deepEqual(Buffer.from(verified), bytes, "matching bytes come back");

  await assert.rejects(
    fetchVerified(url, { sha256: "0".repeat(64), label: "fixture" }),
    /integrity check failed for fixture/,
  );
  await assert.rejects(fetchVerified(url, { label: "fixture" }), /no pinned sha256 for fixture/);
});
