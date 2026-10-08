// scripts/fetch-kev-fixtures.mjs
// Downloads and verifies pinned Kev validation artifacts to /tmp/kev-files/
import { mkdirSync, existsSync, readFileSync, writeFileSync } from "node:fs";
import { resolve } from "node:path";
import { createHash } from "node:crypto";
import { KEV_REVISION, KEV_SHA256 } from "../site/decision-models/kev-manifest.js";

const TARGET_DIR = "/tmp/kev-files";
const BASE_URL = `https://huggingface.co/onnx-community/kev-0.6b-ONNX/raw/${KEV_REVISION}`;
const LFS_URL = `https://huggingface.co/onnx-community/kev-0.6b-ONNX/resolve/${KEV_REVISION}`;

const ARTIFACTS = [
  { file: "config.json", url: `${BASE_URL}/config.json`, sha: KEV_SHA256["config.json"] },
  { file: "tokenizer.json", url: `${BASE_URL}/tokenizer.json`, sha: KEV_SHA256["tokenizer.json"] },
  { file: "onnx/model_q4.onnx", url: `${LFS_URL}/onnx/model_q4.onnx`, sha: KEV_SHA256["onnx/model_q4.onnx"] },
  { file: "onnx/model_q4.onnx_data", url: `${LFS_URL}/onnx/model_q4.onnx_data`, sha: KEV_SHA256["onnx/model_q4.onnx_data"] },
  {
    file: "onnx/model_q4f16.onnx",
    url: `${LFS_URL}/onnx/model_q4f16.onnx`,
    sha: "26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8",
  },
];

mkdirSync(resolve(TARGET_DIR, "onnx"), { recursive: true });

function verifyHash(buffer, expectedSha) {
  const hash = createHash("sha256").update(buffer).digest("hex");
  return hash === expectedSha;
}

for (const art of ARTIFACTS) {
  const dest = resolve(TARGET_DIR, art.file);
  if (existsSync(dest)) {
    const existingBuf = readFileSync(dest);
    if (verifyHash(existingBuf, art.sha)) {
      console.log(`[ok] ${art.file} already verified at ${dest}`);
      continue;
    }
    console.log(`[stale] ${art.file} hash mismatch at ${dest}, re-downloading...`);
  }
  console.log(`[fetch] downloading ${art.file} from ${art.url}...`);
  const res = await fetch(art.url);
  if (!res.ok) throw new Error(`Failed to download ${art.file}: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  if (!verifyHash(buf, art.sha)) {
    throw new Error(`Integrity mismatch for ${art.file}`);
  }
  writeFileSync(dest, buf);
  console.log(`[ok] ${art.file} verified and saved.`);
}

console.log("All Kev validation fixtures ready.");
