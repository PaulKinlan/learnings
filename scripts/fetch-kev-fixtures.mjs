// scripts/fetch-kev-fixtures.mjs
// Downloads and verifies pinned Kev validation artifacts to /tmp/kev-files/
import { mkdirSync, existsSync, writeFileSync } from "node:fs";
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
];

mkdirSync(resolve(TARGET_DIR, "onnx"), { recursive: true });

for (const art of ARTIFACTS) {
  const dest = resolve(TARGET_DIR, art.file);
  if (existsSync(dest)) {
    console.log(`[skip] ${art.file} already exists at ${dest}`);
    continue;
  }
  console.log(`[fetch] downloading ${art.file} from ${art.url}...`);
  const res = await fetch(art.url);
  if (!res.ok) throw new Error(`Failed to download ${art.file}: HTTP ${res.status}`);
  const buf = Buffer.from(await res.arrayBuffer());
  const hash = createHash("sha256").update(buf).digest("hex");
  if (hash !== art.sha) {
    throw new Error(`Integrity mismatch for ${art.file}: expected ${art.sha}, got ${hash}`);
  }
  writeFileSync(dest, buf);
  console.log(`[ok] ${art.file} verified and saved.`);
}

console.log("All Kev validation fixtures ready.");
