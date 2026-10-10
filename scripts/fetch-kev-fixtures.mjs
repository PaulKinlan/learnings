// scripts/fetch-kev-fixtures.mjs
// Downloads and verifies pinned Kev validation artifacts to /tmp/kev-files/
import {
  mkdirSync,
  mkdtempSync,
  lstatSync,
  openSync,
  closeSync,
  rmSync,
  constants,
  createReadStream,
  createWriteStream,
  realpathSync,
} from "node:fs";
import { resolve, dirname, join } from "node:path";
import { tmpdir } from "node:os";
import { randomUUID, createHash } from "node:crypto";
import { pipeline } from "node:stream/promises";
import { once } from "node:events";
import { fileURLToPath } from "node:url";
import {
  KEV_REVISION,
  KEV_SHA256,
  KEV_MAX_BYTES,
  KEV_DEFAULT_MAX_BYTES,
} from "../site/decision-models/kev-manifest.js";

export const TARGET_DIR = process.env.KEV_FILES_DIR || "/tmp/kev-files";
export const BASE_URL = `https://huggingface.co/onnx-community/kev-0.6b-ONNX/raw/${KEV_REVISION}`;
export const LFS_URL = `https://huggingface.co/onnx-community/kev-0.6b-ONNX/resolve/${KEV_REVISION}`;

export const ARTIFACTS = [
  { file: "config.json", url: `${BASE_URL}/config.json`, sha: KEV_SHA256["config.json"], maxBytes: KEV_MAX_BYTES["config.json"] },
  { file: "tokenizer.json", url: `${BASE_URL}/tokenizer.json`, sha: KEV_SHA256["tokenizer.json"], maxBytes: KEV_MAX_BYTES["tokenizer.json"] },
  { file: "onnx/model_q4.onnx", url: `${LFS_URL}/onnx/model_q4.onnx`, sha: KEV_SHA256["onnx/model_q4.onnx"], maxBytes: KEV_MAX_BYTES["onnx/model_q4.onnx"] },
  { file: "onnx/model_q4.onnx_data", url: `${LFS_URL}/onnx/model_q4.onnx_data`, sha: KEV_SHA256["onnx/model_q4.onnx_data"], maxBytes: KEV_MAX_BYTES["onnx/model_q4.onnx_data"] },
  {
    file: "onnx/model_q4f16.onnx",
    url: `${LFS_URL}/onnx/model_q4f16.onnx`,
    sha: "26e059ddd954efd9897772b098db10b9eeda9b873984dc1c73b32a9ce177c7c8",
    maxBytes: 10 * 1024 * 1024,
  },
];

export function assertNotSymlink(path, label = path) {
  const stat = lstatSync(path, { throwIfNoEntry: false });
  if (stat?.isSymbolicLink()) {
    throw new Error(`Refusing to follow symlink at ${label}: ${path}`);
  }
  return stat;
}

export function ensureSafeDir(dirPath, label = dirPath) {
  const stat = assertNotSymlink(dirPath, label);
  if (!stat) {
    mkdirSync(dirPath, { recursive: true });
    assertNotSymlink(dirPath, label);
  } else if (!stat.isDirectory()) {
    throw new Error(`Expected directory at ${label}: ${dirPath}`);
  }
}

export async function hashFile(filePath) {
  assertNotSymlink(filePath, "destination");
  let fd;
  try {
    fd = openSync(filePath, constants.O_RDONLY | constants.O_NOFOLLOW);
  } catch (err) {
    if (err.code === "ELOOP") {
      throw new Error(`Refusing to follow symlink at destination: ${filePath}`);
    }
    throw err;
  }
  const hash = createHash("sha256");
  const stream = createReadStream(null, { fd, autoClose: true });
  for await (const chunk of stream) {
    hash.update(chunk);
  }
  return hash.digest("hex");
}

export async function downloadArtifact(art, targetDir, tempDir, fetchFn = fetch) {
  const dest = resolve(targetDir, art.file);
  const destDir = dirname(dest);
  ensureSafeDir(destDir, "destination directory");

  const stat = assertNotSymlink(dest, "destination");
  if (stat) {
    if (!stat.isFile()) {
      throw new Error(`Destination ${dest} is not a regular file`);
    }
    const existingSha = await hashFile(dest);
    if (existingSha === art.sha.toLowerCase()) {
      console.log(`[ok] ${art.file} already verified at ${dest}`);
      return;
    }
    console.log(`[stale] ${art.file} hash mismatch at ${dest}, re-downloading...`);
  }

  const limit =
    art.maxBytes ??
    KEV_MAX_BYTES[art.file] ??
    (art.file === "onnx/model_q4f16.onnx" ? 10 * 1024 * 1024 : KEV_DEFAULT_MAX_BYTES);

  const tempFile = join(tempDir, `${art.file.replace(/\//g, "_")}-${randomUUID()}.part`);
  let tempFd;
  try {
    tempFd = openSync(
      tempFile,
      constants.O_WRONLY | constants.O_CREAT | constants.O_EXCL | constants.O_NOFOLLOW,
      0o600
    );
  } catch (err) {
    throw new Error(`Failed to create exclusive temp file at ${tempFile}: ${err.message}`);
  }

  console.log(`[fetch] downloading ${art.file} from ${art.url}...`);
  const controller = new AbortController();

  const outStream = createWriteStream(null, { fd: tempFd });
  const streamPromise = new Promise((resolveStream, rejectStream) => {
    outStream.on("error", rejectStream);
    outStream.on("finish", resolveStream);
  });

  try {
    let res;
    try {
      res = await fetchFn(art.url, { signal: controller.signal });
    } catch (err) {
      controller.abort();
      throw err;
    }

    if (!res.ok) {
      throw new Error(`Failed to download ${art.file}: HTTP ${res.status}`);
    }

    const total = Number(res.headers.get("content-length")) || 0;
    if (limit != null && total > limit) {
      controller.abort();
      throw new Error(
        `Failed to download ${art.file}: Content-Length ${total} exceeds maximum allowed size ${limit} bytes`
      );
    }

    const hasher = createHash("sha256");
    let received = 0;

    if (res.body) {
      const reader = res.body.getReader();
      try {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          received += value.byteLength || value.length;
          if (limit != null && received > limit) {
            controller.abort();
            try { await reader.cancel(); } catch {}
            throw new Error(
              `Failed to download ${art.file}: stream exceeded maximum allowed size ${limit} bytes (received ${received} bytes)`
            );
          }
          hasher.update(value);
          if (!outStream.write(value)) {
            await once(outStream, "drain");
          }
        }
      } finally {
        reader.releaseLock();
      }
    }

    outStream.end();
    await streamPromise;

    const actualSha = hasher.digest("hex");
    if (actualSha !== art.sha.toLowerCase()) {
      throw new Error(`Integrity mismatch for ${art.file}`);
    }

    assertNotSymlink(dest, "destination");
    let destFd;
    try {
      destFd = openSync(
        dest,
        constants.O_WRONLY | constants.O_CREAT | constants.O_TRUNC | constants.O_NOFOLLOW,
        0o644
      );
    } catch (err) {
      if (err.code === "ELOOP") {
        throw new Error(`Refusing to follow symlink at destination: ${dest}`);
      }
      throw err;
    }

    try {
      const inStream = createReadStream(tempFile);
      const outDestStream = createWriteStream(null, { fd: destFd });
      await pipeline(inStream, outDestStream);
    } finally {
      try { closeSync(destFd); } catch {}
    }

    console.log(`[ok] ${art.file} verified and saved.`);
  } catch (err) {
    try {
      if (!outStream.destroyed) outStream.destroy();
    } catch {}
    throw err;
  } finally {
    try {
      rmSync(tempFile, { force: true });
    } catch {}
  }
}

export async function fetchKevFixtures({
  targetDir = process.env.KEV_FILES_DIR || "/tmp/kev-files",
  artifacts = ARTIFACTS,
  fetchFn = fetch,
} = {}) {
  ensureSafeDir(targetDir, "target directory");
  ensureSafeDir(resolve(targetDir, "onnx"), "onnx directory");

  const tempDir = mkdtempSync(join(tmpdir(), "kev-fetch-"));
  const cleanup = () => {
    try {
      rmSync(tempDir, { recursive: true, force: true });
    } catch {}
  };
  process.on("exit", cleanup);

  try {
    for (const art of artifacts) {
      await downloadArtifact(art, targetDir, tempDir, fetchFn);
    }
    console.log("All Kev validation fixtures ready.");
  } finally {
    process.off("exit", cleanup);
    cleanup();
  }
}

let isMain = false;
if (process.argv[1]) {
  try {
    const scriptPath = fileURLToPath(import.meta.url);
    isMain =
      resolve(process.argv[1]) === scriptPath ||
      realpathSync(process.argv[1]) === realpathSync(scriptPath);
  } catch {}
}

if (isMain) {
  await fetchKevFixtures();
}
