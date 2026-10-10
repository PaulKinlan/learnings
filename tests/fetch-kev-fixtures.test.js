// tests/fetch-kev-fixtures.test.js
// Focused tests for fetch-kev-fixtures hardening:
// - Symlinks planted at destination or directories are never followed
// - Temp directory is exclusive (mode 0700) and unpredictable
// - Oversize bodies are rejected before buffering (Content-Length and stream chunks)
// - SHA-256 integrity check and safe destination writing
import test from "node:test";
import assert from "node:assert/strict";
import {
  mkdtempSync,
  rmSync,
  writeFileSync,
  readFileSync,
  symlinkSync,
  existsSync,
  statSync,
  readdirSync,
} from "node:fs";
import { join } from "node:path";
import { tmpdir } from "node:os";
import { createHash } from "node:crypto";
import {
  downloadArtifact,
  fetchKevFixtures,
  hashFile,
  assertNotSymlink,
  ensureSafeDir,
} from "../scripts/fetch-kev-fixtures.mjs";

function makeReadableStream(chunks, onCancel = null) {
  let index = 0;
  return new ReadableStream({
    pull(controller) {
      if (index < chunks.length) {
        controller.enqueue(chunks[index++]);
      } else {
        controller.close();
      }
    },
    cancel(reason) {
      onCancel?.(reason);
    },
  });
}

test("symlink planted at destination is not followed and sentinel file is untouched", async () => {
  const targetDir = mkdtempSync(join(tmpdir(), "kev-test-target-"));
  const tempDir = mkdtempSync(join(tmpdir(), "kev-test-temp-"));
  const sentinelDir = mkdtempSync(join(tmpdir(), "kev-test-sentinel-"));
  const sentinelFile = join(sentinelDir, "victim.txt");
  const secretContent = "HIGHLY_CONFIDENTIAL_DO_NOT_OVERWRITE";
  writeFileSync(sentinelFile, secretContent);

  const destFile = join(targetDir, "config.json");
  symlinkSync(sentinelFile, destFile);

  const dummyBytes = new Uint8Array([1, 2, 3, 4]);
  const sha = createHash("sha256").update(dummyBytes).digest("hex");
  const art = {
    file: "config.json",
    url: "https://mock.invalid/config.json",
    sha,
    maxBytes: 1024,
  };

  const mockFetch = async () => new Response(dummyBytes);

  try {
    await assert.rejects(
      downloadArtifact(art, targetDir, tempDir, mockFetch),
      /Refusing to follow symlink at destination/,
      "must refuse to follow symlink at destination",
    );

    // Verify sentinel was not modified or overwritten
    assert.equal(
      readFileSync(sentinelFile, "utf8"),
      secretContent,
      "sentinel file content must remain completely untouched",
    );
  } finally {
    rmSync(targetDir, { recursive: true, force: true });
    rmSync(tempDir, { recursive: true, force: true });
    rmSync(sentinelDir, { recursive: true, force: true });
  }
});

test("symlink planted at destination directory is not followed", async () => {
  const targetDir = mkdtempSync(join(tmpdir(), "kev-test-target-"));
  const tempDir = mkdtempSync(join(tmpdir(), "kev-test-temp-"));
  const sentinelDir = mkdtempSync(join(tmpdir(), "kev-test-sentinel-"));

  // Plant a symlink for onnx/ pointing to sentinelDir
  symlinkSync(sentinelDir, join(targetDir, "onnx"));

  const dummyBytes = new Uint8Array([1, 2, 3, 4]);
  const sha = createHash("sha256").update(dummyBytes).digest("hex");
  const art = {
    file: "onnx/model_q4.onnx",
    url: "https://mock.invalid/model_q4.onnx",
    sha,
    maxBytes: 1024,
  };

  const mockFetch = async () => new Response(dummyBytes);

  try {
    await assert.rejects(
      downloadArtifact(art, targetDir, tempDir, mockFetch),
      /Refusing to follow symlink at/,
      "must refuse to follow directory symlink",
    );

    assert.equal(
      readdirSync(sentinelDir).length,
      0,
      "sentinel directory must have no files written to it",
    );
  } finally {
    rmSync(targetDir, { recursive: true, force: true });
    rmSync(tempDir, { recursive: true, force: true });
    rmSync(sentinelDir, { recursive: true, force: true });
  }
});

test("fetchKevFixtures creates private temp directory with mode 0700", async () => {
  const targetDir = mkdtempSync(join(tmpdir(), "kev-test-target-"));
  let capturedTempDir = null;

  const dummyBytes = new Uint8Array([10, 20, 30]);
  const sha = createHash("sha256").update(dummyBytes).digest("hex");
  const art = {
    file: "config.json",
    url: "https://mock.invalid/config.json",
    sha,
    maxBytes: 1024,
  };

  const mockFetch = async () => {
    // Inspect temporary directories in os.tmpdir() created by kev-fetch-
    const entries = readdirSync(tmpdir()).filter((e) => e.startsWith("kev-fetch-"));
    for (const entry of entries) {
      const p = join(tmpdir(), entry);
      const st = statSync(p);
      const mode = st.mode & 0o777;
      if (mode === 0o700) {
        capturedTempDir = p;
      }
    }
    return new Response(dummyBytes);
  };

  try {
    await fetchKevFixtures({
      targetDir,
      artifacts: [art],
      fetchFn: mockFetch,
    });

    assert.ok(capturedTempDir, "found kev-fetch- temporary directory with private 0700 permissions");
    // Verify cleanup: temp dir must be deleted after fetchKevFixtures finishes
    assert.equal(
      existsSync(capturedTempDir),
      false,
      "temporary directory must be cleaned up on completion",
    );
  } finally {
    rmSync(targetDir, { recursive: true, force: true });
  }
});

test("Content-Length header exceeding maxBytes aborts download before reading stream", async () => {
  const targetDir = mkdtempSync(join(tmpdir(), "kev-test-target-"));
  const tempDir = mkdtempSync(join(tmpdir(), "kev-test-temp-"));

  let bodyAccessed = false;
  const mockResponse = {
    ok: true,
    status: 200,
    headers: new Headers({ "content-length": "20000" }),
    get body() {
      bodyAccessed = true;
      return makeReadableStream([new Uint8Array(20000)]);
    },
  };

  const art = {
    file: "config.json",
    url: "https://mock.invalid/config.json",
    sha: "0".repeat(64),
    maxBytes: 1024,
  };

  try {
    await assert.rejects(
      downloadArtifact(art, targetDir, tempDir, async () => mockResponse),
      /Content-Length 20000 exceeds maximum allowed size 1024 bytes/,
    );
    assert.equal(bodyAccessed, false, "stream body was never accessed because Content-Length failed fast");
    assert.equal(
      existsSync(join(targetDir, "config.json")),
      false,
      "destination file was not created",
    );
    assert.equal(readdirSync(tempDir).length, 0, "temp directory has no leftover files");
  } finally {
    rmSync(targetDir, { recursive: true, force: true });
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("streamed chunks exceeding maxBytes abort streaming before full body is buffered", async () => {
  const targetDir = mkdtempSync(join(tmpdir(), "kev-test-target-"));
  const tempDir = mkdtempSync(join(tmpdir(), "kev-test-temp-"));

  let cancelled = false;
  const chunk1 = new Uint8Array(600);
  const chunk2 = new Uint8Array(600); // Exceeds 1000 limit
  const chunk3 = new Uint8Array(600);

  const stream = makeReadableStream([chunk1, chunk2, chunk3], () => {
    cancelled = true;
  });

  const mockResponse = {
    ok: true,
    status: 200,
    headers: new Headers(), // No Content-Length to force stream chunk enforcement
    body: stream,
  };

  const art = {
    file: "config.json",
    url: "https://mock.invalid/config.json",
    sha: "0".repeat(64),
    maxBytes: 1000,
  };

  try {
    await assert.rejects(
      downloadArtifact(art, targetDir, tempDir, async () => mockResponse),
      /stream exceeded maximum allowed size 1000 bytes \(received 1200 bytes\)/,
    );
    assert.equal(cancelled, true, "ReadableStream was cancelled via AbortController");
    assert.equal(
      existsSync(join(targetDir, "config.json")),
      false,
      "destination file was not created",
    );
    assert.equal(readdirSync(tempDir).length, 0, "temp directory has no leftover part files");
  } finally {
    rmSync(targetDir, { recursive: true, force: true });
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("SHA-256 integrity mismatch throws and leaves no destination or temp file", async () => {
  const targetDir = mkdtempSync(join(tmpdir(), "kev-test-target-"));
  const tempDir = mkdtempSync(join(tmpdir(), "kev-test-temp-"));

  const payload = new Uint8Array([1, 2, 3, 4, 5]);
  const wrongSha = "f".repeat(64);
  const art = {
    file: "config.json",
    url: "https://mock.invalid/config.json",
    sha: wrongSha,
    maxBytes: 1024,
  };

  try {
    await assert.rejects(
      downloadArtifact(art, targetDir, tempDir, async () => new Response(payload)),
      /Integrity mismatch for config.json/,
    );
    assert.equal(
      existsSync(join(targetDir, "config.json")),
      false,
      "corrupted file was never written to destination",
    );
    assert.equal(readdirSync(tempDir).length, 0, "temporary part file was removed on error");
  } finally {
    rmSync(targetDir, { recursive: true, force: true });
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("legitimate download streams to exclusive temp file, verifies sha, and saves to destination", async () => {
  const targetDir = mkdtempSync(join(tmpdir(), "kev-test-target-"));
  const tempDir = mkdtempSync(join(tmpdir(), "kev-test-temp-"));

  const payload = new Uint8Array([65, 66, 67, 68, 69]); // "ABCDE"
  const expectedSha = createHash("sha256").update(payload).digest("hex");
  const art = {
    file: "onnx/model_q4.onnx",
    url: "https://mock.invalid/model_q4.onnx",
    sha: expectedSha,
    maxBytes: 1024,
  };

  try {
    await downloadArtifact(art, targetDir, tempDir, async () => new Response(payload));

    const dest = join(targetDir, "onnx/model_q4.onnx");
    assert.ok(existsSync(dest), "destination file was created");
    assert.deepEqual(readFileSync(dest), Buffer.from(payload), "destination content matches");
    assert.equal(readdirSync(tempDir).length, 0, "temporary part file was cleaned up");

    // Re-running with existing verified file skips re-download
    let fetchCalled = false;
    await downloadArtifact(art, targetDir, tempDir, async () => {
      fetchCalled = true;
      return new Response(payload);
    });
    assert.equal(fetchCalled, false, "verified existing file was not re-downloaded");
  } finally {
    rmSync(targetDir, { recursive: true, force: true });
    rmSync(tempDir, { recursive: true, force: true });
  }
});

test("stale file with hash mismatch is detected and safely re-downloaded", async () => {
  const targetDir = mkdtempSync(join(tmpdir(), "kev-test-target-"));
  const tempDir = mkdtempSync(join(tmpdir(), "kev-test-temp-"));

  const dest = join(targetDir, "config.json");
  writeFileSync(dest, "STALE_CONTENT");

  const newPayload = new Uint8Array([11, 22, 33, 44]);
  const expectedSha = createHash("sha256").update(newPayload).digest("hex");
  const art = {
    file: "config.json",
    url: "https://mock.invalid/config.json",
    sha: expectedSha,
    maxBytes: 1024,
  };

  let downloadCount = 0;
  const mockFetch = async () => {
    downloadCount++;
    return new Response(newPayload);
  };

  try {
    await downloadArtifact(art, targetDir, tempDir, mockFetch);
    assert.equal(downloadCount, 1, "fetch was invoked to replace stale file");
    assert.deepEqual(readFileSync(dest), Buffer.from(newPayload), "destination content replaced with verified bytes");
  } finally {
    rmSync(targetDir, { recursive: true, force: true });
    rmSync(tempDir, { recursive: true, force: true });
  }
});
