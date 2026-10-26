// Request policy for model-artifact downloads (learnings-902).
//
// The manifests pin a sha256 for every artifact, so a substituted or corrupted weight file fails
// verification before it is used. The pin says nothing about what the REQUEST reveals, and these
// downloads are cross-origin and unauthenticated:
//   credentials: 'omit'        — no ambient cookies or HTTP auth are sent to the CDN at all.
//   referrerPolicy: 'no-referrer' — the reader's page URL never leaves the origin (the same
//                                 hardening site/decision-models/core.js applies to provider
//                                 calls, and what the CSP allows for this origin).
//   redirect handling          — a hop-by-hop allowlist is not possible from a page: a
//                                 cross-origin `redirect: 'manual'` returns an opaque-redirect
//                                 response whose Location header is unreadable, and
//                                 `redirect: 'error'` (right for provider APIs) would break the
//                                 hub's legitimate redirect to its CDN. The policy therefore
//                                 follows redirects and validates the FINAL response URL before
//                                 a single byte is consumed. Validation is source-aware: an
//                                 artifact the site vendors first-party (Laya's tokenizer, act
//                                 head, calibration and SHA256SUMS under decision-models/laya/)
//                                 is served from this origin and is accepted as long as it stayed
//                                 same-origin; anything cross-origin must be https on an artifact
//                                 host. A cross-origin redirect off the hub fails loudly instead of
//                                 quietly reading whatever answered (learnings-0px).

/** Hosts (and their subdomains) that may serve an artifact after redirects. */
export const ARTIFACT_HOST_SUFFIXES = ["huggingface.co", "hf.co"];

/**
 * Idle bound for one artifact download (tm-download-no-timeout), shared by both engines. The
 * timer resets on every sign of progress (response headers, each body chunk), so it bounds a
 * STALL — a hub that accepts the connection and then goes quiet — not the total transfer time:
 * a slow link that keeps sending bytes is never killed, however large the artifact. A stalled
 * hub fails the download after this many milliseconds of silence instead of hanging the
 * decision lab UI forever.
 */
export const ARTIFACT_DOWNLOAD_TIMEOUT_MS = 300_000;

export function artifactHostAllowed(hostname) {
  const host = String(hostname ?? "").toLowerCase();
  if (!host) return false;
  return ARTIFACT_HOST_SUFFIXES.some((suffix) => host === suffix || host.endsWith(`.${suffix}`));
}

/** Fetch options shared by every artifact request; `extra` carries the abort signal. */
export function artifactFetchOptions(extra = {}) {
  return { credentials: "omit", referrerPolicy: "no-referrer", ...extra };
}

/**
 * Prototype-safe own-property lookup for a manifest size cap. The frozen MAX_BYTES maps inherit
 * Object.prototype, so a plain `map[filename]` resolves `constructor`, `__proto__`, `toString` and
 * friends to Object members — non-number values that make `total > limit` / `received > limit`
 * compare NaN (always false) and silently disable the bounded fallback. Only an own key whose
 * value is a finite positive number is trusted; anything else falls through the cap chain to the
 * engine's default cap, so every artifact gets a bounded cap (tm-unbounded-download-buffer).
 */
export function ownCap(caps, key) {
  const value = Object.hasOwn(caps ?? {}, key) ? caps[key] : undefined;
  return typeof value === "number" && Number.isFinite(value) && value > 0 ? value : undefined;
}

/**
 * Resolve an explicit maxBytes override at the helper's API boundary. Unlike a manifest lookup
 * (ownCap, which falls through to the default cap), an explicit override is a caller-supplied
 * value, so an invalid one is a bug that must fail loudly instead of silently unbinding the
 * download: NaN, ±Infinity, 0, negatives and non-numeric strings would otherwise make
 * `total > limit` / `received > limit` compare false (or always true) and either accept an
 * oversized body or reject every byte. A missing override (null/undefined) returns undefined so
 * the cap chain continues into the manifest lookup and the default (tm-unbounded-download-buffer).
 */
export function explicitCap(value, label) {
  if (value == null) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(
      `invalid maxBytes for ${label}: expected a finite positive number of bytes, got ${String(value)}`,
    );
  }
  return value;
}

/**
 * Validate an explicit timeoutMs at the helper's API boundary, the same policy as explicitCap:
 * a caller-supplied deadline that is NaN, 0, negative or non-numeric reaches setTimeout, where
 * NaN/0 fire immediately (node clamps to 1ms) and abort EVERY download at once — a silent total
 * outage of the download path instead of a loud bug. Missing (null/undefined) keeps the engine's
 * documented default.
 */
export function validatedTimeoutMs(value, label) {
  if (value == null) return undefined;
  if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
    throw new Error(
      `invalid timeoutMs for ${label}: expected a finite positive number of milliseconds, got ${String(value)}`,
    );
  }
  return value;
}

/** Required-sha policy: the digest is checked, never skipped, whoever the caller is. */
export async function verifyArtifactSha256(bytes, sha256, label) {
  if (!sha256) throw new Error(`no pinned sha256 for ${label}`);
  const digest = await crypto.subtle.digest("SHA-256", bytes);
  const hex = [...new Uint8Array(digest)].map((b) => b.toString(16).padStart(2, "0")).join("");
  if (hex !== sha256.toLowerCase()) {
    throw new Error(
      `integrity check failed for ${label}: sha256 ${hex.slice(0, 16)}… does not match the pinned ${sha256.slice(0, 16)}…`,
    );
  }
}

/**
 * Fetch one artifact URL into a Uint8Array through the artifact request policy, bounded in both
 * dimensions (tm-unbounded-download-buffer, tm-download-no-timeout):
 *   size  — Content-Length is refused before the body is read, and the running total is refused
 *           during streaming, so no path buffers an unbounded body.
 *   time  — an idle deadline is armed before the request and RESET on every progress event
 *           (response headers, each body chunk). It bounds a stalled hub, not the transfer time:
 *           a download that keeps moving — however slowly — is never killed, and one that goes
 *           quiet fails with a TimeoutError naming how many bytes had arrived.
 * One implementation, used by kev-engine.js, laya-engine.js and the SHA256SUMS fetch, so the
 * bounds cannot drift apart again (learnings-0ef).
 *
 * @param caps manifest map consulted (via ownCap) by filename, label basename first
 * @param defaultMaxBytes final fallback cap when nothing else matches
 * @param failurePrefix error prefix; defaults to `download failed for ${label}` (the sums fetch
 *        passes its own so the message names SHA256SUMS, not a fake artifact download)
 */
export async function boundedArtifactRead(url, {
  onProgress = null,
  label = url,
  maxBytes = null,
  timeoutMs,
  caps = null,
  defaultMaxBytes = null,
  failurePrefix = null,
} = {}) {
  const prefix = failurePrefix ?? `download failed for ${label}`;
  const timeout = validatedTimeoutMs(timeoutMs, label) ?? ARTIFACT_DOWNLOAD_TIMEOUT_MS;
  const basename = String(url).split("/").pop()?.split("?")[0] ?? "";
  const labelBasename = String(label).split("/").pop()?.split("?")[0] ?? "";
  const limit =
    explicitCap(maxBytes, label) ??
    ownCap(caps, labelBasename) ??
    ownCap(caps, basename) ??
    defaultMaxBytes;

  const controller = new AbortController();
  let received = 0;
  let deadline = null;
  // Re-armed on every progress event; the abort message reads `received` at fire time, so a
  // download that dies mid-stream says how much had arrived instead of claiming "no data".
  const armDeadline = () => {
    clearTimeout(deadline);
    deadline = setTimeout(() => {
      const detail = received === 0
        ? `no data within ${timeout}ms`
        : `stalled: nothing further within ${timeout}ms after ${received} bytes`;
      controller.abort(new DOMException(`${prefix}: ${detail}`, "TimeoutError"));
    }, timeout);
  };
  armDeadline();
  try {
    const res = await fetch(url, artifactFetchOptions({ signal: controller.signal }));
    armDeadline(); // headers arrived: that is progress too
    assertArtifactResponse(res, label, url);
    if (!res.ok) throw new Error(`${prefix}: HTTP ${res.status}`);
    const total = Number(res.headers.get("content-length")) || 0;
    if (limit != null && total > limit) {
      controller.abort();
      throw new Error(
        `${prefix}: Content-Length ${total} exceeds maximum allowed size ${limit} bytes`,
      );
    }
    const reader = res.body.getReader();
    const chunks = [];
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      received += value.length;
      armDeadline(); // bytes are still flowing: the stall bound restarts
      if (limit != null && received > limit) {
        controller.abort();
        try { await reader.cancel(); } catch {}
        throw new Error(
          `${prefix}: stream exceeded maximum allowed size ${limit} bytes (received ${received} bytes)`,
        );
      }
      chunks.push(value);
      onProgress?.(received, total);
    }
    return chunks.length === 1 ? chunks[0] : (() => {
      const b = new Uint8Array(received);
      let offset = 0;
      for (const chunk of chunks) {
        b.set(chunk, offset);
        offset += chunk.length;
      }
      return b;
    })();
  } finally {
    clearTimeout(deadline);
  }
}

/**
 * boundedArtifactRead plus the required sha256 check. There is deliberately no path that skips
 * the check: the trust decision never depends on the caller, so a substituted or corrupted
 * artifact fails here rather than executing. The digest is required — an absent one throws
 * before the network is touched.
 */
export async function fetchArtifactBytes(url, { sha256, label = url, ...rest } = {}) {
  if (!sha256) throw new Error(`no pinned sha256 for ${label}`);
  const bytes = await boundedArtifactRead(url, { sha256, label, ...rest });
  await verifyArtifactSha256(bytes, sha256, label);
  return bytes;
}

// Schemes that never leave the origin: a locally built fixture or blob has nothing to redirect
// to, so it is judged by the sha256 pin alone.
const LOCAL_SCHEMES = new Set(["data:", "blob:"]);

const parseUrl = (value) => {
  try {
    return new URL(String(value));
  } catch {
    return null;
  }
};

/**
 * Refuse a response that did not end up where this download is allowed to come from.
 *
 * `response.url` is the final URL after redirects. It is accepted when it is still on the
 * request's own origin (a first-party vendored artifact, and a same-origin hop), or when it is
 * https on an artifact host (the hub and its CDN). A response without a URL (a stub) or a local
 * data:/blob: URL is left to the caller's sha256 check.
 *
 * @param {unknown} response fetch response
 * @param {string} label artifact label used in the error
 * @param {string} requestedUrl the URL the caller asked for, so first-party sources are known
 */
export function assertArtifactResponse(response, label = String(response?.url ?? "artifact"), requestedUrl = null) {
  const final = response?.url;
  if (!final) return;
  const url = parseUrl(final);
  if (!url) throw new Error(`download refused for ${label}: unparseable final URL ${String(final)}`);
  if (LOCAL_SCHEMES.has(url.protocol)) return;
  // First-party: the response stayed on the request's own origin (an origin includes the scheme,
  // so a downgrade cannot hide here). This is the vendored path — Laya's tokenizer, act head,
  // calibration and SHA256SUMS ship with the site and are fetched from it.
  const requested = requestedUrl ? parseUrl(requestedUrl) : null;
  if (requested && requested.origin === url.origin) return;
  if (url.protocol !== "https:")
    throw new Error(`download refused for ${label}: redirect downgraded to ${url.protocol}`);
  if (!artifactHostAllowed(url.hostname))
    throw new Error(
      `download refused for ${label}: redirect left the artifact hosts (${url.hostname}); ` +
        `add it to ARTIFACT_HOST_SUFFIXES in site/decision-models/fetch-policy.mjs if it is legitimate`,
    );
}
