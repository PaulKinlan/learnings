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
//                                 a single byte is consumed: it must still be https, and its host
//                                 must be an artifact host. A redirect off the hub fails loudly
//                                 instead of quietly reading whatever answered.

/** Hosts (and their subdomains) that may serve an artifact after redirects. */
export const ARTIFACT_HOST_SUFFIXES = ["huggingface.co", "hf.co"];

export function artifactHostAllowed(hostname) {
  const host = String(hostname ?? "").toLowerCase();
  if (!host) return false;
  return ARTIFACT_HOST_SUFFIXES.some((suffix) => host === suffix || host.endsWith(`.${suffix}`));
}

/** Fetch options shared by every artifact request; `extra` carries the abort signal. */
export function artifactFetchOptions(extra = {}) {
  return { credentials: "omit", referrerPolicy: "no-referrer", ...extra };
}

// Schemes that never leave the origin: a locally built fixture or blob has nothing to redirect
// to, so it is judged by the sha256 pin alone.
const LOCAL_SCHEMES = new Set(["data:", "blob:"]);

/**
 * Refuse a response that did not end up on an allowed https artifact host. `response.url` is the
 * final URL after redirects; a response without one (a stub) or a local data:/blob: URL is left
 * to the caller's sha256 check.
 * @param {unknown} response fetch response
 * @param {string} label artifact label used in the error
 */
export function assertArtifactResponse(response, label = String(response?.url ?? "artifact")) {
  const final = response?.url;
  if (!final) return;
  let url;
  try {
    url = new URL(final);
  } catch {
    throw new Error(`download refused for ${label}: unparseable final URL ${String(final)}`);
  }
  if (LOCAL_SCHEMES.has(url.protocol)) return;
  if (url.protocol !== "https:")
    throw new Error(`download refused for ${label}: redirect downgraded to ${url.protocol}`);
  if (!artifactHostAllowed(url.hostname))
    throw new Error(
      `download refused for ${label}: redirect left the artifact hosts (${url.hostname}); ` +
        `add it to ARTIFACT_HOST_SUFFIXES in site/decision-models/fetch-policy.mjs if it is legitimate`,
    );
}
