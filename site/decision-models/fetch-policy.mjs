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
