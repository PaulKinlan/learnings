# Presentation delay on neural-network chapter pages (learnings-6nb)

Measured 2026-10-22 on the learnings project VM, headless Chromium (`--disable-gpu`,
software raster) via `tests/lib/cdp.mjs`-style CDP drivers. Trusted input only
(`Input.dispatchMouseEvent`) because synthetic `dispatchEvent` produces no Event Timing
entries. Machine was under fleet load throughout — absolute numbers are environment-dependent;
the decomposition is not.

## How to reproduce

Run from the repo root (each script serves `site/` itself):

- `node research/nn-inp/measure.mjs` — Event Timing + longtask + rAF probe on architectures + gradient-descent
- `node research/nn-inp/control.mjs` — same probe on the hub page (index.html) as a control
- `node research/nn-inp/idle.mjs <page> <ms>` — idle longtask census, no input
- `node research/nn-inp/cpu.mjs` — CDP CPU profiles (idle + click phase)
- `node research/nn-inp/trace2.mjs` — devtools.timeline trace of load + idle window

## Findings

1. **Handlers are cheap — confirmed twice.** Event Timing `processingStart→End` is 0–4ms
   for every click/pointer event on both chapter pages (matches the earlier w6x measurement).

2. **The page-side defect is gradient-descent's on-load race animation.** The devtools
   timeline catches `FunctionCall "tick"` (the `requestAnimationFrame` handler in
   `gradient-descent.js animate()`) at **704ms for the first frame and 102ms for the next**:
   every frame clears and fully redraws two canvases (6 optimiser paths × 2 strokes on the
   map + chart axes/paths). Under software raster each frame saturates the main thread, so
   the 900ms animation stretches over several seconds of ~100ms-cadence long tasks. Any
   input during that window queues behind it — that is the 1.3–10.4s hover duration the
   bead was filed with. Fix direction: draw incrementally (only the newest path segment per
   frame) instead of clear-everything-every-frame.

3. **Chapter documents are expensive to lay out once at load.** Trace on
   gradient-descent.html: `UpdateLayoutTree` 495ms + full-document `Layout` 497ms
   (1350 dirty objects) + `Paint` 92ms ≈ 1.1s of main thread before first paint settles —
   the MathML-heavy content (38 `<math>`). This is load cost, not input cost;
   `content-visibility: auto` on chapter sections is the candidate fix.

4. **The residual per-input presentation delay is environmental, not page code.**
   Control experiment: the hub page (`index.html` — light DOM, no canvas, no heavy JS)
   shows the same presentTail per trusted click (~88–104ms) as the chapter pages. CPU
   profiles during clicks show ~0 JS (5995/6000ms idle in 6s; the rest is native
   `(program)`). ~90–100ms is what one damaged-frame software raster costs on this loaded
   VM. On real hardware with GPU raster this is single-digit ms — do not gate quality
   decisions on these absolute numbers.

5. **Cleared hypotheses:** canvas backing stores are at 0.73–1.14× CSS size (fine); no
   continuous CSS invalidation exists (only one 0.3s one-shot keyframe, on architectures);
   no idle animations (gradient-descent's `animate()` stops at p=1; the "idle" bursts in
   early runs were finding 2's animation delayed by VM scheduling).

Follow-up beads: incremental race drawing; content-visibility for chapter sections.
