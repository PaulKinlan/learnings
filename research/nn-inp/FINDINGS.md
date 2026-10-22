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

2. **The page-side cost on gradient-descent is mostly NOT the drawing.** Direct
   instrumentation (wrapping `requestAnimationFrame` and the canvas API before document
   load) shows the real per-tick canvas work is ~1ms (putImageData 0.4–1.2ms, all strokes
   ~0.1ms) and the animation runs only ~4–10 ticks. The big ticks traced as FunctionCall
   400–700ms decompose into: (a) a one-time `background()` heat-map generation (230k
   surface evaluations, ~300–400ms cold inside the first tick — the surface itself costs
   2.3ms warm for all 230k calls in node), and (b) V8 Scavenger minor-GC pauses of ~95ms
   each (measured 94–95ms per pause, i.e. environment-stretched; triggered by page-load
   allocations, not by the ticks' own ~100–200KB). The devtools-timeline "704ms tick" of
   the first run was this combination under heavier VM load. Fix direction
   (learnings-am8, landed): allocation-free hot loops (scalar coordinate math in the same
   order as `toPx()`, index bounds instead of `slice()`, a loop instead of `flatMap()`),
   bit-exact rendering verified by canvas checksums; the residual stalls are GC and VM
   load, not page code.

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

## Measurement caveat (2026-10-22, updated)

Absolute numbers on this VM swing with fleet load: the same page measured
presentTails of ~90–200ms in one run and caught a single 3.4s stall in another;
an idle run of `architectures.html` once showed a 3.7s native long task with
zero input and zero JS (pure CPU starvation). The decomposition (handlers cheap;
gradient-descent animation frames expensive; load-time MathML layout; control
page equal to chapter pages) was stable across runs — the magnitudes are not.
Do not compare absolute ms across runs or gate quality calls on them. Related:
the fast gate for 78534b7 once failed on `tests/cdp-cleanup.test.js` ("runner
did not launch", 20s DevTools-endpoint timeout at load ~5.8; passes 3/3 in
isolation — filed as learnings-gib).
