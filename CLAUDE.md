# Project Instructions for AI Agents

This file provides instructions and context for AI coding agents working on this project.

<!-- BEGIN BEADS INTEGRATION v:1 profile:minimal hash:6cd5cc61 -->
## Beads Issue Tracker

This project uses **bd (beads)** for issue tracking. Run `bd prime` to see full workflow context and commands.

### Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --claim  # Claim work
bd close <id>         # Complete work
```

### Rules

- Use `bd` for ALL task tracking — do NOT use TodoWrite, TaskCreate, or markdown TODO lists
- Run `bd prime` for detailed command reference and session close protocol
- Use `bd remember` for persistent knowledge — do NOT use MEMORY.md files

**Architecture in one line:** issues live in a local Dolt DB; sync uses `refs/dolt/data` on your git remote; `.beads/issues.jsonl` is a passive export. See https://github.com/gastownhall/beads/blob/main/docs/SYNC_CONCEPTS.md for details and anti-patterns.

## Agent Context Profiles

The managed Beads block is task-tracking guidance, not permission to override repository, user, or orchestrator instructions.

- **Conservative (default)**: Use `bd` for task tracking. Do not run git commits, git pushes, or Dolt remote sync unless explicitly asked. At handoff, report changed files, validation, and suggested next commands.
- **Minimal**: Keep tool instruction files as pointers to `bd prime`; use the same conservative git policy unless active instructions say otherwise.
- **Team-maintainer**: Only when the repository explicitly opts in, agents may close beads, run quality gates, commit, and push as part of session close. A current "do not commit" or "do not push" instruction still wins.

## Session Completion

This protocol applies when ending a Beads implementation workflow. It is subordinate to explicit user, repository, and orchestrator instructions.

1. **File issues for remaining work** - Create beads for anything that needs follow-up
2. **Run quality gates** (if code changed) - Tests, linters, builds
3. **Update issue status** - Close finished work, update in-progress items
4. **Handle git/sync by active profile**:
   ```bash
   # Conservative/minimal/default: report status and proposed commands; wait for approval.
   git status

   # Team-maintainer opt-in only, unless current instructions forbid it:
   git pull --rebase
   git push
   git status
   ```
5. **Hand off** - Summarize changes, validation, issue status, and any blocked sync/commit/push step

**Critical rules:**
- Explicit user or orchestrator instructions override this Beads block.
- Do not commit or push without clear authority from the active profile or the current user request.
- If a required sync or push is blocked, stop and report the exact command and error.
<!-- END BEADS INTEGRATION -->


## Build & Test

This repository has zero npm dependencies and no build step (`site/` is deployed directly to GitHub Pages).

```bash
npm start                # Start local static server + loopback API relay (scripts/serve.mjs)
npm test                 # Run Node standard-library test suite (node --test tests/*.test.js)
npm run test:browser     # Run headless Chromium CDP integration checks (tests/browser.mjs)
npm run test:on-device   # Run real in-browser LiteRT.js + ONNX Runtime Web smoke checks (tests/on-device.mjs)
```

## Architecture Overview

- `site/index.html` & `site/style.css`: Top-level Learnings Hub and shared zero-dependency stylesheet.
- `site/decision-models/`: Field guide, interactive decision lab, prompt-based playground (`window.Classifier` polyfill), 37-entry use-case catalogue, and on-device inference engines (`laya-engine.js` via LiteRT.js and `kev-engine.js` via ONNX Runtime Web).
- `site/neural-networks/`: Interactive first-principles Neural Network lab (pure `Float32Array` autograd + raw WebAssembly GEMM kernel) covering Perceptrons/MLPs/Backprop, CNNs, Transformers & Self-Attention, Continuous Image & Discrete Masked Text Diffusion, Decision Pointer Heads, and a composable Block Builder.
- `site/celld/`: Interactive architecture deep-dive and simulator for `celld` (SQLite/LTX cell engine) and Cloudflare Durable Object fleet architecture (CAS epoch fencing, `fleet` vs `bucket` durability, load balancing, and fleet-wide usage accounting & querying).
- `site/opt-chronicles/`: Interactive logbook and cluster failure simulator for Meta AI's OPT-175B training run.
- `scripts/serve.mjs`: Zero-dependency loopback HTTP server and strict same-origin JSON API relay (`/api/jev`, `/api/openai`, `/api/claude`).
- `tests/`: Node `node:test` unit suites (`*.test.js`) and zero-dependency WebSocket CDP browser drivers (`tests/lib/cdp.mjs`, `tests/browser.mjs`, `tests/on-device.mjs`).

## Conventions & Patterns

- **Strict CSP Everywhere**: Every HTML page must include a `<meta http-equiv="Content-Security-Policy">` with `default-src 'none'; script-src 'self' 'wasm-unsafe-eval'; style-src 'self'`. Never use inline `<script>`, inline `style="..."` attributes, or inline event handlers (`onclick=...`).
- **No Unaudited Dependencies**: Keep `package.json` free of runtime or test framework dependencies; any browser WASM/runtime vendor bundles live in `site/vendor/` and are documented in `site/vendor/NOTICE.md`.
- **Untrusted DOM Rendering**: Render model or user outputs using `document.createElement` and `textContent`, never `innerHTML` with untrusted strings.
- **Accessibility**: Include `<a class="skip" href="#main">Skip to content</a>`, `<main id="main">`, visible `:focus-visible` outlines, keyboard handlers (`Enter`/`Space`) on custom interactive controls, and `44px` minimum touch targets.
