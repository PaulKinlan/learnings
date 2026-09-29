# Learnings

## Platform
Web, GitHub Pages.

## Stack
Approved: zero-dependency static HTML/CSS/ES modules, Node standard-library tests.

## Users and purpose
Paul and public readers exploring a topic through a sourced report and experiments. A hub for multiple learning topics; Jev and Kev are first, with Laya-LiteRT documented as an alternative edge architecture.

## Capabilities and constraints
Read a comprehensive report, inspect typed requests and probabilities, experiment with thresholds, run real on-device decisions in-browser via Laya (LiteRT.js) or Kev (ONNX Runtime Web), test the Chrome Built-in AI Classifier API proposal via `window.Classifier`, or query hosted Jev with your key. Generate a validated demo specification using user-supplied OpenAI or Claude keys. No execution of generated code. Keys remain memory-only. No analytics, external fonts or runtime CDN.

Real browser decision inference is delivered on-device: Kev runs in-browser via ONNX Runtime Web (`onnx-community/kev-0.6b-ONNX`), and Laya runs in-browser via Google LiteRT.js (`litert-community/Laya-Multilingual-LiteRT`). The Chrome Built-in AI Classifier API proposal is polyfilled on `window.Classifier` backed by Laya. Weights download on demand from Hugging Face; no user text leaves the tab. Illustrative and synthetic mode has been completely removed in favor of real working execution modes.

Only public, unauthenticated GitHub sources inform repository suggestions. No private data. Paid model requests require user action; no project-owned API credentials.

## Approved direction
Long-form readable report and split experiment workbench, light blue/ink palette, native controls, no generated images. Accuracy and explicit evidence status outrank marketing.
