Vendored Laya files: what is and is not in this repository
==========================================================

`ML-HOST_CONTRACT.md` here is the publisher's host contract, byte for byte: its sha256,
`046647086b8da2fc592ab0f25052c3691ab4a0d801ca6b8256f36e4a59aed7bc`, is the `HOST_CONTRACT.md`
row (line 1) of both `ml-SHA256SUMS` and `en-SHA256SUMS`. `ML-NOTICE.txt` is the publisher's
`NOTICE` in the same way (sha256 `e3e3b3ea…`, line 2). Neither is edited, so the paths inside
the contract describe the publisher's package tree rather than this directory. This note is
repository-authored and maps the paths that do not resolve here.

| path in the contract | in this repository |
|---|---|
| `laya_host.py` (intro, §D) | Not vendored. The arithmetic it defines is ported to JavaScript in `laya-pack.js`, whose comments cite §B and §D of the contract. |
| `android/README.md` (intro, §E) | Not vendored; there is no `android/` directory here. |
| `README.md#minimal-usage` (§E) | Not vendored. |
| `tokenizer.json` (§A) | Vendored as `ml-tokenizer.json` (sha256 `609d8f4c…`, line 75) and, for the English package, `en-tokenizer.json` (`6c8aaa9a…`, the digest `laya-manifest.js` records for it). `tokenizer_config.json` is vendored under its own name. |
| `token_embeddings.json` (§C) | Not vendored. The table it describes, `token_embeddings_fp16.bin` (393,216,000 bytes), is fetched from the pinned revision at runtime by `laya-engine.js`; its digest is line 74. |
| `fixtures/gate_rows_s256.json` (§E) | Vendored as `gate_rows_s256.json` in this directory (sha256 `037bd3c8…`, line 58); there is no `fixtures/` subdirectory. |

Notes that matter:

- `ml-SHA256SUMS` is the multilingual package's own sums file at the pinned revision —
  `laya-manifest.js` records how that was established. `en-SHA256SUMS` is a copy of that list,
  not the English repository's, so it holds no row for any `laya_en_*` graph.
- For the contract and the notice, the matching row in the publisher's own sums is the only tie
  to the publisher's bytes recorded here. Nothing re-derives it at runtime: both files are served
  from this origin and never fetched. Editing either one breaks that tie.
