# Changelog

## 2026-09-28 — Initial public experiment

- Publish a two-host, isolated native-ComfyUI H3 adapter with inherited MIT attribution.
- Include current-forward integration, INT8 ConvRot QKV scale slicing, `no_grad` compatibility and native temporal VAE dispatch.
- Record 104.087 s single-node versus 46.410 s optimized-dual warm generation medians for the fixed 832×480×56 test; include first-run values, raw measurements, equal media hashes and slower small-shape results.
- Add a configurable SSH controller, read-only preflight, unique run directories, dry-run, result verification and CPU CI.
- Provide Chinese/English overviews, setup instructions, architecture, scope limits and a generated sailboat sample.
- GPU results predate public-controller packaging; fresh two-node packaging regression remains pending.

## Singularity two-pass extension — 2026-09-28

- Add a separate `--workflow singularity` backend; retain default Ref2VA behavior.
- Add effective Turbo/LMS QKV stage caches, matching-cache prefetch filtering, and parallel temporal video VAE integration for the two-pass pipeline.
- Publish four-group timing evidence, all 54 technical-check records, exact model hashes, reference regression metadata and two generated samples.
- Measured: 132.940 → 68.981 s versus native single (1.927× overall, 2.105× sampling); 16.00% less elapsed time than original dual. Overall 2× is not reached.
- Add CPU coverage and explicit packaging/custom-node/2K/long-video qualification limits; no production ComfyUI deployment.
