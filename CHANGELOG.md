# Changelog

## 2026-09-28 — Initial public experiment

- Publish a two-host, isolated native-ComfyUI H3 adapter with inherited MIT attribution.
- Include current-forward integration, INT8 ConvRot QKV scale slicing, `no_grad` compatibility and native temporal VAE dispatch.
- Record 104.087 s single-node versus 46.410 s optimized-dual warm generation medians for the fixed 832×480×56 test; include first-run values, raw measurements, equal media hashes and slower small-shape results.
- Add a configurable SSH controller, read-only preflight, unique run directories, dry-run, result verification and CPU CI.
- Provide Chinese/English overviews, setup instructions, architecture, scope limits and a generated sailboat sample.
- GPU results predate public-controller packaging; fresh two-node packaging regression remains pending.
