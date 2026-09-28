# Validation and publication audit

Date: 2026-09-28. Scope: initial Ref2VA release and the subsequent Singularity two-pass update. Historical initial-release findings below remain scoped to Ref2VA.

## Verified on the original two-Spark experiment

- RoCE/NCCL communication and two-rank H3 forwards.
- Single-node, initial-dual and optimized-dual complete video generation: four outputs per series, with matched inputs and precision.
- Warm medians and all 12 media hashes / full-decode exit codes, published as sanitized numeric evidence.
- Six final forward variants with zero observed error, and same-latent native / parallel VAE pixel equality.

The initial release did not qualify production ComfyUI integration, clean-machine installation, Singularity or arbitrary-input correctness. The separate Singularity update is described below.

## Changes made while preparing the public repository

The four runtime files were copied from the optimized experiment's saved snapshot, not from an unrecorded rewrite.

`sp_primitives.py`: documentation corrected to describe the current native-forward adapter; the unused version-description constant now names the measured ComfyUI commit. Function and class ASTs are unchanged.

`sp_protocol.py`: stale local-worker documentation corrected for the cross-host adapter. Function and class ASTs are unchanged.

`parallel_vae.py`: byte-for-byte identical to the tested file.

`lab_runner.py`: removed the private fabric IP default and unused upstream checkout argument, added argument checks, and changed the recorded ComfyUI commit from a hard-coded string to a live Git lookup. Among functions, only `generate` changed, solely in its configuration-record field. Sampling, block dispatch, tensor protocol and VAE computation are unchanged. [provenance.json](results/2026-09-28/provenance.json) contains both sets of hashes and the AST comparison counts.

`scripts/cluster.py` is a new portable controller replacing the private host/path-specific launcher. It adds config validation, pinned-runtime/queue preflight, per-run file staging, bounded execution, logs and collection. This new controller has CPU tests and a no-network dry-run. **It has not received a fresh two-node GPU end-to-end run after packaging.** The published 2.24× result belongs to the recorded original experiment, not a claimed retest of this controller.

## Local / CI checks

`python3 -m compileall -q runtime scripts tests`

`python3 -m unittest discover -s tests -v`

`python3 scripts/verify_results.py`

`python3 scripts/verify_results.py --media` additionally probes and fully decodes the shipped sample when ffmpeg tools are installed. CPU CI validates control logic, syntax and archived evidence; it cannot validate CUDA kernels, NCCL connectivity or performance.

## Publication boundaries

Only selected source files, explicitly allowed benchmark/media fields and a generated sample are included. Original hostnames, usernames, absolute machine paths, fabric/management addresses, private configurations and raw logs are omitted. Local runtime logs and local config files are ignored by Git. Model weights and credentials are not included.

Known limitations: text-encoder/audio-VAE hashes were not archived; only a small, sequential set of timing samples exists; the head's `gpu_peak_GiB` is a cumulative allocator peak, not a per-stage or two-node memory total; encoded-file equality does not prove universal latent equality. These limitations are retained in the public report rather than replaced by stronger claims.

## Singularity update

Five backend files are added under `runtime/singularity/`; the original Ref2VA runtime and its evidence are unchanged. The new backend retains the native dynamic-VRAM setup, full native model forward, two-pass 2+10 schedule, effective Turbo/LMS LoRA weights, AdaLN port, latent upscale and both output previews.

- `stage_qkv.py` and `parallel_vae.py` are byte-identical to the final tested optimization source.
- `sp_primitives.py` and `sp_protocol.py` have documentation corrections (and an unused pin-label correction); all function/class ASTs are unchanged.
- `singularity_runner.py` removes the private master-address default, validates arguments before GPU imports and adds an optional prompt override. The only changed function AST is `generate`, where two lines apply that override. The measured default prompts and computational pipeline remain unchanged.
- The final private runner already contained the single-rank positional-argument fix and expanded research tuner options after r020. The four compute modules match r020; r024/r025/r022/r023/r026 cover subsequent applicable paths. Per-benchmark source hashes remain in the archived evidence; they are not overwritten with public hashes.
- The shared controller selects per-workflow defaults/files/models and rejects incompatible options. Its Singularity preflight checks eight model files/hashes and three required custom-node paths. Exact custom-node installation revisions were not archived: file presence is not an API-compatibility or clean-install guarantee.

[Provenance](results/2026-09-28-singularity/provenance.json) records original/final public source hashes and AST changes. [Benchmark data](results/2026-09-28-singularity/benchmark.json) preserve all four iterations of all four groups, including the slower cached-single control. [54 media records](results/2026-09-28-singularity/media_checks.json) retain whitelisted technical evidence and no private absolute paths. Two generated samples are included byte-for-byte.

Added CPU tests cover workflow selection, backend-specific staging, no-network dry runs, path/prompt quoting, invalid options, preflight profile selection, no-GPU help/validation, and numerical recomputation of the published evidence. `scripts/verify_singularity.py --media` additionally fully decodes both shipped samples. **There has been no fresh two-node GPU end-to-end execution of the public packaging.** The measured speedups belong to the saved original experiments.

This update does not deploy to or modify any production ComfyUI instance. It does not qualify 2K/long videos, arbitrary masks/checkpoints, active sparse attention or all UI reference slots. Technical checks do not substitute for visual/audio acceptance. No new transparent shared-memory capability is claimed.
