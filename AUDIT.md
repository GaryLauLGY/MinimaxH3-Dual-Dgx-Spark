# Validation and publication audit

Date: 2026-09-28. Scope: initial public release of the isolated dual-Spark H3 experiment.

## Verified on the original two-Spark experiment

- RoCE/NCCL communication and two-rank H3 forwards.
- Single-node, initial-dual and optimized-dual complete video generation: four outputs per series, with matched inputs and precision.
- Warm medians and all 12 media hashes / full-decode exit codes, published as sanitized numeric evidence.
- Six final forward variants with zero observed error, and same-latent native / parallel VAE pixel equality.

No production ComfyUI integration, clean-machine installation, Singularity workflow, alternate checkpoint or arbitrary-input correctness qualification is implied.

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
