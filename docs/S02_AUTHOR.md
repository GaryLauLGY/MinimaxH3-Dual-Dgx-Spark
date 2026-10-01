# S02 author-graph adapter and fixed INT8 route

This separately namespaced adapter executes an existing Singularity author graph on two DGX Spark hosts. It preserves the graph's inputs, model branches, Turbo/LMS schedule, latent upscale, video/audio processing and output node IDs. It does not ship a workflow, references or weights. The repository's original `runtime/` command-line runner is unchanged.

The optional speed change selects the **installed Comfy Kitchen CUDA backend's own cuBLAS INT8 fallback** for four eligible linear shapes inside LMS blocks. Quantization, effective LoRA weights, FP32 scales, INT32 accumulation, BF16 outputs and native fused SwiGLU are retained. Turbo, calls outside LMS, bias/residual and other dtypes retain native dispatch. There is no permanent scratch cache or new GPU synchronization. The module patches only the disposable author child process.

## Results and limits

See [the measured comparison](S02_FIXED_INT8_RESULTS.md). Three short pairs reduced whole-child runtime by a median 10.7541%; the first long pair reduced it by 6.6557%. The long result is one pair, not a long-video median. The owner accepted the measured improvement; this release does not claim a universal 10% speedup.

All paired samples passed exact eight-tensor/floating-pixel and full decoded-video comparisons. These comparisons preserve existing content defects; they are not a claim to fix motion, likeness or creative quality. The performance fixtures did not contain audio. An additional audio-bearing fixture exercises the deployed integration separately from the timing samples.

## Existing-installation integration

This is an advanced adapter for a qualified installation, not a one-command ComfyUI installer. Tested with ComfyUI revision `ee71d5c4993f29086b27fde1629a945ae48425bf`, PyTorch 2.11.0+cu130, and the installed CUDA Comfy Kitchen INT8 implementation. Source reference for Comfy Kitchen is v0.2.34, commit `e5e0d020e2add85f50466bb79171bc04ef7492b6`; source identity alone does not attest the installed binary. Private dispatcher APIs and model layouts may change: qualify a different installation before enabling.

Dependencies include the existing Singularity v1.3 INT8 model, Turbo/LMS LoRAs, INT8 video VAE, audio VAE, Qwen text encoder, KJNodes, H3 AdaLN fix, learned latent upscaler, rgthree and VideoHelperSuite. They remain under their own licenses and are not bundled. Actual BlockSparse execution, arbitrary graph patches and every reference combination have not been qualified.

For an already functioning S02 installation:

1. Confirm both hosts by hostname; check **each** queue and independent GPU processes. Production work takes priority.
2. Back up the exact existing backend files and each host's `config.local.json`, recording their hashes. Preserve models, input assets, runs and unrelated changes.
3. Install `runtime/s02/author_backend/author_runner.py` and `fixed_cublas.py` into the configured author backend on both hosts. Its sibling `progress_relay.py` and configured legacy backend must match the supplied source manifest. Keep the route off while staging both hosts.
4. Set `fabric_env.H3_S02_FIXED_CUBLAS_LMS` to string `"1"` on both hosts, preserving other environment entries. Use the same new `package_id` on both. Keep `timeout_s: 0`. The unchanged node reads config on each invocation and launches a fresh child; this backend-only update needs no service restart.
5. Check both `/h3-author/info` responses for matching package and unlimited timeout policy. Confirm a new owned child receives the route flag and writes `fixed_cublas_rankN.json` after LMS forwards. Complete an owned cancellation and exact output comparison before treating the integration as qualified.

`runtime/s02/plugin/` contains the S02 custom-node source for review. New installations must provision their own hostname/path/transport configuration, loopback control bridge and required custom nodes. Copying this plugin into an arbitrary installation is not sufficient. The original author UI graph is submitted normally with `extra_data.extra_pnginfo.workflow.extra.h3_dual_author` containing `enabled: true, world_size: 2`. Unmarked graphs keep their existing route. Unknown model-chain nodes are rejected rather than ignored.

Configuration consumed by `plugin/control.py` includes `role`, `expected_hostname`, `package_id`, `timeout_s`, `fabric_env`, `python`, `comfy_root`, `backend`, `legacy_backend`, `runs_root`, `cache_root`, `master_addr`, and `master_port`. The head additionally needs `worker_hostname` and a **loopback** `worker_url`. Use host-specific absolute paths and a dedicated writable run/cache area. Do not publish local configs or credentials.

## Recovery

After both hosts become idle, restore the backed-up `author_runner.py` and configs, including the previous matching package IDs and route settings. Remove only the newly installed `fixed_cublas.py` after verifying its recorded release hash; preserve unrelated files. No restart is needed for this backend-only rollback. A rapid opt-out can set the route to `"0"` on both hosts with a matching package identity; new children then use native routing.

Cancellation stops only the owned child process group and the exact worker UUID. `timeout_s: 0` removes the wall-clock limit without disabling manual cancellation. Changing the route does not change any already running process.

## Validation and provenance

CPU checks: `python3 -m unittest discover -s tests -v`. The optional `fixed_cublas_selftest.py` requires the matching CUDA environment and an idle GPU; it is not run by CI. It covers native/custom-op dispatch, two candidate forwards, native SwiGLU and guard branches. It writes its results under the S02 source root, so use an isolated writable copy.

The byte hashes of the qualified runtime sources are in [s02_source_sha256.json](../results/2026-10-01/s02_source_sha256.json). The Ulysses/block primitives and protocol retain the existing MIT upstream attribution in [CREDITS](../CREDITS.md). ComfyUI, CUDA, Comfy Kitchen and model dependencies are not redistributed. Operational host addresses, private prompts, workflow assets and original media are deliberately excluded.
