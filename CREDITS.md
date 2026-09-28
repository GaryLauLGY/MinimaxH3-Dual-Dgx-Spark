# Credits and provenance

## Parallel implementation

[buqi-code/buqi-minimax-h3-multigpu](https://github.com/buqi-code/buqi-minimax-h3-multigpu), source commit [`bce083929c135bdabd41679da3ddf6f409336ed3`](https://github.com/buqi-code/buqi-minimax-h3-multigpu/tree/bce083929c135bdabd41679da3ddf6f409336ed3), is the MIT-licensed source of the Ulysses attention/block primitives and tensor metadata protocol extracted into `runtime/sp_primitives.py` and `runtime/sp_protocol.py`. Parallel temporal VAE dispatch also builds on its existing idea.

The original copyright and MIT text are retained in [licenses/buqi-code-MIT.txt](licenses/buqi-code-MIT.txt). We do not claim to have invented Ulysses, chunked all-gather or multi-GPU VAE decoding. [ARCHITECTURE](docs/ARCHITECTURE.md) distinguishes inherited work from this adaptation.

## Native model/runtime

The runner uses an existing [ComfyUI](https://github.com/Comfy-Org/ComfyUI) installation and its MiniMax H3 native model, sampler, conditioning and VAE APIs. ComfyUI is not bundled here; its own license applies. H3 model checkpoints and their licenses are separate from this repository. No model weights are distributed.

## Repository inspiration

[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark) inspired the arrangement of quickstart documentation, dated results, scripts, tests, audit notes and explicit environment settings. No DeepSeek/vLLM engine code, measurements or performance claims were imported into this H3 project. There is no claimed affiliation or endorsement.

## Experiment and public packaging

GaryLauLGY provided the two-Spark environment and project direction. The experiment, adapter changes, benchmark analysis and public packaging were developed with Codex assistance. The sailboat prompt was authored for this test and uses no private reference assets.

Original tested runtime hashes, public runtime hashes and the packaging delta are recorded in [provenance.json](results/2026-09-28/provenance.json) and [AUDIT.md](AUDIT.md). The source upstream checkout is not needed at runtime; the required derived primitives are included with their license.

## Singularity two-pass extension

The same upstream attribution and MIT notice apply to the derived primitives in `runtime/singularity/sp_primitives.py` and `sp_protocol.py`. The effective LoRA-stage cache and process-local prefetch filter are this experiment's additions. Native sampling, dynamic VRAM, quantized weight casting, attention and VAE operations remain supplied by the installed ComfyUI runtime.

The pipeline follows the Singularity v1.3 author two-pass design (Turbo/LMS, AdaLN port and learned latent upscale); we do not claim authorship of the checkpoint, LoRAs or workflow design. It depends on installed ComfyUI-KJNodes, ComfyUI-H3-AdaLN-LoRA-Fix and Comfyui_Minimax_h3_latent_Upscaler components under their own terms. Those third-party sources, weights and the full author UI workflow are not redistributed in this repository. Exact installed custom-node revisions were not archived; see the setup limitation in [SINGULARITY](docs/SINGULARITY.md).
