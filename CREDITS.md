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

## S02 author-graph adapter

The attribution and MIT notice above also cover the derived primitives and metadata protocol in `runtime/s02/legacy_backend/`. The author-graph adapter preserves the installed Singularity workflow design and native ComfyUI operations. The fixed INT8 module selects the installed Comfy Kitchen CUDA backend's existing cuBLAS fallback; it does not redistribute CUDA or Comfy Kitchen. Singularity models, Turbo/LMS LoRAs, custom-node dependencies, author UI graphs and private test assets are not bundled.
