# MiniMax H3 · Dual DGX Spark

[中文](README.md) · [Setup](docs/SETUP.md) · [Measured results](results/2026-09-28/RESULTS.md) · [Credits](CREDITS.md)

An experimental, isolated ComfyUI-native H3 inference runner for **two NVIDIA DGX Spark machines connected over RoCE**. Both GPUs cooperate on the same video using sequence/head parallelism and optional native temporal VAE chunk dispatch.

**Measured warm generation: 104.09 s → 46.41 s (2.24×). Sampling: 95.54 s → 39.84 s (2.40×).**

The result applies to Ref2VA INT8 ConvRot, 832×480, 56 frames at 24 fps, 20 steps, seed 20260928, no LoRA and no reference inputs. Each series contains four runs; medians use runs 1–3. Timing covers prompt conditioning through MP4 save and excludes initial model construction/loading. This is not a cold-start or all-workloads speedup claim.

| Warm median | Single | Initial dual | Optimized dual |
|---|---:|---:|---:|
| Sampling | 95.54 s | 40.82 s | 39.84 s |
| Video + audio decode | 6.15 s | 6.65 s | 4.24 s |
| Conditioning through save | 104.09 s | 49.87 s | 46.41 s |

All 12 recorded MP4 files share one SHA256 and passed full decoding. On the tested final latent, parallel/native VAE pixels were exactly equal. This evidence does not establish bitwise equivalence for arbitrary inputs. [Raw data and limitations](results/2026-09-28/RESULTS.md).

[![Frames 0, 24 and 48](files/sailboat-frames-0-24-48.jpg)](files/sailboat-832x480-56f.mp4)

[Sample MP4 with generated audio](files/sailboat-832x480-56f.mp4).

## Scope

- Existing matching ComfyUI environments and local model weights are required; no driver, network, model download or production-service installation is performed.
- Weights are replicated. This is **not a transparent 256 GB shared GPU memory pool**. The tested RoCE path logged `NET/IB` and `GDR 0` (host staging).
- This is a standalone CLI backend, not a general ComfyUI custom-node integration. Singularity two-pass workflows, LoRAs, reference inputs, ControlNet, long videos and other H3 checkpoints remain unqualified.
- GPU measurements come from the original isolated runner. The published controller/path packaging has CPU checks; a fresh two-node GPU regression of that packaging is still pending. Compute provenance is recorded in [AUDIT.md](AUDIT.md).

## Run

Controller: Python 3.11+, SSH, tar; collection additionally needs rsync with `--protect-args`. Nodes: the [pinned environment](docs/SETUP.md), installed weights, GNU timeout, idle GPUs and configured RoCE.

```bash
git clone https://github.com/GaryLauLGY/MinimaxH3-Dual-Dgx-Spark.git
cd MinimaxH3-Dual-Dgx-Spark
cp config.example.json config.local.json
# Edit all placeholders, including exact hostnames, paths and fabric settings.

python3 -m unittest discover -s tests -v
python3 scripts/verify_results.py
python3 scripts/cluster.py run --config config.example.json --run preview --dry-run
python3 scripts/cluster.py validate --verify-weights

# Sequential, on idle machines; a unique run ID is required each time.
python3 scripts/cluster.py run --run link01 --mode link
python3 scripts/cluster.py run --run single01 --world 1
python3 scripts/cluster.py run --run dual01 --parallel-vae
python3 scripts/cluster.py collect --run dual01
```

The controller checks the configured ComfyUI queues twice before launching, never empties them, and stages four runtime files under a new scratch run directory. It does not reserve GPUs against concurrent submissions or inspect every possible third-party scheduler. Keep the machines idle during the experiment.

Default generation settings match the measured case. Dual attention uses all-gather with four gather chunks and four attention chunks; `--parallel-vae` enables temporal decode dispatch. Inspect `runs/<id>/rank*.log`, use `status --run <id>` for the recorded exit codes, and see [SETUP](docs/SETUP.md) for timeouts, collection and direct rank commands.

## Contributions and attribution

This project adapts existing Ulysses primitives to two independent hosts and current native ComfyUI H3 forwards, handles row-wise INT8 ConvRot scale slicing, uses `no_grad` for quantized parameter compatibility, selects communication/attention chunking with correctness gates, and dispatches native VAE chunks while retaining native blending. It includes matched-input performance and output evidence.

The underlying parallel primitives and ideas come from [buqi-code/buqi-minimax-h3-multigpu](https://github.com/buqi-code/buqi-minimax-h3-multigpu). Repository presentation was inspired by [MiaAI-Lab's dual-Spark DeepSeek project](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark). No first-in-the-world or apples-to-apples superiority claim is made. See [CREDITS](CREDITS.md), [architecture](docs/ARCHITECTURE.md) and [MIT license](LICENSE). Model licenses are separate.
