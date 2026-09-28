# Singularity 双采实测 / Two-pass results

2026-09-28. Same two DGX Spark / RoCE environment as the [original Ref2VA results](../2026-09-28/RESULTS.md), with a different checkpoint and two-pass pipeline. Do not combine their speedups.

**原双机 82.117 → 68.981 秒，耗时减少 16.00%；原生单机 → 优化双机，整片 1.927×，两遍采样合计 2.105×。整片超过两倍尚未实现。**

**Old dual → optimized dual saves 16.00% elapsed time. Native single → optimized dual is 1.927× overall and 2.105× for both sampling passes combined. Overall speedup is below 2×.**

## 固定输入 / Fixed inputs

768×448 → 1152×672, 56 frames at 24 fps (2.333 s), seed 20260928, no references in the benchmark. Singularity v1.3 INT8; Turbo LoRA 1.0 and LMS 0.3, all 50 AdaLN groups ported; Euler/simple; actual 2 + 10 steps; BF16 1.5× learned latent upscale. Dense Comfy Kitchen attention, FF chunks 2. Full prompt and settings: [benchmark.json](benchmark.json); model hashes: [profile.json](../../runtime/singularity/profile.json); actual [sigmas](sigmas.json), [AdaLN report](adaln.json).

Each group has four complete iterations. Exclude iteration 0 and take the median of iterations 1–3. Results are sequential, small-sample measurements, not randomized confidence intervals. First iterations include warm-up effects and are not clean-machine cold-start benchmarks.

| Group | First iteration | Warm median | Warm range |
|---|---:|---:|---:|
| Native single, fair baseline | 161.415 s | **132.940 s** | 129.890–133.920 s |
| Single with retained QKV cache | 152.650 s | 145.789 s | 145.239–148.046 s |
| Original dual, all-to-all | 113.534 s | 82.117 s | 81.750–84.112 s |
| Optimized dual, all-gather + cache + parallel VAE | 99.327 s | **68.981 s** | 66.873–70.191 s |

The cache slows the single-node control. We use the **faster native single**, not the slower cached single, as the speedup denominator. Native single forward counters are zero because that path has no wrapper instrumentation; this does not mean no denoising occurred.

| Warm median stage | Native single | Optimized dual | Speedup |
|---|---:|---:|---:|
| Both sampling passes | 116.694 s | 55.445 s | **2.105×** |
| Second sampling pass alone | 105.912 s | 49.134 s | 2.156× |
| Both video/audio decodes | 11.010 s | 7.681 s | 1.433× |
| Complete timed pipeline | 132.940 s | 68.981 s | **1.927×** |

Summed stage metrics are summed within each iteration before taking medians. Individual medians need not add up to the median total.

总时间包含提示词/参考编码、阶段切换、一采预览、潜空间放大、二采、两次解码/保存及部分证据记录；排除程序导入、初始模型对象构造和最终 latent 事后检查。保留同精度、步数、输入和输出要求，没有减少画质参数。

Total time starts at conditioning and ends after final MP4 save. It includes stage switching, the first preview, upscaling, both decodes/saves and some evidence work. Imports, initial model-object construction and final latent post-checks are excluded. This is inference optimization, not training.

## 实际变化 / Changes

- Chunked hidden-state all-gather with QKV head ownership, gather 4 / attention 4.
- Own the native caster's **effective LoRA-applied** INT8 QKV rows; preserve row scales/bias. Separate immutable Turbo/LMS caches avoid repeated materialization.
- Filter native QKV prefetch only when an owned matching-signature cache exists. Other prefetch stays native. The optimized run records 100 builds, 2300 hits and 2300 skipped prefetches per rank; retained copies cost 5,784,576,000 bytes per rank (~5.39 GiB).
- Dispatch native video VAE temporal chunks across both GPUs while keeping native overlap/blending/crop. Audio remains native on the head.

The main model, LoRA strengths, sampling schedule, precision and previews are unchanged. The four compute modules match the measured final backend; public packaging changes are disclosed in [provenance.json](provenance.json) and [AUDIT](../../AUDIT.md).

## 输出一致性 / Output checks

[media_checks.json](media_checks.json) contains 54 sanitized file-level records, with float-pixel hashes, two latent comparison results, encoded-file hashes, media parameters and full-decode exit codes. All have exact latent equality, max error 0, matching baseline float pixels and successful full decode. The four benchmark groups contribute **32 files**; each stage has one MP4 hash across those groups. Extra ablation/tuning files do not represent independent scenario coverage.

[reference.json](reference.json) records a first-frame image plus the complete 56-frame [generated reference clip](../../files/sailboat-832x480-56f.mp4). Both passes match the original reference baseline. The single cold run takes 181.002 s; **this is a functional parity regression, not a multi-run performance benchmark**.

- [No-reference optimized sample / 无参考样片](../../files/singularity-sailboat-1152x672-56f.mp4)
- [Image + video reference sample / 图＋视频参考样片](../../files/singularity-reference-1152x672-56f.mp4)

Technical equality and decoding checks do not replace subjective image/audio acceptance. Equality is established only for these fixtures, not every possible input.

## 复算与限制 / Recompute and limits

```bash
python3 scripts/verify_singularity.py
python3 scripts/verify_singularity.py --media
```

The second command additionally probes and fully decodes both shipped samples. Other files are represented by archived test records, not downloaded or regenerated by this check. Public runtime hashes are checked against the provenance manifest.

No qualified 2K/long-video result, active sparse attention, arbitrary mask/reference-slot support, general ComfyUI integration or transparent shared 256 GB memory pool is claimed. Custom-node installation revisions were not archived; clean-machine setup is not qualified. The public controller has CPU validation, **not a post-packaging GPU benchmark rerun**. Setup and flags: [SINGULARITY.md](../../docs/SINGULARITY.md).
