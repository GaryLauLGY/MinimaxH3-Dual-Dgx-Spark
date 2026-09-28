# Singularity 双采 / two-pass runner

这是同一个双 Spark 项目的第二条后端路线。原 `runtime/lab_runner.py` 和 Ref2VA 实测保持不变；新代码在 `runtime/singularity/`，控制端用 `--workflow singularity` 选择。它实现已测模板的计算链，**不导入任意工作流 JSON，也尚未接入生产 ComfyUI 菜单**。

This is a second backend in the same project, selected with `--workflow singularity`. It retains the original Ref2VA runner and evidence. It is an isolated CLI implementation of the tested two-pass pipeline, not a general ComfyUI workflow importer or custom-node plugin. [Results / 结果](../results/2026-09-28-singularity/RESULTS.md).

## 计算链 / Pipeline

```text
Singularity v1.3 INT8 + Turbo LoRA 1.0
  → Euler/simple base 6 → ExtendIntermediateSigmas 2 → split 2/0
  → first pass: 2 low-resolution steps → preview video + audio
  → BF16 3D latent upscale 1.5×, align 32 → zero-step initialization
  → retain first-pass audio → Turbo + LMS 0.3 with AdaLN port
  → second pass: 10 high-resolution steps → final video + audio
```

默认 768×448 → 1152×672，56 帧 / 24 fps，seed `20260928`，四轮相同输入。Comfy Kitchen attention、FF chunks 2 / threshold 4096。BlockSparse 配置保留，但本次实际全部走 dense fallback；激活的稀疏注意力、非平凡遮罩、其他量化布局主动拒绝。第一遍预览也保存；没有通过减步或跳过预览制造加速。

Default: 768×448 → 1152×672, 56 frames at 24 fps, four repeats, fixed seed. Active sparse attention, nontrivial masks and other quantization layouts are not qualified. The first-pass preview is still decoded and saved. The native `CreateVideo` output replaces only the UI's VHS save nodes. Imported modules run in fresh processes; production source and weights are not edited.

## 环境与模型 / Requirements

先配置 [SETUP](SETUP.md) 中相同的 ComfyUI pin、环境及 RoCE。模型与 Ref2VA 路线不同；两端需安装下表文件，完整大小和 SHA256 见 [profile.json](../runtime/singularity/profile.json)。仓库不分发模型或安装第三方节点。

Use the same pinned ComfyUI/driver/PyTorch environment as [SETUP](SETUP.md). Both machines require these model files; the profile records all eight hashes, unlike the older Ref2VA manifest.

| ComfyUI model folder | File |
|---|---|
| `diffusion_models` | `Minimax-h3_Singularity_ref2va_v1.3_int8.safetensors` |
| `text_encoders` | `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` |
| `vae` | `minimax_h3_video_vae_int8_convrot.safetensors` |
| `vae` | `minimax_h3_audio_vae_fp32.safetensors` |
| `loras` | `minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors` |
| `loras` | `minimax_h3_lms_v1.0_r64.safetensors` |
| `latent_upscale_models` | `minimax_h3_latent_upscaler_3d_bf16.safetensors` |
| `diffusion_models/singularity_support` | `ref2va_adaln_TABLE_ONLY_NOT_MODEL.safetensors` |

还需现有安装内以下三个节点路径 / Three existing custom-node modules are required:

- `custom_nodes/ComfyUI-KJNodes/nodes/minimax_nodes.py` — `MiniMaxChunkFeedForward`.
- `custom_nodes/ComfyUI-H3-AdaLN-LoRA-Fix/__init__.py` — LMS curve-to-dense AdaLN port; all 50 groups must port, none may be stripped.
- `custom_nodes/Comfyui_Minimax_h3_latent_Upscaler/nodes/minimax_h3_latent_upscaler_3d.py` — `MinimaxH3LatentUpscaler3D`.

**复现限制：本次未归档这三个第三方节点的精确安装提交，不提供从干净机器一键安装的保证。** 预检只核对所需模块存在；不能据此确认任意新版兼容。AdaLN 的四个小型时间嵌入输入在 CPU FP32 拟合；本进程关闭模型目录缓存写入，并要求 `ported=50, stripped=0, unportable=0`。这两个适配仅作用于当前进程。

**Exact installed revisions of these custom nodes were not archived.** Preflight checks file presence, not revision/API equivalence; clean-machine installation remains unqualified. AdaLN basis fitting uses four small CPU FP32 inputs and disables model-directory cache writes in this process. Obtain dependencies and weights under their own licenses; they are not bundled here.

## 运行 / Run

编辑 `config.local.json`，确认两台没有其他 GPU 工作。下列命令顺序执行，每次换新 run ID。所有 `--compare-dir`、`--reference-video` 均为 **head 上的绝对路径**；控制端不会自动上传参考素材。默认仍为原 Ref2VA 路线，必须写 `--workflow singularity`。

Edit the existing local config. Run sequentially on idle machines; use new run IDs. Reference/comparison paths are absolute paths on the head, not controller-local paths. The default workflow remains Ref2VA for compatibility.

```bash
# Local only, no SSH or CUDA
python3 scripts/cluster.py run --workflow singularity --run preview \
  --config config.example.json --dry-run --parallel-vae --keep-stage-qkv
python3 scripts/verify_singularity.py --media

# Read-only remote preflight, including all eight model hashes
python3 scripts/cluster.py validate --workflow singularity --verify-weights

# Native single baseline (no retained QKV cache)
python3 scripts/cluster.py run --workflow singularity --run sg_single01 --world 1

# Original dual: all-to-all, native VAE, no retained stage cache
python3 scripts/cluster.py run --workflow singularity --run sg_old01 \
  --exchange alltoall --attention-chunks 1

# Optimized dual: four gather/attention chunks, stage caches, parallel video VAE
# Replace /ABSOLUTE/SCRATCH with the head node's configured scratch directory.
python3 scripts/cluster.py run --workflow singularity --run sg_dual01 \
  --parallel-vae --keep-stage-qkv \
  --compare-dir /ABSOLUTE/SCRATCH/runs/sg_single01/output --compare-run-zero
python3 scripts/cluster.py collect --run sg_dual01
```

`--compare-dir` 检查两遍 video/audio latent 逐元素一致和解码浮点像素 SHA；差异会导致失败。未指定时只记录本次证据，**不代表通过单机一致性对照**。参考对照必须使用相同参考、提示词与参数，不能拿无参考基线比较。

`--compare-dir` requires equal video/audio latents and float-pixel hashes for both passes. Without it, outputs are recorded but no baseline equality is established. Reference runs require a matched reference baseline.

### 参考入口 / Reference input

`--reference-video /absolute/path/boat.mp4` 将第一个解码帧作为 `ref_image_0`，整段视频作为 `ref_video_0`；这是一个图＋视频的固定入口，不是完整版 UI 的所有参考槽位。默认提示词随是否提供参考切换；`--prompt '...'` 可覆盖。实际提示词写进输出 `config.json`。

The reference option supplies its first decoded frame as an image plus the complete clip as a video. Default prompts match the archived sailboat tests; `--prompt` overrides them. All decoded frames are loaded in memory; this is not a streaming long-video loader. The shipped [Ref2VA sailboat](../files/sailboat-832x480-56f.mp4) is the exact reference source used in the recorded regression. No ControlNet is involved.

### 缓存与寻优 / Cache and tuning

缓存从原生 dynamic-VRAM caster 获取 **LoRA 已生效** 的 INT8 QKV 权重，再复制当前 rank 的头与逐行 scale/bias。Turbo/LMS 分开缓存；只有命中同签名 QKV 才跳过重复原生预取。实测每台缓存约 5.39 GiB，权重仍各存一份，未形成透明共享内存池。

Effective LoRA-applied QKV rows are copied into owned memory, never retained as evictable native pointers. Stage/signature checks prevent Turbo/LMS cross-contamination. Other native prefetch remains unchanged. The same cache made the single-node control slower, so the **faster native single** is the published baseline.

底层入口保留 `--tune` / `--tune-stage` / `--tune-grid` 用于研究；控制端普通生成不启用它们。寻优计时包含额外前向，不能作为出片成绩。更大分块 8/4 在一次第二遍前向筛选中仅快 0.86%，本轮没有采用，仍用 4/4。底层入口绕开控制端的主机/版本/队列检查。

The raw runner retains research tuning options (`--help` works without GPU dependencies). Tuning runs are not generation benchmarks. The 8/4 candidate gained only 0.86% in a limited forward test and was not adopted. Direct rank invocation bypasses controller preflight.

## 验证边界 / Qualification

已测固定小样：无参考四组各四轮，共 32 个 MP4；图＋视频参考一次完整一致性回归；连同筛选和消融，本轮累计 54 个 MP4 完整解码与各自基线一致。54 个文件不代表 54 种不同场景。未验证 2K、长片、其他模型、所有参考槽位或生产菜单集成。

The four benchmark groups have 32 files; reference parity has one cold iteration, not a timed speedup comparison. The 54-file total includes repeated inputs and ablations. 2K, long videos, arbitrary checkpoints and complete UI/reference integration remain unqualified. **The public controller/argument packaging has CPU tests but has not received a fresh two-node GPU run.** See [AUDIT](../AUDIT.md) and [source provenance](../results/2026-09-28-singularity/provenance.json).
