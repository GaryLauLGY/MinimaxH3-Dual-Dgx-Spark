# 实现、优化来源与边界

## 一条视频如何用到两台机器

控制端分别通过 SSH 启动两个独立 Python 进程，所有新增脚本与缓存位于配置的实验目录。两个进程各用本机 CUDA device 0，rank 0 负责文本编码、采样调度、最终输出；rank 1 等待每步指令并参与模型计算。生产 ComfyUI 服务的代码和进程不被打补丁。

rank 0 用 Gloo 发送元信息，用 NCCL 广播音视频 latent、timestep、context 等张量。每个 H3 block 按序列行分配本地计算。注意力采用 Ulysses 的序列 / 头分工，最后恢复原序列顺序。

`alltoall` 路线交换 QKV 的序列与头维度；`allgather` 路线收集隐藏状态，每个 rank 只投影自己负责的 QKV 头，再交换输出。通信与注意力分块允许异步传输与计算重叠。数学意图保持不变，实际数值仍需要逐配置验证。

本项目没有继续复制整段旧版 H3 `sp_forward`。当前 ComfyUI 原生 `_forward` 仍负责 embedding、payload 处理、最终 head 与音频缩放，只在重计算 block 的 `forward` 处接入并行原语，避免旧实现依赖已移除的 `time_shift_slope`。

## INT8 与推理模式适配

对 QKV 输出行分头时，INT8 ConvRot 权重对应的逐行 scale 必须按同样的行索引切片，bias 也要一起切。输入轴旋转保持不变；未知量化布局拒绝运行。这是本次模型正确性所需的适配，不是额外量化或改变精度。

当前量化 Parameter 在模型准备 / 转换时需要版本计数。早期 `inference_mode` 运行器触发 `Cannot set version_counter for inference tensor`；完整测试改为 `no_grad`。生成过程不计算梯度，不进行训练，不改变权重。

## VAE

视频 VAE 按当前原生 `_decode_temporal_chunks` 生成任务，两端交替计算 `_adaptive_decode` 的时间块。head 收回块后，仍调用原生 `vae.decode` 完成原有的重叠混合、裁切、像素处理和输出。单块会退回本机解码。音频 VAE 留在 head。

分块解码与跨卡分配已有上游先例。本项目的变化是接到当前原生 VAE 和跨主机调度；同 latent 完整像素对照在采样计时之外执行。

## 实现归属

| 项目 | 来源 / 本仓库工作 |
|---|---|
| Ulysses、all-to-all / all-gather、异步分块 | buqi-code 上游已有实现，本仓库抽取并适配 |
| Tensor metadata 协议 | buqi-code 上游派生；本地独立双 rank 入口 |
| 两台 Spark 的独立 SSH 启动与 NCCL 连接 | 本仓库适配及配置化 |
| 当前原生 H3 forward 的保留与 block 接入 | 本仓库版本适配 |
| INT8 ConvRot QKV 的 row scale / bias 切片 | 本次适配 |
| 当前原生 VAE 跨主机任务派发 | 本仓库适配，上游已有并行 VAE 思路 |
| 单机 / 双机 A/B、哈希、门槛、结果核算 | 本次实验与发布记录 |
| 目录与报告展示 | 参考 MiaAI-Lab 双 Spark DeepSeek 仓库组织方式 |

没有与 Mia 的 DeepSeek 引擎共享模型实现，也没有把 LLM 的 tokens/s 作为视频模型指标。H3 应以固定质量参数的采样秒数、完整生成秒数和正确性衡量。

## 当前兼容范围

| 场景 | 状态 |
|---|---|
| 本报告的 Ref2VA INT8 ConvRot 文生视频 | GPU 已验证 |
| 同配置单机与双机完整生成 | GPU 已验证 |
| VAE 同 latent 完整像素对照 | GPU 已验证 |
| 公共控制端、dry-run、结果复核 | CPU 已验证；公开封装尚无新的 GPU 回归 |
| Singularity 作者双采 / LoRA | 未验证，公共 CLI 也没有该工作流接线 |
| 参考图 / 视频 / 音频驱动 / ControlNet | 未验证；协议里有部分参考字段不等于整条路线受支持 |
| 非平凡 denoise mask、自定义 block replacement | 当前主动拒绝 |
| 其他 H3 checkpoint、任意 ComfyUI 版本 | 未验证 |
| 任意第三方 ComfyUI 节点、超过两台机器 | 不在当前接口范围 |
| DeepSeek / GLM / Qwen 语言模型 | 需要各自推理引擎，不能直接使用此 H3 adapter |

两端权重是复制存储。当前工作不是参数分片、跨机统一寻址、通用 GPU 内存池或原生 ComfyUI 队列集群。更长视频、更大尺寸和更多参考条件会改变算力 / 内存 / 通信比例，必须重新测量。

## Singularity backend

`--workflow singularity` stages only `runtime/singularity/` Python files, keeping its stage-aware primitives separate from the original Ref2VA backend. The native model forward remains authoritative; each LoRA stage dispatches its metadata and tensor inputs to the other rank.

`stage_qkv` obtains effective quantized weights through the native dynamic-VRAM caster, owns the selected head rows/scales/bias, and keys reusable copies by stage, projection, rank, device/dtype and patch signature. Turbo and LMS must not share an effective-weight cache. A matching cache permits skipping only that QKV module in native prefetch; other modules retain native behavior. The single-node cached control was slower, so retained caches are not silently enabled by default.

Both video previews use the existing native temporal VAE plan with alternating chunk ownership and native merge/crop. Audio stays on rank 0. These two passes reuse the existing nodes' sampler and learned upscaler, rather than reducing work or changing precision. [Pipeline, flags and limits](SINGULARITY.md).
