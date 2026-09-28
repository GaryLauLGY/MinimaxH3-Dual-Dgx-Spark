# 2026-09-28 · Dual DGX Spark H3 实测

## 结论与口径

固定 Ref2VA INT8 ConvRot、832×480、56 帧 / 24 fps、20 步、seed 20260928、无 LoRA、无参考输入。`BasicGuider`（CFG 1）、`simple` scheduler、`res_multistep` sampler、comfy-kitchen attention。完整自编提示词与环境在 [config.json](config.json)。

三组各连续生成四轮，排除第 0 轮，取第 1/2/3 轮的中位数；阶段各自取中位数，所以阶段中位数之和不要求等于总耗时中位数。完整生成计时从 conditioning 开始，到 MP4 保存及 CUDA 同步结束。初始权重读取、模型构造、SSH、预检不计入。

| 热运行中位数 | 单机 r006 | 初版双机 r005 | 优化双机 r007 |
|---|---:|---:|---:|
| Conditioning | 1.760 s | 1.803 s | 1.753 s |
| Sampling | 95.542 s | 40.821 s | 39.837 s |
| Video + audio decode | 6.153 s | 6.650 s | 4.244 s |
| Save | 0.538 s | 0.565 s | 0.554 s |
| Total | **104.087 s** | **49.867 s** | **46.410 s** |

采样加速 `95.54161622887477 / 39.83700829092413 = 2.398313×`。

完整生成加速 `104.0866868630983 / 46.41018546000123 = 2.242755×`。

优化双机相对初版双机节省约 **6.93% 总时间**。这次同时把注意力分块从 1 改为 4，并启用并行视频 VAE；不能把总收益精确归因于其中一项。

| 系列 | GPU / 通信配置 | 视频 VAE |
|---|---|---|
| single | 1 个 Spark；原生 H3 | 原生单机 |
| dual_initial | 2 个 Spark；all-gather 4 块；attention 1 块 | head 原生单机 |
| dual_optimized | 2 个 Spark；all-gather 4 块；attention 4 块 | 原生时间块分配到两端 |

逐轮原始数值：[single.json](single.json)、[dual_initial.json](dual_initial.json)、[dual_optimized.json](dual_optimized.json)。仅去除了输出绝对路径，计时数值原样保留。汇总：[summary.json](summary.json)。

## 首次运行与统计限制

| 第 0 轮 total（不含初始构造 / 权重加载） | 秒 |
|---|---:|
| 单机 | 209.472 |
| 初版双机 | 163.660 |
| 优化双机 | 103.108 |

r007 在生成前运行过前向 benchmark，因此它的第 0 轮与前两组不是公平的冷启动对照。单机 DiT 初次加载另用了约 204 秒。**本报告没有给出从空进程开始到首条视频完成的单机 / 双机加速比。**

只有一个完整生成输入、每组 3 个热运行样本。组间按顺序运行，未交错随机化；没有跨日重复、置信区间、功耗或能效测量。大于 2 的比例是该环境、该输入的观测，不是理论扩展效率保证。更高效的矩阵形状、工作集和通信重叠可能解释部分收益，但尚无 profiler 证据将这些原因分离。

## 正确性与媒体

- 原始记录中的 12 个 MP4 均为 SHA256 `192475cd29b1c9bed1db0d19d86d81e73c8c661d4fb2d3dcc20ed41943f24d92`，大小 661,804 字节。仓库只放一份相同样本以免重复。
- 全部 12 条均经 `ffprobe` 与 `ffmpeg -xerror` 完整解码，退出 0；见 [media_checks.json](media_checks.json)。公开记录只保留媒体字段、哈希、系列 / 轮次与退出码，去除了主机路径。
- H.264、832×480、56 帧、24 fps，视频 / 容器约 2.333333 秒；AAC 32 kHz 双声道，音轨约 2.325 秒。音轨略短是实际编码结果，未声称采样级等长。
- 同一最终 latent 的完整 VAE 像素对照 `exact=true`、`max_abs=0`、形状 `[56,480,832,3]`，见 [vae_parity.json](vae_parity.json)。对照在第 0 轮生成计时之外执行。
- r007 六组前向候选的音频 / 视频误差均为 0，参考 RMS 非零，见 [benchmark.json](benchmark.json)。

相同 MP4 哈希只证明本次编码文件相同；不应据此声称所有潜变量、所有输入或其他环境都逐位等价。技术通过也不代替用户对画面、动作与声音的最终验收。

## 配置选择与反例

r007 的 832×480×56 合成前向：每个候选 1 次预热 + 3 次计时。使用 `torch.no_grad()`，与完整生成一致。最优为 all-gather 4 / attention 4，中位数约 1.911 秒；原生约 4.564 秒。这是单次前向，不是完整视频总耗时。

此前 r004 做过三档尺寸诊断，记录在 [synthetic_earlier_inference_mode.json](synthetic_earlier_inference_mode.json)：

| 合成 latent 等效尺寸 | 原生前向 | 最快双机前向 | 比率 |
|---|---:|---:|---:|
| 320×192×22 | 0.292 s | 0.322 s | **0.91×（更慢）** |
| 832×480×56 | 4.366 s | 1.915 s | 2.28× |
| 1280×736×56 | 11.920 s | 6.254 s | 1.91× |

此较早诊断使用 `inference_mode`。随后真实采样触发量化 Parameter 的 version counter 错误，修为 `no_grad` 后才完成 r005–r007。因此该表仅作历史诊断，不能与完整视频时间混用，也不是当前公开运行器可以逐字节重放的旧代码快照。

## 链路

[link.json](link.json) 记录两端 12 次 all-reduce 的单次均值。原脚本字段 `MB` 实际表示 MiB：16 MiB 约 20.68 GB/s，64 MiB 约 17.90 GB/s，均为算法吞吐计算 `payload_bytes / seconds`。它不是额外测得的总线带宽，更不是 H3 加速比。

网络观测为 200 Gbps RoCE，NCCL `NET/IB/0`、`NET/IB/1`，GID 3，`GDR 0`。主机暂存仍在路径中；没有证明 GPU Direct RDMA，也没有把两台 Spark 的内存变成一致性共享池。

## 复核

```bash
python3 scripts/verify_results.py
python3 scripts/verify_results.py --media  # 额外需要 ffprobe / ffmpeg
```

第一条重算中位数与加速比，检查 12 份媒体记录、样本文件 SHA、VAE / 前向数值记录以及公开运行代码的 SHA。第二条另外探测并完整解码仓库内的视频。两条均不启动 GPU 或远端任务。

已测脚本 SHA、公开脚本 SHA、函数语法树变更范围在 [provenance.json](provenance.json)。公开控制脚本为打包阶段新增；尚未再次做双机 GPU 回归，详见 [AUDIT](../../AUDIT.md)。
