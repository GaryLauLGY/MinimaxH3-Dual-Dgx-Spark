# 环境与运行 / Setup

本仓库复用两端已经可用的 ComfyUI H3 环境。它没有从裸系统安装到完整推理的验证记录；不要把它当作一键部署发行版。公共控制端在 CPU 上测试，已测 GPU 计算代码来自隔离实验，见 [AUDIT](../AUDIT.md)。

## 已测环境

| 项目 | 值 |
|---|---|
| 节点 | 2 × NVIDIA DGX Spark / GB10 / ARM64 / 每台 128 GB 统一内存 |
| 驱动 | 580.159.03 |
| PyTorch / CUDA | 2.11.0+cu130 / 13.0 |
| NCCL | 2.28.9 |
| comfy-kitchen | 0.2.34 |
| ComfyUI（两端一致） | `ee71d5c4993f29086b27fde1629a945ae48425bf` |
| 上游原语来源 | buqi-code `bce083929c135bdabd41679da3ddf6f409336ed3` |
| 通信 | 200 Gbps RoCE，`NET/IB/0` + `NET/IB/1`，GID 3，GDR 0 |

控制端需要 Python 3.11+、SSH、tar；收集需要支持 `--protect-args` 的 rsync（较旧 macOS 自带版本可能不支持）。远端需要上述 Python 环境、GNU `timeout`、tar、Git、`nvidia-smi`。无需 Ray、Docker 或两节点之间的 SSH 登录权限：控制端分别连接两台，模型通信走直连网络。

控制脚本核对 ComfyUI 的精确提交，并拒绝 `comfy/`、`comfy_extras/`、`nodes.py`、`folder_paths.py` 的已跟踪改动。环境版本会显示出来，不会自动升级或降级。请在独立、匹配的已有安装上运行，避免为了匹配版本修改正在使用的生产环境。

## 模型文件

两台均需放入各自 ComfyUI 默认模型搜索路径（或对应位置的现有链接）。此独立入口不加载服务器命令行传入的 `extra_model_paths.yaml`，不能只凭生产 UI 能选到模型就假定这里也能找到。

| ComfyUI 分类 | 文件 |
|---|---|
| `diffusion_models` | `minimax_h3_ref2va_int8_convrot.safetensors` |
| `text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `vae` | `minimax_h3_video_vae_fp16.safetensors` |
| `vae` | `minimax_h3_audio_vae_fp32.safetensors` |

DiT 文件 34,038,894,550 字节，SHA256 `9eef934046a0671bc8a5daf87100705e1478419c574cfde70c50fbe6885f76a9`。

视频 VAE SHA256 `7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522`。

本次没有归档文本编码器和音频 VAE 的哈希；它们只有文件名与环境记录，属于复现证据的限制。权重需从各自合法发布源取得，本仓库不分发权重，也不代替其许可。

## 配置

复制根目录 `config.example.json` 到 `config.local.json`，编辑下面的字段。示例 `192.0.2.10` 是文档保留地址，不能用于真实运行。

| 字段 | 用途 |
|---|---|
| `master_addr` | rank 0 的直连网卡 IPv4 地址，两端可达 |
| `master_port` | rendezvous 端口，默认 29628；同一机器不要并发使用 |
| `nodes[0]` / `nodes[1]` | rank 0（head）/ rank 1（worker） |
| `ssh` | 控制端已有的 SSH 别名或 user@host |
| `expected_hostname` | `hostname` 的准确输出；防止误连 |
| `python` | 该节点现有 ComfyUI 环境内 Python 的绝对路径 |
| `comfy_root` | 现有 ComfyUI 根目录的绝对路径，只读使用 |
| `scratch` | 当前用户拥有的独立实验目录，不能位于 ComfyUI 内或包含它 |
| `comfy_ports` | 该节点所有可能使用 GPU 的 ComfyUI 实例端口；空队列才允许启动 |
| `fabric_env.NCCL_IB_HCA` | 本机 RDMA 设备名，按实际链路填写 |
| `fabric_env.NCCL_IB_GID_INDEX` | 本机 RoCE GID 索引；已测值 3 不保证适用其他网络 |
| `fabric_env.NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` | 本机直连网卡名 |

可用 `rdma link`、`ibv_devinfo`、`ip -br addr` 只读查看设备；设备名与两端接口可能不同。SSH 密钥仍由系统 SSH 管理，不写入配置。`config.local.json`、运行日志和下载结果被 `.gitignore` 排除。

NCCL / Gloo 需要节点间连通，rendezvous 端口并不代表其全部连接都只用这一端口。仅在可信、隔离且已配置好的网络上运行；对象通信采用 PyTorch 的对象序列化，不能暴露给不可信节点。本脚本不改防火墙。

## 推荐运行顺序

```bash
# 只读；核对输出里的主机、模型、队列、GPU 进程及环境版本
python3 scripts/cluster.py validate --verify-weights

# 通信，再检查数值，再做完整视频对照；依次执行，不并发
python3 scripts/cluster.py run --run link01 --mode link
python3 scripts/cluster.py run --run parity01 --mode parity --width 320 --height 192 --frames 22
python3 scripts/cluster.py run --run single01 --world 1
python3 scripts/cluster.py run --run dual_initial01 --attention-chunks 1
python3 scripts/cluster.py run --run dual_optimized01 --parallel-vae

# 对齐原 r007 的顺序：先做六组前向选择，再四轮完整生成
python3 scripts/cluster.py run --run benchmark_then_video01 --mode benchmark \
  --benchmark-shapes 832x480x56 --generate-after-benchmark --parallel-vae

python3 scripts/cluster.py status --run dual_optimized01
python3 scripts/cluster.py collect --run dual_optimized01
```

启动器每次先预检，在两端创建全新 `<scratch>/runs/<id>/runtime`，只上传四个运行文件，再次检查队列，先启动 worker 再启动 head。输出位于 head 的 `<scratch>/runs/<id>/output/`；控制端日志在 `runs/<id>/rank0.log` 和 `rank1.log`。这些本地路径以仓库根目录为基准。

`collect` 从 head 复制输出到 `runs/<id>/collected/`，不会删除远端数据，目标存在则拒绝覆盖。`status` 只显示本地保存的退出结果，不是实时远端进程检查。日志中可能有主机名和路径，发布新结果前须单独脱敏。

超时默认每个 rank 为 3600 秒，远端 `timeout` 发 TERM 后再等待最多 30 秒强制结束该实验进程组。可用 `--timeout` 调整。Ctrl+C 结束本地 SSH 时，远端可能继续到 timeout；应在已配置的准确主机检查本次实验进程，不能用通用 `pkill python`。本工具不停止生产服务，也不发送清队列或 `/free` 请求。

预检不是资源预留机制。空队列不代表其他程序一定空闲；应核对打印的 GPU 进程，确保没有并行任务，测试期间不向这些机器追加工作。已有空闲模型占用内存时，先自行协调释放；本脚本不会自动清理别人的缓存。

## 数值检查与直接运行

`parity` 比较单次合成 latent 的原生 / 双机前向，检查有限值与相对最大误差。`benchmark` 每个候选一次预热、三次计时，并要求相对最大误差 < 2%、相对 L2 < 1%。这是工程检查阈值；本次观测值均为 0，不代表将来出现阈值内差异时画质必然可接受。

`--parallel-vae` 在第一轮计时之外，用同一最终 latent 做完整原生解码对照，最大像素差 > `1e-5` 则失败。以后改变模型或尺寸应重新检查。

要查看独立 rank 的原始命令，用 `run --dry-run`；查看底层入口参数用 `python3 runtime/lab_runner.py --help`。单独使用底层入口会绕开控制脚本的主机、版本和队列检查，需要自行保证这些条件。完整生成目前固定模型组件、24 fps、CFG 1、`simple` / `res_multistep`，不支持从任意 ComfyUI JSON 导入工作流。

## 常见问题

| 现象 | 优先检查 |
|---|---|
| ComfyUI pin 不匹配 / API 导入错误 | 对照本页已测提交，不要直接修改生产环境 |
| rendezvous / NCCL 超时 | 两端 rank、直连地址、接口、RDMA 设备、GID 和可信网络连通性 |
| worker 提前退出 | 先看 `rank1.log` 的第一条异常，head 的后续超时常是连带结果 |
| 模型找不到 | 默认模型搜索目录、文件名、符号链接目标 |
| OOM / 可用内存不足 | 现有模型与其他任务占用；权重是复制而非分片存储 |
| 小尺寸双机更慢 | 通信与同步成本可能超过节省的计算，见较小尺寸的实测 |
| 数值 / VAE gate 失败 | 保留日志与参数；该配置不能作为已验证加速结果 |
