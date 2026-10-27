# DeltaForge

DeltaForge 为 Qwen 模型中的 Gated DeltaNet（GDN，一种用状态矩阵保存历史信息的计算）提供可替换后端，研究数值稳定性、图编译和完整训练步性能。

第一次阅读只打开 [一步步认识 DeltaForge 项目](一步步认识DeltaForge项目-friendly.md)。它默认已掌握 embedding、Q/K/V、PyTorch 和训练基础，用 72 个小步骤依次解释 GDN state、chunk、旧算法故障、block 修复、后端门禁与真实 LoRA 结果。读懂后进入 [完整开发实战](DeltaForge完整开发实战-friendly.md)，实际操作代码检索、验证、编译、构建、证据重算和监控。其他材料从 [阅读入口](READING_GUIDE.md) 按需查证。

## 当前结果

在单张 H20、固定 Qwen3.5-2B、BF16、batch 1、256-token bucket 的 LoRA 训练中，分块求解与直接编译版本通过原定数值检查。LoRA 只训练附加的小矩阵；本轮共有 8,409,600 个可训练参数。三个合格后端各完成两个同 seed 的独立进程，每进程十次更新，其中八次计时。

compiled 完整训练步中位数为 0.38–0.41 秒，torch 参考为 0.82–0.84 秒，同重复性能比为 2.02–2.20×。进程共享磁盘编译缓存，显存峰值均约 4.516 GiB。短程结果支持固定配置下的兼容性与速度观察，尚不能判断模型质量或长期训练收益。

FLA 是已有的高性能 GDN 实现。本轮随机算子测试中它明显更快；原生默认与 FLA 的真实模型梯度未通过本项目门槛，原因尚未定位，因此还没有完成与它们的完整训练性能比较。完整条件、失败记录和支持范围见 [技术报告](TRAINING_RESULTS.md)。

## 文档与代码

| 用途 | 入口 |
|---|---|
| 唯一首读教程 | [一步步认识 DeltaForge 项目](一步步认识DeltaForge项目-friendly.md) |
| 深度工具链与完整开发流程 | [DeltaForge 完整开发实战](DeltaForge完整开发实战-friendly.md) |
| 原理、结果、接入和性能观测的查证地图 | [READING_GUIDE.md](READING_GUIDE.md) |
| 真实模型后端接口与复跑 | [TRAINING_INTEGRATION.md](TRAINING_INTEGRATION.md) |
| 每项任务的问题、候选、决定和 timeline | [小周报](progress/WEEKLY_2026-10-25.md) |
| 项目技术描述与 PDF | [RESUME.md](RESUME.md) |
| 早期双硬件微基准及源码追溯 | [RESULTS.md](RESULTS.md)、[证据索引](artifacts/README.md) |
| 主流框架支持标准与后续研究 | [迁移方案](MAINSTREAM_MIGRATION.md)、[研究计划](TRAINING_RESEARCH_PLAN.md) |

| 实现模块 | 职责 |
|---|---|
| `ops.py` | 逐 token 参考、分块 GDN、求解算法与输入检查 |
| `backends.py` | 共同接口、可恢复 Qwen3.5 adapter、直接 compile 与实例缓存 |
| `compiler.py` | 历史领域求解算子的 FX 改写、Inductor／TorchAir 入口；仍使用旧 doubling |
| `cuda_decode.py` | 推理用 Triton 融合 kernel，返回新的 FP32 state |
| `training_probe.py` | 真实层诊断、整模型数值检查和 LoRA 训练 |
| `profile.py` | CPU／CUDA 时间线采集与聚合 |

当前训练使用可微 PyTorch 分块计算；旧语义 custom op 没有训练 autograd，Triton decode 也未提供训练 backward。代码属于独立项目，尚不是 vLLM-Omni 上游贡献。

## 安装和复跑

下列命令从独立 GitHub 仓库根执行。该目录包含 `pyproject.toml`、`deltaforge/` 包和 `tools/`；本地 flat 实验目录的组织略有不同。

```bash
python -m pip install --no-deps -e .
python -m deltaforge.cpu_gate --output ./cpu-gates.json
python tools/report_training.py
```

安装前需已有匹配硬件的 PyTorch。CUDA 实测为 PyTorch 2.9.1／Triton 3.5.1；NPU 历史实验使用匹配的 torch-npu／TorchAir。模型依赖、资产核验和 GPU 命令见 [接入阅读版](TRAINING_INTEGRATION-friendly.md)。14 项 CPU 检查通过不代表 GPU 或真实模型训练已复现。

## 离线性能实验台

打开 [Profiling Lab](profiling-lab/profiling-lab.html) 可查看已有微基准、CPU／CUDA 时间线、模拟滑杆和判断练习，无需后台服务。浏览器只显示数据，不运行模型。

CUDA 教学采集共含 1032 个 kernel；两次 decode 对照均为 PyTorch 15 个、Triton 1 个。这个结果说明本次调用的融合减少了 kernel 数。采集中的同步会改变时间线，不能用其阶段时长代替正式延迟测量。学习方法见 [观测阅读版](PROFILING_GUIDE-friendly.md)。

## 同时签出 main 与 doc

main 保存代码和结果，doc 另有手稿、原始证据与历史记录。Git worktree 允许同时打开两个分支：

```bash
mkdir deltaforge-workspace
git clone --branch main \
  https://github.com/CheongRoach/deltaforge.git \
  deltaforge-workspace/main
git -C deltaforge-workspace/main fetch origin main doc
git -C deltaforge-workspace/main worktree add \
  -b doc ../doc origin/doc
```

得到：

```text
deltaforge-workspace/
├── main/
└── doc/
```

两个目录共享 Git 对象库，但可以同时签出、编辑和提交。已有本地 `doc` 分支时，去掉 `-b doc` 并把 `origin/doc` 改为 `doc`。日常同步用：

```bash
git -C deltaforge-workspace/main pull --ff-only origin main
git -C deltaforge-workspace/doc pull --ff-only origin doc
git -C deltaforge-workspace/main worktree list
```

公开仓库省略组织专用 worker、平台响应和签名 URL，保留来源 hash；详见 [PUBLIC_EXPORT.md](PUBLIC_EXPORT.md)。实验过程有 AI 辅助，个人贡献表达及独立练习见 [STAR](STAR.md)。
