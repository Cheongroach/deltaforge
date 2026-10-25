# DeltaForge

可恢复的 Qwen3.5 Gated DeltaNet 后端、异构编译实验和离线 Profiling Lab。固定 H20／Transformers 5.2.0 的 LoRA 路径已验证；主流框架采用、生产 SLA 和全参数训练尚无证据。

## 同时签出 main / doc

推荐使用 Git worktree，把一个仓库组织成“母目录 + main/doc 两个子工作区”。两个目录共享 Git 对象库，但可以同时签出、编辑和提交不同分支：

```text
deltaforge-workspace/
├── main/  # 代码、Profiling Lab、可复跑实验
└── doc/   # 手稿、日志、原始证据、对话轨迹
```

首次建立：

```bash
mkdir deltaforge-workspace
git clone --branch main \
  https://github.com/CheongRoach/deltaforge.git \
  deltaforge-workspace/main

git -C deltaforge-workspace/main fetch origin main doc
git -C deltaforge-workspace/main worktree add \
  -b doc ../doc origin/doc
```

如果已经存在本地 `doc` 分支，将最后一条改为：

```bash
git -C deltaforge-workspace/main worktree add \
  ../doc doc
```

日常同步与查看：

```bash
git -C deltaforge-workspace/main pull --ff-only origin main
git -C deltaforge-workspace/doc pull --ff-only origin doc
git -C deltaforge-workspace/main worktree list
```

删除 doc 工作区不会删除远端分支：

```bash
git -C deltaforge-workspace/main worktree remove \
  ../doc
git -C deltaforge-workspace/main worktree prune
```

也可以把 `doc` 放在 `main/doc` 内形成严格的父子目录，但它会出现在 main 工作区的未跟踪路径中，需要额外维护 `.git/info/exclude`。因此默认使用上面的并列子工作区形式。

## 训练后端与本轮进展

- [接入与复跑](TRAINING_INTEGRATION.md)：安装、资产 hash 核验、可恢复 adapter、模型门禁与 LoRA。
- [训练结果与支持矩阵](TRAINING_RESULTS.md)：14 项 CPU 回归，torch／DeltaForge／compiled 各两次十步 LoRA；compiled 完整步中位数相对 torch 为 2.02–2.20×。FLA 的随机算子明显更快，真实模型梯度分歧保留。
- [小周报](progress/WEEKLY_2026-10-25.md)：timeline、每个任务的动作、问题、难处、方案取舍和后续计划。
- [主流方向迁移方案](MAINSTREAM_MIGRATION.md)：M1 范围内完成，M2 对 torch oracle 的固定 LoRA 子范围通过；原生默认比较未闭合，总体部分完成。

```bash
python -m pip install --no-deps -e .
# 在匹配 PyTorch / Triton 的环境中安装可选依赖；版本见接入说明。
python -m pip install -e ".[qwen,fla]"
python -m deltaforge.cpu_gate --output ./cpu-gates.json
python tools/report_training.py
```

`GDN` 默认使用稳定 block；旧 doubling 和 `compiler.py` 的语义 lowering 仍仅供历史实验，不能作为新训练支持。两次独立进程共享磁盘缓存，十步数据不证明业务质量。

## 现在就能玩

下载 Code → Download ZIP，解压后打开 `profiling-lab/profiling-lab.html`。不需要联网或后台服务。包含真实微基准、CPU/CUDA 时间线、明确标为模拟的参数滑杆、判断题、导入 Chrome trace 和导出练习记录。

建议依次完成：**02 找到 NPU 编译回归 → 03 对比 decode kernel → 04 模拟融合 → 05 判断练习 → 06 自己采集**。

## 已验证与边界

- 历史算子快照：H20 / Inductor prefill 相对同 chunk FP32 eager 为 1.45–1.52×。910B4 / TorchAir 为 0.81–1.46×，T=512 建议保留 eager。
- H20 Triton decode 相对自实现 PyTorch 为约 5.00–5.20×；不是 FLA 或 vLLM 的性能优势证据。
- 另采 CPU/CUDA profile，CUDA 窗口内有 1032 个 kernel；两次 decode 对照均为 PyTorch 15 个、Triton 1 个。profile 干预后的阶段时间不替代无采集基准。
- 数值、梯度、padding、GQA 与 state handoff 已做算子验证。固定 Qwen3.5 LoRA 已接入，原生默认／FLA 梯度门禁仍失败；未完成全参数训练、原生 AscendC backward 或整模型 vLLM 对比。

## 复跑

从仓库根目录执行。CPU 需已有 PyTorch≥2.7。CUDA 实采 PyTorch 2.9.1、Triton 3.5.1、H20 / SM90；NPU 需厂商匹配的 PyTorch、torch-npu、TorchAir 与 CANN。不要用 CUDA wheel 覆盖 NPU 依赖。

```bash
python -m deltaforge.validation
python -m deltaforge.profile --device cpu --output ./my-profile-cpu --iterations 2
python -m deltaforge.profile --device cuda --output ./my-profile-cuda --iterations 2
python -m deltaforge.benchmark --device cuda --output ./my-benchmark
```

采集产生 `summary.json` / `trace.json` / `trace.json.gz`。在 Lab 的 03 页导入 `trace.json`；数据仅在本地处理。刷新打包数据：

```bash
python tools/build_lab_data.py --cpu my-profile-cpu/summary.json --cuda my-profile-cuda/summary.json
```

## 代码与资料

- `deltaforge/ops.py`：独立 recurrence、chunk GDN、领域语义求解与输入契约。
- `deltaforge/compiler.py`：历史语义 solve／幂零展开、静态常量提升、Inductor/TorchAir；不纳入新训练路径。
- `deltaforge/backends.py`：共同 GDN 契约、stable block／direct compiled 与可恢复 Qwen3.5 adapter。
- `deltaforge/cuda_decode.py`：函数式 FP32 state 的 Triton 融合 decode。
- `deltaforge/profile.py`：真实 CPU/CUDA 采集、时间区间并集与算子表。
- [性能实测](RESULTS.md) · [实操手册](PROFILING_GUIDE.md) · [训练与研究路线](TRAINING_RESEARCH_PLAN.md)。
- [doc 分支](https://github.com/Cheongroach/deltaforge/tree/doc)：设计手稿、模型调研、简历、原始事件、源码快照、profile 和对话轨迹。

内部 Merlin 绑定、凭据、签名 URL 不在公开仓库。实采源码 hash 保留作为 provenance；公开快照省略了组织专用的 `platform.py`，详见 doc 分支 PUBLIC_EXPORT.md。

## 下一轮

先闭合原生默认／FLA 的真实梯度分歧，再以更长预算验证冷／热编译缓存、bucket 和重算。后续扩展 packed／第二框架与推理 state slots，验收条件见小周报。会议方向可以考虑系统研究，但当前结果不足以宣称顶会级新贡献；按 [研究路线](TRAINING_RESEARCH_PLAN.md) 的可证伪标准推进。
