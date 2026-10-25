# DeltaForge

Gated DeltaNet 算子与异构图编译实验，以及可以离线操作的 Profiling Lab。AI 辅助原型、调研与实机实验；不是生产推理／训练框架。

## 现在就能玩

下载 Code → Download ZIP，解压后打开 `profiling-lab/profiling-lab.html`。不需要联网或后台服务。包含真实微基准、CPU/CUDA 时间线、明确标为模拟的参数滑杆、判断题、导入 Chrome trace 和导出练习记录。

建议依次完成：**02 找到 NPU 编译回归 → 03 对比 decode kernel → 04 模拟融合 → 05 判断练习 → 06 自己采集**。

## 已验证与边界

- H20 / Inductor prefill 相对同 chunk FP32 eager 为 1.45–1.52×。910B4 / TorchAir 为 0.81–1.46×，T=512 建议保留 eager。
- H20 Triton decode 相对自实现 PyTorch 为约 5.00–5.20×；不是 FLA 或 vLLM 的性能优势证据。
- 另采 CPU/CUDA profile，CUDA 窗口内有 1032 个 kernel；两次 decode 对照均为 PyTorch 15 个、Triton 1 个。profile 干预后的阶段时间不替代无采集基准。
- 数值、梯度、padding、GQA 与 state handoff 已做算子验证。未完成完整模型训练、原生 AscendC backward 或整模型 vLLM 对比。

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
- `deltaforge/compiler.py`：torch.export/FX 幂零展开、静态常量提升、Inductor/TorchAir。
- `deltaforge/cuda_decode.py`：函数式 FP32 state 的 Triton 融合 decode。
- `deltaforge/profile.py`：真实 CPU/CUDA 采集、时间区间并集与算子表。
- [性能实测](RESULTS.md) · [实操手册](PROFILING_GUIDE.md) · [训练与研究路线](TRAINING_RESEARCH_PLAN.md)。
- [doc 分支](https://github.com/Cheongroach/deltaforge/tree/doc)：设计手稿、模型调研、简历、原始事件、源码快照、profile 和对话轨迹。

内部 Merlin 绑定、凭据、签名 URL 不在公开仓库。实采源码 hash 保留作为 provenance；公开快照省略了组织专用的 `platform.py`，详见 doc 分支 PUBLIC_EXPORT.md。

## 下一轮

先建立锁版本 FLA/FlashQLA 基线、解析 VJP／激活保存与重算，再接入真实模型训练步与训练框架。会议方向可以考虑系统研究，但当前结果不足以宣称顶会级新贡献；按 [研究路线](TRAINING_RESEARCH_PLAN.md) 的可证伪标准推进。
