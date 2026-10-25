# DeltaForge

Gated DeltaNet 算子与异构编译实验项目，面向 AI Infra / 算子开发 / AI 编译岗位。

- [STAR 项目讲解](STAR.md)：负责人范围、技术决策、实测结果、面试讲述和简历段落。
- [训练接入](TRAINING_INTEGRATION.md) · [终态结果与支持矩阵](TRAINING_RESULTS.md) · [小周报](progress/WEEKLY_2026-10-25.md)。
- [近期模型调研](RESEARCH.md)：Qwen3.8-Flash-Next、DeepSeek-V4.1、FlashQLA 与选题取舍。
- [技术设计与数学手稿](DESIGN.md)：递推 / chunk 等价、幂零求解展开、后端与验收。
- [实测报告](RESULTS.md)：真实 H20 / 910B4，基线、数值、原始样本与失败边界。
- [技术简历](RESUME.md)：按本次完成情况撰写，附 PDF；不补造学历或工作经历。

## 实现入口

| 文件 | 职责 |
|---|---|
| ops.py | 独立 recurrence、chunk prefill、语义 solve、输入契约 |
| compiler.py | torch.export / FX 领域 lowering、Inductor / TorchAir |
| cuda_decode.py | CUDA Triton 融合 decode，函数式 FP32 state |
| validation.py | 数学、梯度、padding、GQA、state handoff、图降低验证 |
| benchmark.py | 双后端计时、配置扫描、编译与数值证据 |
| platform.py | 自包含平台任务、预检、限定单卡、只操作本项目新建任务 |

## 复跑

从包含 deltaforge/ 的父目录执行。CPU 需 PyTorch≥2.7；CUDA 实测环境 PyTorch 2.9.1 / Triton 3.5.1 / SM90；NPU 需匹配的 CANN、torch-npu、TorchAir。不要在 NPU 环境随意用 CUDA wheel 覆盖厂商依赖。

```bash
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python3 -m deltaforge.validation
python3 -m deltaforge.benchmark --device cuda --output /tmp/deltaforge-results-cuda
python3 -m deltaforge.benchmark --device npu --output /tmp/deltaforge-results-npu
```

本地开发机没有可用 GPU。CPU 测试断言通过；早期本地运行曾遇到线程 comm 文件的沙箱限制，设备容器内另完成整套 CPU 回归。完整运行证据见 artifacts/runs/。

benchmark 使用固定种子随机算子夹具，不运行整个模型。prefill 默认 FP32 运算与 state；decode 使用 BF16 Q/K/V＋FP32 state。只对相同形状、精度、初始化 state 的结果比较。

platform.py 固定本次用户授权的 CN / search_douyin_life_nlp / cloudnative-hl 资源，单任务单卡、最多 3600 秒、不自动重启；复跑提交前需要重新查资源并准备当前 CLI schema，先执行 dry-run 与预检。data.status=error 禁止提交；warn/hint 展示后可继续；保留式合并 precheck_sid，并在 30 分钟内提交。已有 receipt 禁止重复创建。当前 submit 直接在入口携带经过 SHA-256 检查的源码归档，启动后自动执行测试；不需要额外 stage。stage 仅保留为兼容入口。原始平台响应、签名日志 URL 和安装缓存只存 /tmp，不放入交付包。

## 结论使用范围

Triton decode 的对照是本项目 PyTorch token recurrence；prefill compiler 对照为同 chunk 的 portable eager。它们不是 FlashQLA / FLA 的领先性证据。固定 Qwen3.5 LoRA 已接入；原生默认比较、原生 AscendC kernel、融合 backward 或线上 SLA 验证未完成。

AI 辅助完成原型、调研和实验。建议独立复跑、推导与修改一个 schedule 后，再在面试中承诺独立掌握相应实现。

## 证据与简历复建

原始事件、源码快照、环境与聚合 CSV / JSON 见 [artifacts/](artifacts/README.md)。从父目录执行：

```bash
python3 deltaforge/tools/report_results.py
python3 deltaforge/tools/write_report.py
python3 deltaforge/tools/build_resume.py --font /path/to/cjk-font.ttf
```

PDF 复建需 reportlab 和支持中文的 TrueType 字体；PDF 已交付，可直接打开。交付 ZIP 包含整个 deltaforge/，解压到空目录后按上述命令复跑。

实机完成后仅改进了 platform.py 的日志 JSON 提取：厂商编译进度可能出现在事件 marker 前，采集器现在从 marker 解码，恢复了 3 条 TorchAir compiled measurement。性能数值未重新生成；实测源码以 snapshots/ 与 worker hash 为准。

## 可操作的观测学习版

打开 [Profiling Lab](../profiling-lab/profiling-lab.html)，无需联网。真实 CPU/CUDA 采集在 H20 容器完成，CUDA 窗口含 1032 个 kernel，两次 decode 对照都为 PyTorch 15 个、Triton 1 个。profile 不是稳定 latency；滑杆明确标为教学模拟。

[实操手册](PROFILING_GUIDE.md) · [训练与研究下一阶段](TRAINING_RESEARCH_PLAN.md) · [对话覆盖说明](../conversations/COVERAGE.json)。

本页旧算子章节保留为历史材料；当前代码与复跑入口见 [仓库 README](../README.md)，新训练报告见 [TRAINING_RESULTS](../TRAINING_RESULTS.md)。历史内部平台 helper 不在公共代码中。
