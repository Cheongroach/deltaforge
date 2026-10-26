# DeltaForge 双硬件实测报告

实测日期：2026-10-25。本报告保留旧 doubling／语义编译路径的历史微基准协议与数值。工作负载为固定 seed 随机算子输入，锚定 Qwen3.5-2B GDN 维度，没有加载完整模型权重。后来的 block 真实训练另见 [TRAINING_RESULTS.md](TRAINING_RESULTS.md)，不沿用这里的 NPU 性能结论。

本轮检验编译覆盖、同 chunk 编译收益与 CUDA decode 融合。先读 [结果阅读版](RESULTS-friendly.md) 可以了解比较对象、回本计算和实际用途；本文保留所有测量表及失败记录。prefill 处理一段 token，decode 处理一个新 token；此处均为算子实验。

## 已完成结果

同一修复版本在 H20 / Inductor 与 910B4 / TorchAir 上各完成 3 组 fullgraph prefill 编译与数值验收；CUDA Triton decode 完成 24 组配置测量、GQA / 非规则维度校验与连续 128 步状态交接。两端最终任务均须同时具备 done、Pod exit=0 和 benchmark_complete，不能用提交、预检成功或仅有终止事件替代编译验收。

所有速度均以同一进程、相同 shape / dtype / state 的对照计算。prefill 对照为同 chunk 的 portable eager；decode 对照为本项目 PyTorch token recurrence。这不是 FLA / FlashQLA 领先性或整模型收益证据。

## 环境与方法

| 后端 | 实采环境 | 资源 |
|---|---|---|
| H20 | NVIDIA H20 / SM90；PyTorch 2.9.1；Triton 3.5.1；TF32 matmul 关闭 | 单卡、20 CPU、230400 MB |
| Ascend | Ascend910B4-1；PyTorch 2.7.1+cpu；torch-npu 2.7.1；TorchAir 可导入 | 单卡、15 CPU、128000 MB |

TorchAir 无可用 distribution version；本轮未收集精确 CANN / driver build，因此当前版本标识并不覆盖整个软件栈。NPU 镜像中的 Triton 包不证明 Triton-Ascend backend 可用。当前 NPU 实作为 PyTorch / TorchAir 图路径。

输入 Q/K 先归一化；g∈[-0.15,0]，β∈[0,1]，初始 state 非零。prefill 运算及 state 为 FP32；decode 输入 Q/K/V 为 BF16、state 为 FP32。每格先数值后计时：4 次预热、7 组重复、每组 5 次调用；表中为 wall 中位数，原始 7 组 wall / device event 样本都保留。event 会包含提交空洞，不能当作纯 kernel compute。

本轮属于同一进程内的微基准配置扫描，候选选择与报告共用这些样本。尚未进行多次独立运行的候选确认或统计显著性检验。

## Prefill：同 chunk 的编译性能比

B=1，K=V=128；Hk=Hv=H；三个形状分别 T=128,H=16 / T=512,H=16 / T=1024,H=4。H=4 是小 head 数夹具，不是已运行 TP 的证据。

| 后端 | T / H | chunk | eager ms | compiled ms | 性能比 | 导出+编译 s | 回本调用数 |
|---|---:|---:|---:|---:|---:|---:|---:|
| H20 / Inductor | 128 / 16 | 64 | 1.0674 | 0.7341 | 1.454x | 6.072 | 18220 |
| H20 / Inductor | 512 / 16 | 64 | 1.8978 | 1.2559 | 1.511x | 4.927 | 7676 |
| H20 / Inductor | 1024 / 4 | 64 | 3.0567 | 2.0123 | 1.519x | 7.429 | 7114 |
| 910B4 / TorchAir | 128 / 16 | 64 | 0.6510 | 0.6201 | 1.050x | 27.129 | 879793 |
| 910B4 / TorchAir | 512 / 16 | 64 | 1.4458 | 1.7825 | 0.811x | 16.382 | 不回本 |
| 910B4 / TorchAir | 1024 / 4 | 64 | 2.0627 | 1.4115 | 1.461x | 19.704 | 30259 |

910B4 的 T=512 编译延迟高于 eager；dispatch-plan.json 对该形状建议 eager。T=128 的收益仅约 5%，须独立确认后再采用。这是离线配置建议，尚未实现在线 dispatch。

编译列是 export/lower 加上首次编译与执行，不是稳定 cache miss 均值。回本调用数=ceil(该成本×1000 / 每调用节省毫秒)，为本轮估算；未实现在线编译 cache。形状改变会重新特化，不能无条件复用本次收益。

原生 triangular solve 另保留测量，但其 GPU / NPU 路径、融合和 dtype 策略不同。不能只选慢原生 solve 的巨大比值来代表相对成熟 GDN kernel 的收益。

## Decode：CUDA Triton 融合与配置扫描

T=1，Hk=Hv=16，K=V=128。候选 tile=8/16/32/64，warps=4/8；每个 B 测量 8 组候选。表中包括函数调用和输出/state 分配。

| B | tile / warps | PyTorch us | Triton us | 加速 |
|---:|---:|---:|---:|---:|
| 1 | 16 / 4 | 211.58 | 42.33 | 4.999x |
| 4 | 8 / 8 | 212.09 | 40.79 | 5.199x |
| 16 | 16 / 4 | 212.54 | 42.37 | 5.016x |

kernel 将 decay、KᵀS、误差、rank-1 更新和 QᵀS 融合，state 按 value tile 只读写一次。增大 tile 的并行度与 Q/K 冗余存在取舍。当前没有 nsys / ncu 采样，不能据延迟断言 HBM 饱和、具体 occupancy 或 Tensor Core 使用比例。

## 数值与梯度

两端最终容器均通过 8 项 CPU 回归（含 double gradcheck、完整六输入 autograd、state handoff 与 FX 改写）；设备分别用 FP32/BF16 各验证 7 组小形状/边界夹具。compiled prefill 的独立 oracle 在 CPU 按 token 执行。

FP32 output 门禁 atol=5e-5、rtol=5e-4；BF16 output 为 atol=5e-3、rtol=1e-2；所有 FP32 state 统一 atol=5e-5、rtol=5e-4。混合容差不代表每个测试能覆盖全部输入分布。

| 路径 | output max abs | state max abs |
|---|---:|---:|
| h20-constants compiled prefill（3 形状最大值） | 4.47035e-08 | 4.17233e-07 |
| npu-constants compiled prefill（3 形状最大值） | 4.47035e-08 | 5.21541e-07 |
| Triton 128 步 handoff | 3.05176e-05 | 4.17233e-07 |

连续 handoff 检查 FP32 state 不累计 BF16 截断。非规则 decode 另检查 B=2、Hk=2、Hv=4、K=37、V=53。padding 的无效 token 作为恒等状态转移。纯 PyTorch backward 已验证；语义 custom op 和 Triton kernel 未实现融合训练 backward。

## 编译失败与修复证据

| 轮次 | 观察 | 修复或处置 |
|---|---|---|
| npu-initial | Dynamo Unsupported：generated Python 构造 NPU torch.device | 将 factory device 规范化为字符串 |
| npu-device-fix | BackendCompilerFailed：aten.eye.default ge_converter is not implemented | 将固定 eye / ones 提升为图外 target-device buffer |
| h20 / npu-stage-blocked | 只有设备 probe；execute-script 需要缺失的 WebShell SSO cookie | 停止这两项等待传源码的任务，改为入口携带完整源码 |
| h20 / npu-constants | 实机编译后重新数值验收与计时 | 不把 eager fallback 记作 compiled |

FX 只匹配领域 unit_lower_solve；GDN 构造严格下三角 L，确保 L^C=0。有限倍增是实数域精确改写，浮点误差仍需测量。当前每个 prefill 图降低 1 个 batched domain op，并提升 2 个固定 factory 常量。CPU 小形状示例的 before/after FX 图位于 artifacts/ir-example/；它不是设备生成 kernel / GE IR 的导出。

## 任务闭合与证据索引

| 别名 | 状态 | 完整 benchmark | Pod exit |
|---|---|---|---|
| h20-initial | done | True | [0] |
| npu-initial | done | True | [0] |
| npu-device-fix | done | True | [0] |
| h20-stage-blocked | stopped | False | [-70000] |
| npu-stage-blocked | stopped | False | [-70000] |
| h20-constants | done | True | [0] |
| npu-constants | done | True | [0] |

平台请求先核对 schema / dry-run，再将同一最终 payload 包装到 check-items。data.status=error 时停止；warn/hint 展示后可继续；成功 sid 保留式合并到 request_context.precheck_sid，并在 30 分钟内提交。所有本项目新任务均为单设备、3600 秒硬上限、不重启。已停止的只限本项目等待传输任务。

run-index.json 仅保留实验别名、终态、退出码与资源信息；原始平台响应和签名 URL 留在私有临时目录。每个真实 worker 的 source_sha256 与启动归档逐文件核对。首轮 H20 / NPU 和后续修复轮次的源码都保留，避免把不同代码版本混成一次实验。

文件：[原始事件与快照](artifacts/README.md) · [聚合 JSON](artifacts/measurements.json) · [CSV](artifacts/measurements.csv) · [比较与回本 JSON](artifacts/summary.json)。

## 实用边界与下一步

这轮闭合了固定形状的双硬件算子／编译实验。后续已采 CPU／CUDA profile，并以新 block 完成固定 H20 Qwen3.5 LoRA 与 FLA 随机算子对照，分别见 [观测手册](PROFILING_GUIDE.md) 和 [训练结果](TRAINING_RESULTS.md)。

新算法的 NPU 训练、原生 AscendC、多卡、packed serving、完整模型推理与融合 backward 仍未验收。这里的旧算法速度不能直接写成模型训练或服务吞吐收益。
