# DeltaForge 向主流框架可用后端迁移

实验状态截至 2026-10-25，文档整理于 2026-10-26。本文回答：怎样把一个能跑的 GDN 算子实验，变成其他开发者可以安装、调用和维护的框架后端？

目前，固定版本接口、稳定分块求解和相对 torch 参考的 H20 LoRA 子范围已通过。原生默认／FLA 梯度分歧未定位，长预算、更多形状和外部复跑仍待完成。当前证据见 [训练结果](TRAINING_RESULTS.md)。

## 目标与范围

把 DeltaForge 从固定形状算子实验，迁移为主流框架可以调用、替换、复跑和维护的 GDN 后端组件。第一阶段使用 Transformers 的实际 Qwen3.5 模型进行单卡训练对照；第二阶段接入与 FLA 兼容的训练路径或 Flame；推理支线再对接 vLLM。

组件负责 GDN 计算、梯度、后端选择与编译缓存。模型框架继续负责模型参数、数据、优化器和训练循环；推理框架负责请求调度、状态槽与缓存生命周期。只对明确支持的模型、版本、shape 和 dtype 提供承诺。

“主流级可用”的建议判据为：外部开发者可以在干净环境安装，在真实框架中开启或关闭该后端，得到正确的完整训练／推理结果，并在公开适用范围内获得可复现的时间、显存或后端覆盖收益。进一步称为“主流采用”，还需要独立使用、维护反馈或上游接受；仓库公开与文档完善本身不证明采用。

下文 adapter 是框架接入层；state 是 GDN 记忆；packed 指把多个样本放入连续 token buffer，并用边界数组隔离它们；bucket 指按长度分组、减少编译形状数；fallback 指明确切回受支持实现。这些能力都需要对应的接口和验收。

## 先判断当前还缺哪一步

| 阶段 | 要回答的问题 | 当前状态 |
|---|---|---|
| M1 契约与基线 | 是否在比较同一计算？ | 固定版本共同契约、随机算子对照完成；实际输入梯度分歧保留 |
| M2 完整单卡训练 | 替换后能否正确更新参数？ | 相对 torch 的固定 LoRA 子范围通过；原生默认条件未满足，总体部分完成 |
| M3 编译与成本 | 首次成本、形状变化和尾延迟是否可控？ | 直接 compiled 有固定配置证据，成本策略未闭合 |
| M4 框架与维护 | 外部开发者能否安装、复跑和恢复？ | 有安装及发布材料，缺完整外部复跑与扩展验收 |
| M5 采用 | 是否实际被外部使用或上游接受？ | 尚无采用证据 |

以下接口核对解释为什么选择当前接入点；后半部分给出每层能力的具体标准。下一轮优先完成 M2 的基线分歧，而不是同时接入所有框架。

## 官方接口核对

| 来源 | 当前代码／文档事实 | 对迁移的影响 |
|---|---|---|
| Transformers Qwen3.5 | GDN 的 chunk 调用具有 initial_state、output_final_state、Q/K normalization 等参数，并通过独立 kernel/fallback 入口组织。[1](#source-1) | 对接 GDN 接口；普通 full attention 的配置不能默认当作 GDN 后端开关 |
| FLA GDN | 接口包含 scale、state_v_first、cu_seqlens、initial_state 与 final_state；实现包含 backward、重算与后端 dispatch。[2](#source-2) | 统一契约后做强基线，不能把 padding 和 packed 序列当作同一个语义 |
| FLA GDN | 本次读取的默认入口具有 torch.compiler.disable；可选 beta 模式支持范围扩展到 [0,2)，而本项目当前契约为 [0,1]。[2](#source-2) | 固定实际基线配置；不支持的功能应明确拒绝或选择支持它的后端，不能静默当作等价 |
| PyTorch custom operators | 要区分 functional/mutable 契约、fake metadata、opcheck 与数值／梯度检查；组合现有 PyTorch 原语时未必需要 custom op。[3](#source-3) | 外部 kernel 需要可靠注册；现有组合路径优先保持简单。opcheck 不代替 gradcheck |
| vLLM Qwen3.5 | 模型使用专门的 QwenGatedDeltaNetAttention，并提供状态 dtype、shape、copy 接口；该文件为推理实现。[4](#source-4) | 不能直接将独立 T=1 kernel 替换整个层；还需适配卷积状态、请求槽和图执行 |
| Flame | 官方 README 描述与 FLA／Transformers 的集成、编译、梯度累积和分布式训练选项，并指定 torchtitan 版本。[5](#source-5) | 可以作为训练扩展候选，但须核验锁定版本实际支持的模型；不能默认支持任意 Qwen 模型或直接复制全量数据配方 |

以上为官方源码／文档事实，不是外部平台性能复现。GitHub main 是移动引用；实施前必须锁 commit。PyTorch 文档部分功能面向较新的框架版本，不能把这些接口默认套用到现有 PyTorch 2.7／2.9 实测环境。

## 三个产品方向的比选

| 方向 | 收益对象 | 工作量与风险 | 建议 |
|---|---|---|---|
| Transformers 训练后端 | 实际模型的 forward/backward、激活与编译适配 | 需要正确梯度、层接口和 checkpoint，单卡就能开始 | 主线第一阶段 |
| FLA 兼容后端／Flame 接入 | 序列模型训练、packed 输入与后端扩展 | 强基线已经有 backward 和重算，必须找到具体缺口 | 统一接口与单卡闭合后推进 |
| vLLM 推理后端 | 实际请求的 TTFT、TPOT、吞吐与状态管理 | 需要 cache slots、变长请求、图捕获与 scheduler 接口 | 推理支线；不替代训练主线 |

推荐形成“CUDA 复用成熟内核＋DeltaForge 针对实际缺口提供候选＋NPU 适配／参考回退”的结构。候选不胜出时采用成熟后端。跨硬件可用性或显存收益也可以形成真实价值，不要求所有 CUDA shape 都击败 FLA。

## 必须补齐的六层能力

### 1. 稳定计算与状态契约

- 输入 shape、dtype、stride、device；Q/K 的 normalization 位置与 epsilon；scale 只施加一次；g 是 raw gate 还是 log decay；beta 支持范围。
- state 的 K×V／V×K 布局、FP32 累积、初始 state 可选、final_state 是否输出、跨 segment 的梯度是否保留。
- 训练的 functional state 与推理的可变 state/out buffer 使用不同、明确的接口。禁止偶尔分配、偶尔原地修改却不声明契约。
- 如果需要布局转换或 contiguous 拷贝，记录开销；不能把调用方的准备成本排除后宣称系统收益。

### 2. 前向、反向与编译组合

梯度检查与编译训练需要分别验证。本轮直接 compiled 已在固定 BF16 LoRA／非重入 checkpoint 上通过；其他 shape、梯度累积和外部 kernel 仍需对应验证。外部 kernel 按版本提供 fake/autograd 注册，执行 opcheck（检查注册契约）；梯度公式另做 double gradcheck 与真实参数梯度对照。

解析 solve VJP、保存中间量与重算是可比选候选。先计量保存 tensor 字节、峰值显存、backward 时间，避免为了“自定义 backward”增加复杂度却降低完整步性能。Triton T=1 decode 暂作为推理路径，不把它算作训练 backward 实现。

### 3. 变长与 packed 边界

当前 lengths/padding 验证没有覆盖 packed 序列。cu_seqlens 指示一条连续 token buffer 中不同样本的边界；state 必须在各样本边界隔离，不能跨文档传递历史。

建议从有限长度 bucket 开始，记录编译次数、cache 命中、padding 浪费与首次成本。缓存键包含代码、backend、硬件、软件版本、shape、dtype、布局与计算选项；固定形状特化不等于动态形状支持。

### 4. 真实框架训练步

先对接完整小模型，复用原模型参数、卷积、投影、门控和 norm，只替换 GDN 计算。对照从单层输出与参数梯度开始，然后检查整模型 logits、loss 和训练。

真实步包括 forward、loss、backward、优化器及必要同步。训练数据需来源、许可、版本、划分与 tokenizer；同一比较使用同一初始化、样本顺序、有效 token、batch 和优化器预算。可以先做 LoRA，但明确 LoRA 结果不能代表全参数训练。

### 5. 有实际作用的性能证据

| 验证范围 | 基线 | 指标 |
|---|---|---|
| 算子 | 锁版本 FLA／FlashQLA、同契约默认后端与 PyTorch oracle | forward/backward、准备与拷贝、保存激活、编译成本、数值 |
| 完整单卡训练 | 原框架默认路径；当前先以显式 torch 作数值参照 | 有效非 padding tokens/s、step 中位数与尾延迟、峰值显存、loss／任务质量 |
| 多卡训练 | 同硬件原框架 DDP/FSDP 或实际支持的方案 | 通信与计算、吞吐、显存、恢复、梯度／质量 |
| vLLM 推理 | 同版本 vLLM 默认 GDN 后端 | 同模型和负载下的 TTFT、TPOT、吞吐、显存、cache hit 与正确状态隔离 |

每个关键结论做独立进程重复与相同预算对照，给出原始样本和变动范围。profiler 用于提出／解释假设，最终延迟另行在无采集干预的运行中测量。慢 shape、编译成本和失败情况都保留。

“不劣化”需要预先确定指标和容忍范围；不能在看见结果后放宽。先选一个真实使用场景获得清晰收益，再扩大支持范围。没有收益时，回退策略或可编译性本身必须被实际用户验证为有价值。

### 6. 安装、发布与长期维护

提供干净环境安装、可选 CUDA／NPU 依赖、版本矩阵、锁定环境与可复跑测试。编译报错、unsupported 与数值错误分开处理；fallback 记录路径和原因，数值错误不得静默吞掉。对无权限 NPU/CUDA 环境提供明确诊断。

CI 按成本分层：CPU 契约／数学、真实设备算子、模型训练集成、性能回归。分布式、变长与 cache 的测试只对已承诺功能要求。建立版本发布、回滚、问题复现模板与支持范围。

上游提交以一个有实证的窄改动为单位，例如某后端编译支持或某个模型路径的优化。维护者接受、独立用户复跑和实际使用反馈应另记录；不能从 adapter 合并推断整个项目成为主流。

## 执行次序与阶段退出条件

| 阶段 | 具体交付 | 退出条件 |
|---|---|---|
| M1 强基线与契约 | 锁版本的 FLA／默认后端对照、adapter 规格与错误模式 | 相同输入、state、精度及准备成本；不支持选项明确处理 |
| M2 单卡完整训练 | 同一个 Qwen3.5 小模型可切换后端，公开数据与原框架训练步 | 层／梯度／logits／loss 检查通过；独立重复的真实步数据 |
| M3 编译与成本控制 | checkpoint、compile、混合精度、bucket/cache 与可用回退 | 独立重复确认时间、显存或覆盖收益超过测量波动；慢 shape 不误选 |
| M4 框架扩展与维护 | 有限版本的训练框架或 vLLM adapter、CI、发布与复跑材料 | 干净环境可用；实际框架状态／恢复测试；外部复跑 |
| M5 实际采用 | 窄上游贡献、独立使用案例与维护记录 | 真实采纳证据；不以 star 数、文档数或内部提交代替 |

首个实施目标为 M1＋M2，而不是同时完成所有框架、所有卡型。双硬件资源用于独立后端验证；多卡阶段先做同厂商 native 分布式，不能把跨厂商 CPU/TCP 同步当作等价替代。

## 学习如何嵌入实现

1. 从 recurrent 的 shape、scale、normalization 和 state 接口开始，写出与目标框架逐项对应的契约表。
2. 通过 opcheck 与梯度对照理解“能导出”“能编译”“能反向传播”的区别。
3. 在真实模型单层上比较输出与参数梯度，再跟踪完整训练步中的 forward/backward／优化器。
4. 修改一项保存激活／重算或 schedule 策略，用同一框架和相同样本复跑。保留反例和 fallback 决策。

现有 Profiling Lab 继续作为观测学习入口。正式迁移优先让用户能在自己的框架中运行、解释并复现结果。

## 来源

<a id="source-1"></a>1. [Transformers Qwen3.5 实现](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)，本次读取于 2026-10-25；该文件自动生成，上游修改应从对应 modular 源进入。

<a id="source-2"></a>2. [FLA GDN chunk 实现](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/fla/ops/gated_delta_rule/chunk.py)；[FLA README](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/README.md)，本次读取于 2026-10-25。

<a id="source-3"></a>3. [PyTorch Custom Python Operators](https://docs.pytorch.org/tutorials/advanced/python_custom_ops.html)，本次读取于 2026-10-25；部分文档功能不属于当前实测版本。

<a id="source-4"></a>4. [vLLM Qwen3.5 实现](https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/model_executor/models/qwen3_5.py)，本次读取于 2026-10-25。

<a id="source-5"></a>5. [Flame README](https://raw.githubusercontent.com/fla-org/flame/main/README.md)，本次读取于 2026-10-25；功能描述仍需在锁定版本和目标模型上实际核验。

本项目已完成情况见 [RESULTS.md](RESULTS.md) 与 [PROFILING_GUIDE.md](PROFILING_GUIDE.md)。本轮已执行部分与限制见 [TRAINING_RESULTS.md](TRAINING_RESULTS.md)，timeline／问题／比选见 [小周报](progress/WEEKLY_2026-10-25.md)。M1 范围内完成，M2 相对 torch oracle 的 LoRA 子范围通过、原生默认退出条件未满足；M3 的直接 compiled 有基础证据，成本控制与 M4/M5 尚待实施。
