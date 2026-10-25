# DeltaForge：GDN 训练后端迁移的 STAR 项目讲解

2026-10-25 · 独立实践项目，AI 辅助实现、调研与实验。

本轮负责把 Gated DeltaNet 算子实验接入真实 Qwen3.5 训练路径，完成数值诊断、稳定求解、编译与完整 LoRA 步验证。STAR 按“背景是什么 → 我负责解决什么 → 如何判断与取舍 → 得到了什么结果”组织；重点在技术决策及其证据。

## S — Situation：为什么要做

DeltaForge 最初已有独立 recurrence、chunk GDN、图编译和 Triton decode 实验，但性能主要对照自实现 PyTorch，缺少真实模型训练和成熟后端的共同契约比较。这不足以证明它能成为主流框架可用的训练后端。

迁移到 Qwen3.5-2B 后，随机算子验证没有暴露的问题出现了：第二轮 DeltaForge／compiled 的整模型 logits relative L2 约 32%，无法进入训练验收。原来的有限幂零展开虽然在实数域精确，浮点稳定性却没有得到真实输入的保证。

真正的问题是：既要让后端接入实际 forward/backward，又要建立能发现数值故障、区分基线和保留失败证据的验证流程。

## T — Task：我负责什么，怎样算完成

以项目负责人视角，本轮职责覆盖四个环节：

1. **框架接入：** 固定 Transformers 5.2.0 的 Qwen3.5 GDN 接口，只替换 GDN callable；统一 normalization、scale、GQA、FP32 functional state 和跨 segment 梯度，保持模型参数、卷积、投影、full attention 和优化器由框架管理。
2. **正确性：** 建立 CPU 契约、double gradcheck、真实层固定输入／上游梯度、整模型 logits／全部 trainable 参数梯度的分层门禁。
3. **性能：** 在单 H20、同初始化与数据顺序下比较 torch、DeltaForge、direct compiled、FLA；完整 step 包含 forward、loss、backward、clip、AdamW 和同步。
4. **复现：** 固定模型、公开数据、依赖和源码 hash，保存三轮失败／成功终态，提供安装、复跑与证据汇总入口。

预先固定的模型门禁为 logits relative L2≤3%、参数梯度聚合 relative L2≤8%、loss abs≤0.05；最终 validation loss 对 torch abs≤0.05。失败后端跳过训练，阈值保持不变。

本轮先选 BF16、LoRA rank 8、batch 1、256-token bucket。它能验证完整训练链路，资源成本较低；全参数训练、packed、多卡和第二训练框架属于后续范围。

## A — Action：怎样判断，做了什么，舍弃了什么

### 1. 先固定契约，再替换后端

采用可恢复的层级 context manager，退出时恢复原 callable；明确拒绝 packed 和未知选项。统一 Q/K normalization 与 scale，避免重复归一化或重复缩放。固定 fla-core 0.4.1，保留原生默认 callable 和共同 normalization 的 FLA adapter 两种比较对象。

选择可微 PyTorch 组合运算作为训练路径，直接 `torch.compile(GDN)` 验证 forward/backward。旧 `compiler.py` 的 semantic solve lowering 仍为 doubling，且没有 custom-op autograd，未纳入新真实模型训练支持。

### 2. 用真实层输入隔离故障

第一轮默认路径对 torch 的梯度误差超过门禁；第二轮改为逐后端落盘，避免一个失败遮蔽其他候选。

从模型第 0／12／22 层截取真实 Q/K/V/g/beta，同时固定对应上游梯度，分别比较 torch chunk32／64、旧 doubling、新 block、直接 triangular、独立 recurrence 和 FLA。这样可以区分局部 kernel 差异与前层误差传播。

另构造近共线 key、弱 decay 的反例。对于 `L=0.999×tril(ones,-1)`，C=32 的旧逆展开 relative L2=4.57553；C=64 最大绝对误差达到 2.57698e10。证据表明，中间矩阵幂的大数抵消会破坏浮点结果，编译本身不能修复算法稳定性。

### 3. 比选求解算法，再恢复训练

旧方法利用幂零性计算有限展开：

$$
(I+L)^{-1}=I-L+L^2-\cdots+(-L)^{C-1}.
$$

新默认使用分块三角逆：

$$
\begin{bmatrix}A&0\\B&D\end{bmatrix}^{-1}
=\begin{bmatrix}A^{-1}&0\\-D^{-1}BA^{-1}&D^{-1}\end{bmatrix}.
$$

它避免形成大矩阵幂，并保留 PyTorch matmul 的 autograd。直接 `solve_triangular` 和独立 recurrence 用于交叉核验；跨厂商编译覆盖与性能需要分别验证。

| 候选 | 判断依据 | 本轮决定 |
|---|---|---|
| doubling | 实数域精确，但已出现浮点反例和真实层大误差 | 保留为实验对照 |
| block | 反例与真实层误差显著降低，可微且可直接编译 | 新默认训练候选 |
| direct triangular | 独立求解可核验 block；编译覆盖与成本仍需实测 | 保留候选／oracle |
| token recurrence | 独立数学参照；逐 token 执行成本较高 | 正确性核验 |
| FLA | 随机算子明显更快，但本轮真实模型梯度门禁失败 | 保留强基线证据，继续定位梯度差异 |

新增共线 key／弱 decay 的 forward/backward 回归，CPU 套件扩展到 14 项；第三轮只为合格后端执行训练。

### 4. 用完整步和终态评价结果

每后端两次独立进程，各十个更新，其中两步 warmup、八步测量；第二次反转后端顺序。保存 372 个 trainable gradient tensor 的完整性、loss、有效 token、step 时间、显存和 adapter hash。

逐文件核验九个模型资产与公开 ESCI 数据，保留源码快照和失败 JSON。worker 不共享开发机用户路径，因此采用自包含 payload；遇到 HTTP 413 后，比选 Base85 与 Base64，最终精简必需源码／依赖，以低于请求限制的 Base64 包提交。

wheel 在隔离 venv、干净工作目录验证，复用基础运行依赖。安装后的 14 项 CPU 检查通过；本地沙箱退出限制与 worker 正常 exit 0 分别记录。

## R — Result：得到什么，边界在哪里

### 正确性与训练结果

| 指标 | 结果 | 比较条件 |
|---|---:|---|
| 第 22 层输出 relative L2 | 旧 doubling 16.5455 → block 3.34325e-5 | 同真实输入、同上游梯度，参照 torch |
| DeltaForge 模型 logits／参数梯度 relative L2 | 2.2011%／2.7236% | 通过原 3%／8% 门禁 |
| compiled 模型 logits／参数梯度 relative L2 | 2.2350%／2.6875% | 通过原 3%／8% 门禁 |
| CPU 回归 | 14 项通过 | worker 正常退出 |
| 真实 LoRA | torch／DeltaForge／compiled 各两次十步 | 单 H20，8,409,600 trainable 参数 |
| 最终 validation loss 对 torch 最大 abs | 约 0.001605 | 低于预设 0.05，四条 validation |

六个训练臂均完成 forward→backward→optimizer，整体 adapter hash 相对初始化发生变化；同后端的两次最终 hash 一致。这证明本轮更新及重复结果，不代表所有参数元素分别变化，也不承诺跨作业位级确定性。

### 完整步性能

| 后端 | 两次 step 中位数 | 相对 torch 同重复中位数 |
|---|---:|---:|
| torch adapter | 837.78／822.58ms | 参考 |
| DeltaForge block | 621.17／635.23ms | 1.29–1.35× |
| direct compiled | 380.44／406.48ms | **2.02–2.20×** |

这是固定 256-token bucket 的 LoRA 观测结果。每次仅八个计时样本，torch 有 2.54s 长尾；进程共享磁盘编译缓存。整模型 allocated 峰值均约 4.516GiB，未观察到整模型显存节省。

FLA 在随机算子基准上明显更快；本轮默认／FLA 的真实模型参数梯度误差为 21.89%／22.40%，未安排训练。该分歧尚未定位，不能推出 FLA 普遍错误，也不能把上述加速比表述为胜过 FLA。

阶段结果是：**M1 的固定契约与随机强基线比较完成；M2 对显式 torch 参考的固定 LoRA 子范围通过，原生默认比较未闭合，因此 M2 总体部分完成。** 主流采用、业务质量、全参数训练和 NPU 新算法训练尚无证据。

## 60–90 秒项目讲述

> 我负责 DeltaForge 从独立 GDN 算子实验向 Qwen3.5 训练后端迁移，范围包括接口、数值验收、稳定求解和完整步性能。
>
> 难点是随机测试通过后，真实模型仍出现约 32% 的 logits 误差。我先固定 normalization、state 和模型版本，再截取首、中、末层输入及上游梯度，结合共线 key 反例，定位到幂零展开的大数抵消。
>
> 比选 doubling、block、直接 triangular 和 recurrence 后，我把默认路径改为可微的分块三角逆，并保持原来的数值门禁。最终 14 项回归通过，DeltaForge 和 compiled 各完成两次十步 H20 LoRA；compiled 的参数梯度误差为 2.69%，完整步中位数从 torch 参考的约 0.82–0.84 秒降到 0.38–0.41 秒，观测加速 2.02–2.20 倍。
>
> 当前价值是固定模型路径的稳定、可编译训练后端与可复核证据。FLA 的随机算子仍更快，原生默认／FLA 梯度分歧尚未闭合，下一步优先解决基线比较和首次编译成本。

## 简历项目段落

**DeltaForge：Qwen3.5 GDN 训练后端适配与稳定性验证**

2026.10 · 独立实践项目，AI 辅助实现与实验

- 负责 Qwen3.5-2B GDN 后端迁移，建立可恢复 Transformers adapter，统一 normalization／scale／GQA／FP32 functional state，支持跨 segment 梯度及非重入 checkpoint。
- 通过真实层固定输入／上游梯度与共线 key 反例定位幂零展开的浮点抵消，采用可微分块三角逆；第 22 层输出 relative L2 从旧算法 16.55 降至 3.34e-5，新增稳定性回归。
- 完成 14 项 CPU 回归及三后端各两次十步 H20 LoRA；compiled 模型参数梯度 relative L2 为 2.69%，完整步中位数相对显式 torch 参考加速 2.02–2.20×。
- 固定模型／公开数据／依赖与源码 hash，交付安装、门禁、训练 probe、原始证据与三轮公共源码快照；保留 FLA 梯度分歧、首次编译成本和未支持范围。

## 面试追问与实现入口

| 追问 | 回答重点 | 实现／证据 |
|---|---|---|
| 为什么数学等价仍会失败？ | 幂零性保证有限展开，不能保证浮点中间量稳定；用反例与真实层交叉验证 | [ops.py](https://github.com/Cheongroach/deltaforge/blob/649210e0ccaa8cfefc3a9d2d0ed88d3ebdababf4/deltaforge/ops.py)、[反例](../training-results/cancellation-counterexample.json) |
| 如何排除模型前层误差？ | 固定局部 Q/K/V/g/beta 和对应上游梯度，比较输出与输入梯度 | [training_probe.py](https://github.com/Cheongroach/deltaforge/blob/649210e0ccaa8cfefc3a9d2d0ed88d3ebdababf4/deltaforge/training_probe.py)、[层诊断](../training-results/attempt-03/evidence/layer-diagnostic/layer-diagnostics.json) |
| 怎样证明 2×？ | 同预算完整 step、两个独立进程、反转次序；比较同重复中位数，披露长尾与共享 cache | [训练汇总](../training-results/summary.json) |
| 为什么不称为比 FLA 更快？ | 算子基准 FLA 更快；真实模型梯度分歧使完整训练比较尚不可验收 | [结果报告](TRAINING_RESULTS.md) |
| 下一步怎样推进主流可用？ | 闭合原生／FLA 梯度、隔离冷／热 cache、长预算复核，再扩展 packed／第二框架／外部复跑 | [迁移方案](MAINSTREAM_MIGRATION.md)、[小周报](progress/WEEKLY_2026-10-25.md) |

数值来源为已完成的第三轮终态及前两轮失败记录；本文整理项目表达，没有新增 GPU 实验。完整配置、原始样本和限制见 [TRAINING_RESULTS.md](TRAINING_RESULTS.md)。
