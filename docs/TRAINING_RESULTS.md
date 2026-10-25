# Qwen3.5 单卡训练结果与支持范围

日期：2026-10-25。第三轮 worker 终态 `done`、exit 0；14 项 CPU 检查和六个独立训练进程全部完成。DeltaForge 的分块求解与直接 `torch.compile(GDN)` 已在固定 Qwen3.5-2B LoRA 配置下通过数值门禁。参照为显式 torch adapter，原生默认／FLA 的本轮梯度门禁仍失败；因此 M2 尚未满足最初的原生默认路径退出条件，也没有“达到主流采用”的证据。

## 配置与判据

- 单张 NVIDIA H20，PyTorch 2.9.1、Triton 3.5.1、Transformers 5.2.0、PEFT 0.18.1、fla-core 0.4.1；BF16 输入、FP32 GDN 累积／state，关闭 TF32。
- 公开 Qwen/Qwen3.5-2B，text-only；九个资产文件和权重加载信息全部核验。固定 wheel 和逐文件来源／revision／SHA256 见 [lock](../training-assets/lock.json)。本轮没有用 FLA 的 models/layers wheel。
- ESCI 公开数据子集，12 train＋4 validation、query 隔离；使用公开文字作 causal-LM 兼容性探针。数据 hash `dcf4a6f54b1225f11a1535ef66648371c8367ccac23702dfd99eaa33286e31e1`。共十个更新只使用前十个训练样本；不作业务质量评估。
- LoRA rank 8，共 8,409,600 可训练参数；AdamW lr=1e-4，非重入 checkpoint，batch 1、256-token bucket，两个 warmup 更新＋八个测量更新。
- 同 seed、初始化、样本顺序和优化器预算，每后端两次独立进程，第二次反转训练后端次序。进程共享设备和磁盘编译缓存；独立进程不等于独立冷缓存。
- 原定门禁不变：model logits relative L2≤3%，全部 trainable 参数梯度的聚合 relative L2≤8%，loss abs≤0.05；kernel output≤5%、每种输入梯度≤8%；最终 validation loss 对 torch abs≤0.05。

## 三轮结果与故障定位

| 轮次 | 结果 | 处理 |
|---|---|---|
| attempt-01 | 默认路径对 torch 梯度误差 22.57%，在后端训练前停止；CPU 13 项正常结束 | 保留失败；增加逐后端和真实层输入诊断 |
| attempt-02 | 仅 torch 合格；DeltaForge／compiled logits 误差约 32%，FLA 梯度约 18%；torch 两次十步训练成功 | 未用随机夹具合格替代真实模型合格；定位幂零展开的大数抵消 |
| attempt-03 | torch／DeltaForge／compiled 合格并各两次训练；default／FLA 不合格；所有安排的子任务 exit 0 | 新默认 block；旧 doubling 留作反例；失败后端跳过训练 |

有限幂零展开在实数域精确，但不保证浮点稳定。反例 `L=0.999×tril(ones,-1)`：C=32 原展开 inverse relative L2=4.57553、max abs=32；block relative L2=1.4028e-8、max abs=5.9605e-8。C=64 原展开 max abs=2.57698e10，block 仍为 5.9605e-8。完整 recurrence 反例与回归见 [cancellation-counterexample.json](../training-results/cancellation-counterexample.json)。

第三轮固定真实 Q/K/V/g/beta **和对应上游梯度**，排除整模型前层误差传播：

| 层 | 旧 doubling 输出 relative L2 | 新 block 输出 relative L2 | block 最大输入梯度 relative L2 |
|---|---:|---:|---:|
| 0 | 3.49122 | 1.96996e-6 | 7.4e-6 |
| 12 | 4.14491e-5 | 2.75823e-5 | 4.5e-5 |
| 22 | 16.5455 | 3.34325e-5 | 2.6e-5 |

同输入的直接 triangular 与独立 recurrence 也接近 torch，支持本次修复与真实模型故障的对应关系。FLA/common 与 native 的局部 g/beta 梯度误差明显高于只替换 normalization；原因仍未闭合。结论限定于本版本、输入、精度与参照，不能推出 FLA 普遍错误。跨轮 torch 的误差／最终 hash 有变化；同轮两个训练重复 hash 一致，本次不承诺跨作业位级确定性。

## 整模型门禁

以下均相对显式 torch adapter（chunk32，共同 BF16 normalization）；默认为当前安装环境的原生 callable，FLA 为共同 normalization 的 adapter。所有 372 个 trainable gradient tensor 都存在且有限。

| 后端 | logits relative L2 | 参数梯度 relative L2 | loss abs | 合格 |
|---|---:|---:|---:|---|
| torch | 参考 | 参考 | 参考 | 是 |
| DeltaForge block | 2.2011% | 2.7236% | 0.001089 | 是 |
| compiled block | 2.2350% | 2.6875% | 0.003372 | 是 |
| native default | 2.7366% | 21.8864% | 0.000919 | 否；未训练 |
| FLA/common | 2.6041% | 22.4042% | 0.007889 | 否；未训练 |

误差为聚合相对范数；合格不代表所有元素一致，也不代表默认／FLA 已作为正确的训练强基线完成验收。保存各参数主要误差贡献者，方便继续定位。

## 完整 LoRA 训练

step 包含 forward→loss→backward→clip→AdamW→CUDA 同步。每次测量八步合计 1552 个有效预测 token，padding 不计入。中位数按八个样本计算；吞吐为总有效 token／总 step 时间，不能用中位数直接换算。

| 后端 | 重复 | step 中位数 ms | step 最大值 ms | 有效 tokens/s | 峰值 allocated GiB | validation 前→后 |
|---|---:|---:|---:|---:|---:|---|
| torch | 0 | 837.78 | 2539.62 | 177.57 | 4.5162 | 3.48992→2.33193 |
| torch | 1 | 822.58 | 1017.39 | 227.71 | 4.5162 | 3.48992→2.33193 |
| DeltaForge | 0 | 621.17 | 838.25 | 291.12 | 4.5162 | 3.49018→2.33033 |
| DeltaForge | 1 | 635.23 | 900.47 | 282.82 | 4.5162 | 3.49018→2.33033 |
| compiled | 0 | 380.44 | 699.27 | 444.98 | 4.5169 | 3.49082→2.33199 |
| compiled | 1 | 406.48 | 856.86 | 419.11 | 4.5162 | 3.49082→2.33199 |

同重复中位数比：DeltaForge 相对 torch **1.29–1.35×**，compiled **2.02–2.20×**；这是固定配置的观测值。torch 重复 0 有 2.54s 长尾，所以 tokens/s 的较大比例不能直接作为稳定的系统加速承诺。峰值 allocated 近似相同，没有整模型显存节省证据。每后端两次 final adapter hash 完全一致，且与共同初始化不同；不同后端的最终 hash 不同。最终 validation loss 对 torch 最大 abs 约 0.001605，低于预设 0.05。

validation 只用于数值兼容性，四条数据、十步更新不足以评价泛化或业务收益。测量开始前已经执行 validation 与两个 warmup 更新；step0 不等于纯冷启动成本。

## 强基线算子比较

随机固定 seed，实际 Qwen GDN 维度 `[B=1,T,H=16,K=128,V=128]`，同 state、共同 normalization、BF16 输入；准备、normalization 和必要拷贝包括在调用计时内。所有后端在 T=256／2048 均通过随机输入的输出、state、六种输入梯度门禁。以下为两个进程中各十个稳态样本的中位数范围，forward/backward 分别统计。

| T | 后端 | forward ms | backward ms | saved tensor MiB |
|---:|---|---:|---:|---:|
| 256 | torch | 5.68–5.70 | 13.80–17.69 | 73.49 |
| 256 | DeltaForge | 3.76–3.81 | 7.79–7.89 | 67.64 |
| 256 | compiled | 2.18–2.28 | 3.41–3.62 | 41.02 |
| 256 | FLA | 0.70–0.74 | 1.24–1.39 | 12.56 |
| 2048 | torch | 22.61–23.50 | 65.73–70.36 | 580.95 |
| 2048 | DeltaForge | 15.41–16.09 | 42.56–45.63 | 534.14 |
| 2048 | compiled | 9.48–11.98 | 13.78–15.67 | 321.13 |
| 2048 | FLA | 0.68–0.69 | 1.23–1.31 | 86.50 |

FLA 明显更快且保存量更低，DeltaForge 没有 CUDA 领先性证据。saved tensor 是 hooks 记录的逻辑字节总和，没有按底层 storage 去重，不等于真实激活峰值。计时不带 hooks，保存量另采。

首次 forward/backward 含 dispatch／编译／运行，不能作为纯 compile 时间：第一次 kernel 进程的 compiled T=256 为 14.28s、T=2048 为 105.99s；第二进程共享磁盘 cache 后为 1.25s／3.11s。FLA 第一次 T=256 为 52.81s，第二次为 78.09ms；之前的真实模型子任务可能已经预热部分 kernel。首次成本和 cache 的影响已保存，尚未得到完全隔离冷缓存的对比。

## 支持矩阵与阶段状态

| 路径／能力 | 当前证据与状态 |
|---|---|
| stable block eager | CPU 契约／double gradcheck／跨 segment state 梯度／随机 GPU 和真实单层／固定 LoRA 完整训练链路验证；默认候选 |
| direct compiled GDN | H20 固定 shape 下 forward＋backward＋checkpoint＋BF16 LoRA 通过；实例 LRU，磁盘 cache 可复用；非全模型 compile |
| torch adapter | 本轮明确参照，两个完整 LoRA 重复通过 |
| FLA 0.4.1 adapter／native | 随机算子合格，真实层／整模型梯度门禁未通过；不安排本轮训练，不自动替换到它 |
| doubling | 数学等价但已有浮点反例；仅实验比选，不用于本轮训练支持 |
| compiler.py semantic solve lowering | 仍为旧 doubling，无 custom-op autograd；原算子历史报告保留，未纳入新真实模型支持 |
| NPU | 历史固定 shape 算子实验存在；新 block、训练 backward 和真实模型尚无 NPU 证据 |
| packed／动态 shape／decode slots／多卡／全参数训练 | 未完成验收；unsupported 明确拒绝或不提供承诺 |
| M1 | 固定版本、共同契约和强基线随机算子对照完成；实际输入基线梯度分歧保留 |
| M2 | 对 torch oracle 的固定 LoRA 子范围闭合；原生默认正确性比较仍未闭合，因此总体部分完成 |
| M3–M5 | 编译基础证据已增加；成本控制、框架扩展、外部复跑／采用尚待推进 |

## 安装验证

0.3.0 wheel 已在隔离 venv 安装，从干净目录确认 import 来自 site-packages；共线 key smoke、安装后的 14 项 CPU 检查和九个模型文件／数据 SHA256 验证通过。基础 PyTorch／Transformers 依赖被复用，不是完全新依赖环境复现。Debian 缺少 ensurepip，采用现有 pip 的 `--python`；本地 CPU 使用显式 `--sandbox-exit`，外层工具 teardown 仍报告 /proc 限制，不能将外层工具退出宣称为正常。worker CPU／训练另有正常 exit 0 证据。记录见 [install-verification.json](../training-results/install-verification.json)。

## 复跑与证据

[接入说明](TRAINING_INTEGRATION.md) 提供安装／model hash 核验／gate／train 命令。从独立仓库根执行：

```bash
python tools/report_training.py
```

该工具从第三轮终态证据重算 [summary.json](../training-results/summary.json)，核对门禁、训练预算、参数变化、数据 hash、同轮重复 hash 与最终 validation loss；不把数值失败视为训练成功。

`training-results/attempt-01..03/` 保留 compact 原始事件与 JSON；doc 分支 `docs/artifacts/training-snapshots/` 另保存各轮公共源码，逐文件核对 worker hash。组织专用 worker／launcher／平台原始日志不公开，省略项的 hash 保留。旧 RESULTS 与 profile 仍对应旧源码快照，不能作为新 block 的性能证据。

各任务的 timeline、失败、难点和方案取舍见 [小周报](progress/WEEKLY_2026-10-25.md)。
