# 早期双硬件算子实验怎样理解

这份报告对应 2026-10-25 的早期算法快照：输入是固定随机数，只执行 GDN 算子，没有加载完整模型。后来的真实模型训练使用修复后的 block 算法，见 [训练结果阅读版](TRAINING_RESULTS-friendly.md)。两轮结果不能合并成同一算法的跨硬件训练收益。

本轮的问题是：同一段满足数学契约的计算，能否在 H20 和 Ascend 910B4 上编译运行？编译后是否值得使用？CUDA 上融合一次 decode 能减少多少调用成本？

## 先把三个词对应到工作

prefill 是处理一段已有 token；decode 是继续生成时处理一个新 token。这里两者都只是算子输入形状，不代表已经测了完整服务。

eager 是框架逐步执行算子，compiled 是编译器处理后的计算图。本轮 prefill 比较相同 chunk 的 eager 和 compiled；decode 比较本项目逐 token PyTorch 实现与 Triton 融合实现。

## 编译成功以后，也可能更慢

两个设备各有三个形状完成完整图编译与数值检查。性能比定义为 `eager 时间 / compiled 时间`；大于 1 表示 compiled 更快，小于 1 表示更慢。

| 设备 | 三个形状的性能比 | 怎样使用这条结果 |
|---|---:|---|
| H20／Inductor | 1.45–1.52× | 本轮同 chunk FP32 对照观察到收益 |
| 910B4／TorchAir | 0.81–1.46× | 收益依赖形状，T=512 应保留 eager |

例如 910B4 的 T=512：eager 约 1.446 ms，compiled 约 1.783 ms。编译能完成与是否值得调用是两个问题。项目保留这个慢形状，生成离线建议，没有实现自动在线调度。

## 每次只省一点，首次编译怎样回本

首次导出、编译和执行需要额外时间。粗略回本次数为：

$$
N=\left\lceil\frac{\text{首次额外成本}}{\text{每次调用节省}}\right\rceil.
$$

分子和分母必须使用相同时间单位；每次没有节省时，就不会回本。

910B4 的 T=128 每次约省 0.0309 ms，记录的首次成本为 27.129 秒，按原始未四舍五入数据估算需 879,793 次调用，约 88 万次。这个数说明小收益可能难以摊销，尚不是冷缓存平均成本或生产部署预测。

## Decode 融合为什么值得试

一次 GDN 更新涉及衰减、读取 state、计算修正、写回 state 和读取输出。Triton 融合把这些动作放进一个设备 kernel，减少独立提交和状态重复读写。

本轮 B=1／4／16、T=1 的候选约为 40.8–42.4 μs，自实现 PyTorch 约为 212 μs，观测比为 5.00–5.20×。候选来自同一轮 tile／warps 扫描；还需要独立重复确认最佳配置。这个比值的参照是本项目 PyTorch 实现，不能直接描述为 vLLM 或 FLA 加速。

后续教学 profile 的两个 decode 对照都记录到 PyTorch 15 个 kernel、Triton 1 个。它支持本次融合减少发射数量，不能仅凭数量判断显存带宽已满或 GPU 计算单元利用率。读时间线见 [观测阅读版](PROFILING_GUIDE-friendly.md)。

## 怎样确认不是“编译失败后悄悄回退”

验收同时检查完整图编译事件、输出／state 数值、原始计时、任务 `done` 和 Pod exit 0。只有任务成功结束不能证明每条 compiled 路径都成功。

910B4 先遇到 generated Python 构造 NPU device 失败，再遇到 `eye` converter 缺失。修复分别是规范 device 表达、把固定矩阵提升到图外。修复后重新在真实设备做编译与数值检查，旧错误和源码快照保留。

本轮没有实现原生 AscendC kernel；软件栈也未收齐精确 CANN／driver build。两端 CPU 配额和软件不同，不能把 wall 时间比直接归因为芯片差异。

## 原始结果放在哪里

[RESULTS.md](RESULTS.md) 保留完整形状、容差、计时和失败表；[artifacts/README.md](artifacts/README.md) 解释原始 JSON、CSV 与源码指纹。新 block 的真实训练、FLA 对照及剩余问题需要查看后来的训练报告。
