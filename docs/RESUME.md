# AI Infra 技术简历

目标方向：算子开发 / AI 编译 / 异构推理

## DeltaForge：GDN 异构算子编译与调优

2026.10 · 独立实践项目（AI 辅助原型、调研与实验）

技术栈：Python、PyTorch、torch.export / FX、Inductor、Triton、TorchAir、H20、Ascend 910B4。

- **问题与方案**：面向 Qwen 混合注意力中的 Gated DeltaNet，设计从数学递推到领域 IR、图编译和融合算子的验证链路。采用可在单 H20 与单 910B4 复跑的真实模型维度，隔离后端差异与编译成本。
- **领域编译**：实现 unit_lower_solve 语义算子与 FakeTensor 形状推断；利用严格下三角矩阵 L 的幂零性，将 (I+L) 的求解降低为有限倍增矩阵乘，执行 torch.export / FX 改写并对接 Inductor、TorchAir 的 fullgraph 编译。仅改写满足领域契约的算子。
- **融合与调优**：编写 Triton decode kernel，融合衰减、状态读出、rank-1 更新与输出投影；采用 FP32 函数式状态，扫描 value tile 与 warp 数。prefill 拼接 U/W 的 RHS 共用求解，扫描 chunk=16/32/64，生成按形状的离线候选建议，识别 910B4 T=512 编译回归并保留 eager。
- **数值与工程**：建立独立 token oracle、双精度 gradcheck、完整 autograd 对照、GQA、K≠V、padding 和状态交接测试；修复 NPU generated device 与缺失 eye converter，通过静态常量提升闭合编译路径。记录源码 SHA-256、后端环境、原始计时样本和失败案例。

### 实测结果

| 路径 | 结果 | 对照与边界 |
|---|---:|---|
| H20 / Inductor | 1.45-1.52x | FP32 prefill；同 chunk eager 对照 |
| 910B4 / TorchAir | 0.81-1.46x | prefill 性能比；T=512 建议保留 eager |
| H20 / Triton | 40.8-42.4 us | decode；自实现 PyTorch 对照 5.00-5.20x |

条件：prefill 为 B=1、T=128/512/1024、H=16/16/4、K=V=128；decode 为 B=1/4/16、T=1、H=16、BF16 输入与 FP32 state。每格 4 次预热、7 组重复、每组 5 次调用，表中为本轮配置扫描候选的 wall 中位数。

验收：两端各完成 8 项 CPU 回归、14 组设备数值检查和 3 组完整图编译验收；CUDA 另验证非规则维度与连续 128 步 state handoff。详细误差与计时见 [RESULTS.md](RESULTS.md)。

基线为自实现 PyTorch 与同 chunk eager；未验证领先 FLA / FlashQLA、完整模型 TTFT / TPOT、原生 AscendC kernel 或融合 backward。

项目材料：[设计手稿](DESIGN.md) · [近期模型调研](RESEARCH.md) · [原始证据](artifacts/README.md) · [PDF 简历](RESUME.pdf)

## 面试准备与投递使用

这是项目版简历，未添加未提供的姓名、联系方式、学历、任职经历。正式投递时可将项目段落并入个人简历。保持 AI 辅助实践标注，并先独立完成以下练习：

1. 从 token recurrence 推导 U/W/R 与 chunk state 更新，解释 prefix 差如何避免 underflow。
2. 独立修改一个 Triton schedule，复跑数值检查与计时，说明 tile 并行度、访存重复与寄存器压力的取舍。
3. 说明有限倍增为何只对严格下三角成立；用真实编译成本计算回本次数，并解释图编译与原生 AscendC 算子的区别。
4. 给出从算子到模型的验收计划：逐层误差、logits KL、自然生成质量、TTFT / TPOT、显存与多请求状态隔离。
