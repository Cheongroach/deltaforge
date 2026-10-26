# AI Infra 项目简历

目标方向：训练后端、算子开发与 AI 编译。2026.10 · 独立实践项目，AI 辅助实现、调研与实验。

## DeltaForge：Qwen3.5 GDN 后端适配与数值稳定性

技术栈：Python、PyTorch、Transformers、PEFT、Inductor、FX、Triton、TorchAir；H20 与 Ascend 910B4。

- 负责将独立 GDN 算子接入 Qwen3.5-2B，建立可恢复的层级 adapter，统一 normalization、scale、GQA 与 FP32 functional state，验证跨段梯度和非重入 checkpoint。
- 针对随机测试通过、真实模型仍出现约 32% logits 误差的问题，固定首／中／末层输入与上游梯度，结合共线 key 反例定位旧幂零展开的浮点抵消；比选后采用可微分块三角逆，第 22 层输出 relative L2 从 16.5455 降至 3.34325e-5。
- 完成 14 项 CPU 回归及 torch／block／compiled 各两次十步 H20 LoRA；compiled 参数梯度 relative L2 为 2.69%，通过原 8% 门槛。固定模型、公开数据、依赖与源码 hash，交付安装、门禁和证据汇总工具。
- 保留旧双硬件图编译与 Triton decode 实验；修复 NPU device 构造与 eye converter 缺失，记录 T=512 编译负收益和首次成本，按形状保留 eager 候选。

## 真实模型结果与范围

| 指标 | 观察结果 |
|---|---|
| torch 参考完整 step 中位数 | 837.78／822.58 ms |
| compiled 完整 step 中位数 | 380.44／406.48 ms；同重复性能比 2.02–2.20× |
| 最终 validation loss 最大绝对差 | 对 torch 约 0.001605；预设上限 0.05 |
| 整模型 allocated 峰值 | 均约 4.516 GiB；无整模型显存节省证据 |

条件：单 H20、Qwen3.5-2B、BF16、LoRA rank 8、8,409,600 个可训练参数、batch 1、256-token bucket。每进程十次更新，二次预热、八次计时；两个同 seed 进程共享磁盘缓存。完整 step 包含前向、损失、反向、裁剪、AdamW 和同步，保留原始长尾。

FLA 在随机算子对照中更快，原生默认／FLA 的真实模型梯度尚未通过本轮门槛，未安排其训练。因此当前收益只相对显式 torch 参考；短程训练不证明业务质量、长期收敛或主流采用。新 block 的 NPU 训练、全参数、多卡与 packed 尚未验收。

## 证据与投递使用

材料：[STAR 讲解](STAR-friendly.md)、[训练结果](TRAINING_RESULTS.md)、[历史算子结果](RESULTS.md)、[PDF](RESUME.pdf)。本页可并入个人简历，未补造个人经历；技术职责应与本人实际参与及掌握程度一致。

独立准备时，推导一次状态更新，复现浮点反例，再修改一个 kernel 配置并解释数值与耗时结果。公式与实现入口见 [DESIGN.md](DESIGN.md) 和 [STAR.md](STAR.md)。
