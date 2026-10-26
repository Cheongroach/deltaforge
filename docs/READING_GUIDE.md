# DeltaForge 阅读入口

DeltaForge 研究的是模型内部一段计算如何稳定地执行、接入训练框架并被编译加速。你可以先把它理解为“给模型更换一个计算部件，并检查更换后是否还能正确训练”。

目前得到的结果是：在固定的 Qwen3.5-2B、单张 H20、短程 LoRA 训练中，分块求解和编译版本通过了预设数值检查。LoRA 只训练附加的小矩阵。编译版本的完整训练步中位数相对本项目 torch 参考缩短到约一半。成熟 FLA 内核在随机算子测试中更快，但它与原生默认路径的真实模型梯度比较尚未通过本轮门槛，原因仍需定位。

## 按你现在想理解的问题阅读

| 想知道什么 | 先读 | 需要核对细节时 |
|---|---|---|
| 项目到底做了什么，我应怎样讲？ | [STAR-friendly](STAR-friendly.md) | [STAR 技术版](STAR.md) |
| GDN、递推、分块求解是什么意思？ | [DESIGN-friendly](DESIGN-friendly.md) | [数学与实现设计](DESIGN.md) |
| 实验能证明什么，不能证明什么？ | [TRAINING_RESULTS-friendly](TRAINING_RESULTS-friendly.md) | [配置、误差和完整表格](TRAINING_RESULTS.md) |
| 怎样自己安装、检查和训练？ | [TRAINING_INTEGRATION-friendly](TRAINING_INTEGRATION-friendly.md) | [接口契约与命令](TRAINING_INTEGRATION.md) |
| 怎样看性能时间线、找优化线索？ | [PROFILING_GUIDE-friendly](PROFILING_GUIDE-friendly.md) | [采集协议与指标定义](PROFILING_GUIDE.md) |
| 早期 H20／910B4 算子实验做了什么？ | [RESULTS-friendly](RESULTS-friendly.md) | [历史微基准完整报告](RESULTS.md) |

推荐第一次按“STAR → DESIGN → 训练结果”读，理解之后再动手复跑。每篇阅读版都保留结论所需的条件；公式、API 和完整计时表放在对应技术版中。

## 不需要双版本的材料

[模型调研](RESEARCH.md) 解释为什么选择 GDN；[框架迁移](MAINSTREAM_MIGRATION.md) 解释怎样才算可用；[研究计划](TRAINING_RESEARCH_PLAN.md) 给出下一步假设与否定条件。这三篇直接用原名更新。

[小周报](progress/WEEKLY_2026-10-25.md) 按小任务记录问题、难处、候选和决定，后半部分保留时间表。[项目简历](RESUME.md) 和 [PDF](RESUME.pdf) 用于投递材料，阅读时可以从 STAR 版回查技术依据。

[证据索引](artifacts/README.md) 说明每份 JSON、CSV 和源码快照的用途；[公开导出说明](PUBLIC_EXPORT.md) 说明公开包的范围；[对话索引](conversations/README.md) 解释原始记录怎样使用。原始记录保留原样，避免编辑文字时改变实验历史。

## 先认清三个容易混淆的比较

“算子”是模型中的一段计算；“完整训练步”还包括模型其他层、损失、反向传播和参数更新；“业务质量”要靠足够的训练预算和评估数据判断。三者需要各自的证据。

早期双硬件结果来自旧算法的随机输入算子实验。后来的训练结果来自修复后的分块算法、真实模型权重和短程 LoRA。旧数据不能作为新算法在 NPU 上训练的证明。

文档整理日期为 2026-10-26，实验日期为 2026-10-25。本次重绘和整理使用已有终态证据，没有新增 GPU 实验。
