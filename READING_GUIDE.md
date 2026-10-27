# DeltaForge 阅读入口

第一次阅读只打开一篇：

[一步步认识 DeltaForge 项目](一步步认识DeltaForge项目-friendly.md)

它参照“一步步认识 TensorRT 项目”的教学方式，把主线拆成 72 个小步骤。每一步只增加一个概念，并在末尾要求你停下来回答。文中默认已经掌握 token、embedding、Q/K/V、PyTorch、GPU、前向传播与反向传播，不重复这些基础知识。

不要同时打开其他 friendly 文档。先按顺序读完一个阶段，回答该阶段的问题；答不出来就停在对应步骤。

## 主教程读完以后

能够闭卷回答主教程最后 12 个问题后，再读：

[DeltaForge 完整开发实战](DeltaForge完整开发实战-friendly.md)

这篇实战稿把认知映射到真实终端命令、代码实现、参数选择、失败修复、CPU／GPU 验证、性能测量、监控落地、构建安装和 Git 发布。2026-10-27 能在当前环境完成的链路已实际复跑；GPU 部分只回放已保存的 H20／910B4 终态证据。

主教程把认知顺序串起来。下面的材料不再承担“从零讲完整项目”的任务，只用于复习某一部分或核对证据：

| 想复习或核对什么 | 阅读版 | 技术版 |
|---|---|---|
| 项目到底做了什么，我应怎样讲？ | [STAR-friendly](STAR-friendly.md) | [STAR 技术版](STAR.md) |
| GDN、递推、分块求解是什么意思？ | [DESIGN-friendly](DESIGN-friendly.md) | [数学与实现设计](DESIGN.md) |
| 实验能证明什么，不能证明什么？ | [TRAINING_RESULTS-friendly](TRAINING_RESULTS-friendly.md) | [配置、误差和完整表格](TRAINING_RESULTS.md) |
| 怎样自己安装、检查和训练？ | [TRAINING_INTEGRATION-friendly](TRAINING_INTEGRATION-friendly.md) | [接口契约与命令](TRAINING_INTEGRATION.md) |
| 怎样看性能时间线、找优化线索？ | [PROFILING_GUIDE-friendly](PROFILING_GUIDE-friendly.md) | [采集协议与指标定义](PROFILING_GUIDE.md) |
| 早期 H20／910B4 算子实验做了什么？ | [RESULTS-friendly](RESULTS-friendly.md) | [历史微基准完整报告](RESULTS.md) |

## 其他材料

[模型调研](RESEARCH.md) 解释为什么选择 GDN；[框架迁移](MAINSTREAM_MIGRATION.md) 解释怎样才算可用；[研究计划](TRAINING_RESEARCH_PLAN.md) 给出下一步假设与否定条件。这三篇直接用原名更新。

[小周报](progress/WEEKLY_2026-10-25.md) 按小任务记录问题、难处、候选和决定，后半部分保留时间表。[项目简历](RESUME.md) 和 [PDF](RESUME.pdf) 用于投递材料，阅读时可以从 STAR 版回查技术依据。

[证据索引](artifacts/README.md) 说明每份 JSON、CSV 和源码快照的用途；[公开导出说明](PUBLIC_EXPORT.md) 说明公开包的范围；[对话索引](conversations/README.md) 解释原始记录怎样使用。原始记录保留原样，避免编辑文字时改变实验历史。

主教程补写于 2026-10-27，使用 2026-10-25 已有终态证据，没有新增 GPU 实验。
