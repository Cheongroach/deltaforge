# 近期模型进展与 DeltaForge 选题

调研截止：2026-10-25。面向 AI Infra、算子开发、AI 编译器岗位。

## 为什么选择 GDN

选题要同时满足两个条件：在当前单卡资源上能验证；能够深入到计算依赖、数值和编译，而不只停留在模型调用。Gated DeltaNet（GDN）用固定大小状态矩阵保存 token 历史，其递推可以转换为分块矩阵运算，适合检验这两个条件。

因此选择 DeltaForge：围绕 GDN 求解、图编译和后端适配建立独立项目。领域 IR 指明确表达这类计算语义的中间表示；编译器只对满足其契约的计算改写。异构验证指分别在 H20 和 910B4 运行，不是跨厂商分布式训练。

GDN 原论文发表于 2024 年；FlashQLA 已有成熟 CUDA 优化。本项目价值要靠实际兼容性、稳定性或性能缺口证明，不能把重新实现已有算法当作原创。[来源 1](#s1)[来源 2](#s2)

## 哪些架构变化提供了动机

| 模型 / 工具 | 官方可核验进展 | 算子与编译机会 |
|---|---|---|
| Qwen3.8-Flash-Next，2026-08-26 | GDN＋QSA；四路 Gated Residual；主体 125B、激活 6B，另有 51B N-gram embedding。[来源 3](#s3) | 递推状态、稀疏索引、残差访存、host embedding 异步预取 |
| DeepSeek-V4.1-Flash，2026-09-10 | Causal Encoder–Decoder；552B MoE，输入 / 输出分别激活 8B / 16B。[来源 4](#s4) | 非对称 prefill / decode、压缩与复用 KV、缓存布局 |
| DeepSeek-V4 Preview，2026-04-24 | token compression＋DSA；官方提供开放权重和技术报告入口。[来源 5](#s5) | 压缩、索引与 sparse attention 的融合与调度 |
| FlashQLA，2026-04 发布，仓库 2026-09 更新 | TileLang GDN 融合、上下文并行、前后向优化；README 列明 SM90 / 100 / 103 / 120 / 121、CUDA≥12.8、PyTorch≥2.8。[来源 2](#s2)[来源 6](#s6) | 研究融合边界、片上资源、并行度与自动配置 |
| Qwen3.5-2B 配置与 Transformers 实现 | GDN K/V=128、Hk/Hv=16；24 层中每 4 层有 1 层 full attention；包含 FP32 state 和 chunk fallback。[来源 7](#s7)[来源 8](#s8) | 单卡可验证的真实模型形状与可读数学参考 |

参数和日期按官方陈述，不是独立性能复现。Qwen 博客与 Qwen GitHub 不是两份独立性能证据。FlashQLA 作者报告的 2–3× forward、2× backward 不作为本项目结果。GDN 原理可交叉阅读原论文、FlashQLA 和 Transformers；最新模型的精确发布信息仍属于官方单源事实。

表中 QSA／CSA／DSA 指相应模型的稀疏或压缩 attention 机制；MoE 指按 token 选择部分专家计算；N-gram embedding 指按相邻 token 组合查表。这里只用它们说明可能涉及的索引、访存和调度问题，未实现或评测这些模型。

## 为什么没有同时推进其他方向

| 选题 | 深度 | 当前资源可落地性 | 取舍 |
|---|---|---|---|
| GDN 异构编译 | 递推代数、三角求解、图降低、片上状态与形状特化 | 单 H20＋单 910B4 可完成局部算子验证 | 本项目选择；已有成熟基线，必须准确界定贡献 |
| QSA / CSA 稀疏注意力 | 索引、压缩、稀疏访存、长序列并行 | 需要更多模型、数据与 kernel 基准准备 | 后续复用本项目编译与证据基础 |
| mHC / Gated Residual | 小矩阵归一化、分支融合、FP8 状态 | 局部算子易开始；完整模型关联更难 | 适合扩展，首次项目容易停留在单个小 kernel |
| N-gram host embedding | 预取、缓存、异步 DMA、内存层次 | 大表与 host 带宽影响复现，已有 LongCat 经历 | 与既有工作重叠；可作为后续系统方向 |

以上是工程判断，不是论文性能排名。一个能讲清“为何改写 → 合法条件 → 后端生成 → 数值门禁 → 收益与代价”的项目，比罗列多个模型更适合作为简历主项目。

## 与现有经历的衔接

现有 servedistill profiling 曾定位 FP32 fallback 和细碎 kernel 发射；GDN 编译可以作为后续接入对象，但本次没有将新算子接入那次训练，不能合并两次实验的速度数字。现有 H20 / 910B4 推理实验提供模型与硬件理解，本项目进一步进入算子数学和编译层。

新模型提供选题动机。早期随机输入 fixture 锚定 Qwen3.5-2B 形状；后续已下载锁定权重并完成该模型的短程 LoRA 兼容性验证，见 [训练结果](TRAINING_RESULTS.md)。本项目未运行完整 Qwen3.8-Flash-Next 或 DeepSeek-V4.1，也未复现表中作者的性能主张。

## 来源

<a id="s1"></a>1. [Gated Delta Networks: Improving Mamba2 with Delta Rule](https://arxiv.org/abs/2412.06464)，2024-12-09 提交，2025-03-06 修订；原论文。

<a id="s2"></a>2. [FlashQLA 官方技术说明](https://qwen.ai/blog?id=flashqla)，2026-04-28；chunk 计算、融合和 CP 取舍。

<a id="s3"></a>3. [Qwen3.8-Flash-Next 官方发布](https://qwen.ai/blog?id=qwen3.8-flash-next)，2026-08-26；参数与架构。

<a id="s4"></a>4. [DeepSeek-V4.1-Flash 官方发布](https://www.deepseek.com/en/news/deepseek-v4-1-flash/)，2026-09-10；参数、Causal Encoder–Decoder。

<a id="s5"></a>5. [DeepSeek-V4 Preview 官方发布](https://www.deepseek.com/en/news/v4-preview/)，2026-04-24；压缩＋DSA 及开放资源入口。

<a id="s6"></a>6. [FlashQLA README](https://raw.githubusercontent.com/QwenLM/FlashQLA/main/README.md)，检索于 2026-10-25；版本、平台与 API。main 是移动引用，正式基准需锁 commit。

<a id="s7"></a>7. [Qwen3.5-2B 官方 ModelScope config](https://modelscope.cn/models/Qwen/Qwen3.5-2B/resolve/master/config.json)，最初调研下载配置；后续真实训练的模型文件与 revision／SHA256 另见 [训练资产 lock](training-assets/lock.json)。

<a id="s8"></a>8. [Transformers Qwen3.5 实现](https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/qwen3_5/modeling_qwen3_5.py)，检索于 2026-10-25；形状、状态精度与 normalization 契约，main 不是锁定的模型 revision。
