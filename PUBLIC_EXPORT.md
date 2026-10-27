# 公开材料的范围

main 提供运行代码、实验结果、阅读文档和离线 Profiling Lab；doc 另保存完整手稿、公共源码快照和历史记录。阅读入口为 [READING_GUIDE.md](READING_GUIDE.md)，实验文件的用途见 [证据索引](artifacts/README.md)。

## 可以怎样追溯结果

算子和训练证据保留数值、原始计时样本、终态和来源 hash。doc 的历史快照省略组织专用文件，所包含的公共源码仍与 worker 采集的 SHA256 逐文件核对。被省略文件和原始 bundle 的 hash 只提供来源指纹，不能据此在外部重建内部平台。

训练报告以显式 torch adapter 为参照，保留 native default／FLA 的梯度失败、前两轮失败、共享缓存首次成本和旧 doubling 的抵消反例。短程 LoRA、随机算子性能和历史 NPU 编译属于不同验证范围。

## 哪些材料没有公开

公开包没有凭据、签名日志链接、内部资源挂载、平台原始响应、receipt 或组织专用 worker／launcher。早期名为 `platform.py` 的组织 helper 在公共快照中省略，captured hash 继续保留。公开 compact JSON 不是全部 stdout／stderr。

调研保留出处、读取日期、配置事实和原始字节 hash，省略网页 HTML dump 与长篇转载。原研究材料与官方陈述仍需按对应来源核对。

## 时间线和对话记录的覆盖

[conversations/COVERAGE.json](conversations/COVERAGE.json) 记录可获得的来源数量、导出范围和截止时间。公开消息文字及工具调用索引保留；云端记录没有工具结果正文，因此对话不能替代工程日志。系统／开发者上下文、非公开控制内容和隐藏推理未导出。

截止时间之后的训练和发布动作另记在 [小周报](progress/WEEKLY_2026-10-25.md)。本次文档整理不是补录完整对话，也没有改变原始记录。

## 哪些文件经过阅读整理

2026-10-26 对 authored 文档重新组织，并为六篇技术材料增加 `-friendly` 版本；2026-10-27 增加 72 步单一首读教程 [一步步认识 DeltaForge 项目](一步步认识DeltaForge项目-friendly.md) 和 [完整开发实战](DeltaForge完整开发实战-friendly.md)。实战稿配套 [manifest](development-validation/20261027/manifest.json) 只记录本次 CPU 工具链复跑和历史 GPU 证据重算，不新增 GPU 结果。三张图从第三轮 JSON 重绘；原始 JSON／CSV、源码快照、对话和第三方许可保留。整理范围及三轮编辑核验见 [documentation-map.json](documentation-map.json)。

profile 页面用 CPU `user_annotation` 定义阶段，保留原 summary 和采样源码，以便解释数据投影。算子表聚合整次采集，页面时间线按阶段筛选，两者不能直接当作同一统计范围。
