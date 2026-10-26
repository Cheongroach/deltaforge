# 实测证据怎样回查

先在报告中找到要核对的结论，再找对应原始样本与运行源码。文件 hash 是内容指纹，用于确认“这份结果来自哪版程序”；它本身不证明数值正确或性能改善。

## 早期双硬件算子结果

本目录保存 7 轮任务记录、193 条算子事件、123 格测量，每格保留 7 组 wall 与 device event 样本。两轮等待传输的任务没有执行 benchmark。这里的结果对应旧算法，报告见 [RESULTS.md](../RESULTS.md)。

| 要核对什么 | 文件 | 阅读方法 |
|---|---|---|
| 任务是否结束、是否执行了 benchmark | [run-index.json](run-index.json) | 同时看终态、Pod exit、benchmark_complete |
| 原始数值、计时与失败 | `runs/*.events.jsonl` | 按运行别名、shape、event 找记录；不是平台 stdout 全量日志 |
| 汇总和原始样本 | [measurements.json](measurements.json)、[CSV](measurements.csv) | CSV 便于浏览，完整字段与样本看 JSON |
| 报告中的性能比和回本 | [summary.json](summary.json) | 6 格同 chunk prefill、3 格 decode；确认参照与成本单位 |
| 为什么某形状保留 eager | [dispatch-plan.json](dispatch-plan.json) | 离线建议，还没有独立确认或在线 dispatcher |
| 实际运行代码 | `snapshots/<run>/*.py` | 由启动归档恢复，逐文件核对 worker 声明的 hash |
| 编译改写了什么 | `ir-example/` | CPU 示例 FX／generated Python；不是设备机器码或 GE IR |
| 调研来源 | `sources/manifest.json` | URL、日期、原始字节 hash；公开包省略长网页原文 |

## 后续真实模型训练结果

[training-results](../training-results/summary.json) 是另一条证据链，包含三轮 attempt 和固定 H20 LoRA。[TRAINING_RESULTS.md](../TRAINING_RESULTS.md) 解释数值与性能，`attempt-03/evidence/` 保留模型 gate、六个训练臂和两次 kernel 对照。

模型／数据来源与指纹见 [training-assets/lock.json](../training-assets/lock.json)。doc 分支的 `docs/artifacts/training-snapshots/` 保存各轮公共源码，manifest 逐文件核对 worker hash。不能用旧 NPU 快照证明新 block 在 NPU 训练通过。

[figures/evidence.json](../figures/evidence.json) 保存三张阅读图的原始样本、统计值和来源 SHA256；图没有增加实验样本。

## 重算与复建

从独立 GitHub 仓库根执行：

```bash
python tools/report_results.py
python tools/report_training.py
python tools/plot_evidence.py
python tools/build_resume.py
```

前两条从已有事件重算汇总，不运行 GPU；绘图需要 Matplotlib，PDF 需要 reportlab，默认使用仓库附带中文字体。`build_resume.py` 读取新版 Markdown，不覆盖其文字。

`write_report.py` 保留早期报告生成模板，默认输出到临时文档构建目录，供追溯，不覆盖当前编辑稿。运行代码或采集新结果的命令见 [接入说明](../TRAINING_INTEGRATION.md)。

公开记录省略组织专用 worker、原始平台响应和签名 URL，范围见 [PUBLIC_EXPORT.md](../PUBLIC_EXPORT.md)。早期软件版本未覆盖精确 CANN／driver build；已有 hash 核验也不能替代独立复现。
