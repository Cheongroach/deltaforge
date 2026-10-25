# 实测证据索引

7 轮任务记录、193 条算子事件、123 格测量；每格保留 7 组 wall 与 device event 样本。

- `run-index.json`：脱离内部任务 ID 的生命周期证据与源码快照 hash；等待传输的两轮并未运行 benchmark。
- `runs/*.events.jsonl`：原始算子 JSON 事件，含数值、源码 hash、计时与编译失败摘要。没有加入平台响应或签名 URL。
- `snapshots/<run>/*.py`：从实际启动/传输归档恢复的源码。发生 benchmark 的轮次逐文件与 worker 声明的 hash 相等。
- `measurements.json`：全部事件合并，附 run 别名。
- `measurements.csv`：所有测量格的常用字段，原始样本见事件 JSON。
- `summary.json`：最终同修复版本的 6 格 prefill 与 3 格 decode 对照，包含编译成本与回本估算。
- `dispatch-plan.json`：离线配置建议；回归的编译候选保留 eager，尚未执行独立确认或在线 dispatch。
- `ir-example/`：CPU 示例 shape 的改写前后 FX / generated Python 文本与数值误差；不是 GPU/NPU 机器码。
- `sources/manifest.json`：模型资料 URL、下载日期与精确字节 hash。Qwen 直接下载 HTML 是应用壳，正文另以 verified excerpt 保存。

从 deltaforge/ 父目录复建：

```bash
python3 deltaforge/tools/report_results.py
python3 deltaforge/tools/write_report.py
python3 deltaforge/tools/build_resume.py --font /path/to/cjk-font.ttf
```

原始服务日志可能包含环境细节，未加入交付包。归档来源为本轮 CLI log-proxy 收集；当前核验没有代替精确 CANN/driver 或多轮独立性能复现。
