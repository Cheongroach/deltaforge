# 怎样从时间线读出性能问题

profile 是一次运行的事件记录：CPU 提交了什么、GPU 执行了什么、每件事发生在什么时候。它帮助解释耗时，正式性能数字仍需在没有采集干预的运行中测量。

打开 [Profiling Lab](../profiling-lab/profiling-lab.html)，选择 **03 读时间线**。页面可以离线打开；显示的是已经采集的数据，浏览器不执行 CUDA 或训练。

## 先读一个矩形

选 CUDA 采集和 `decode_torch/0` 阶段，点击一个矩形。start 是事件开始时间，duration 是持续时间；时间单位通常为微秒。矩形在横轴上的位置表示发生顺序。

| 事件类型 | 在哪执行 | 怎样理解 |
|---|---|---|
| `cpu_op` | CPU | 框架在主机侧调用的操作 |
| CUDA runtime／driver | CPU | 主机请求 GPU 执行、同步或分配等 API |
| `kernel` | GPU | 设备实际执行的一段程序 |
| `user_annotation` | CPU | 代码给一个阶段加的名字 |

CPU 调用结束不总意味着 GPU 已执行结束。GPU 工作通常先提交再执行，队列和同步会影响二者对应关系。本教学采集特意在阶段内同步，便于对齐 CPU 标签和 GPU 工作，也会改变原本可能存在的重叠。

## 比较融合前后发生了什么

再选 `decode_triton/0`。已有两次对照中，PyTorch 都发射 15 个 kernel，Triton 都是 1 个。融合把衰减、state 读取、修正和输出放入一个程序，减少提交次数。

这条证据支持“本次融合减少了 kernel 数”。如果想进一步说 CPU 提交是瓶颈，还需比较无采集延迟、batch 和其他执行方式；想判断寄存器、缓存或带宽，则需要更细的硬件指标。仅凭矩形少，不能推出任何 GPU 都更快。

## 时间为什么不能简单相加

CPU 算子经常嵌套，外层的 total 时间已经包含内层。把它们全部相加会重复计算；Self CPU 指该项自身、扣除子调用后的主机时间。

不同 GPU stream 可能同时执行。假设一个 kernel 占据 0–4 ms，另一个占据 2–6 ms，持续时间相加为 8 ms，实际覆盖的时间并集只有 6 ms。时间并集描述“这段时间有没有 kernel 在执行”，不等于计算单元 occupancy，也不等于显存带宽利用率。

本 Lab 的算子表聚合整份采集，时间线只显示选中阶段。比较数字前要先确认范围相同。

## 用模拟理解优化能影响多少

在 **04 模拟优化** 中设算子占原模型时间的 20%，让这个算子快 5 倍。假定其他耗时不变，整体加速为：

$$
\frac{1}{(1-0.2)+0.2/5}\approx 1.19.
$$

这就是 Amdahl 定律在这个简化例子中的结果。它说明局部 5× 不能直接变成整模型 5×。页面滑杆是教学模型，不是 vLLM 或本项目的实测训练曲线。

在 **02 看真实测量** 选择 910B4／prefill／T=512，性能比小于 1：编译后更慢。把这一例与模拟放在一起，可以理解为什么后端选择需要同时考虑执行时间和首次编译成本。

## 自己采集时检查什么

从独立 GitHub 仓库根执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m deltaforge.profile \
  --device cpu --output ./my-profile-cpu --iterations 2

python -m deltaforge.profile \
  --device cuda --output ./my-profile-cuda --iterations 2
```

输出包括 `summary.json`、`trace.json` 和压缩 trace。CPU 命令没有 GPU 执行证据；CUDA 命令需要可用设备。导入 Lab 的 trace 上限为 25 MiB，大文件使用 Perfetto／Chrome tracing。

核对 summary 中的形状、源码 hash、设备活动和数值检查，再用不带 profiler 的 benchmark 比较延迟。当前采集对象是算子夹具，prefill 的 K=V=32；旧性能报告主要 K=V=128，不能拼成同一次对照。

若要看 CPU／CUDA 提交与 stream 链路，下一步用 Nsight Systems（nsys）；若要看单 kernel 的寄存器、缓存、occupancy 或 DRAM 指标，用 Nsight Compute（ncu）。本项目目前没有实采这两类数据。

完整采集边界、指标定义与陷阱见 [PROFILING_GUIDE.md](PROFILING_GUIDE.md)。当前真实 LoRA 结果属于另一轮实验，见 [训练结果阅读版](TRAINING_RESULTS-friendly.md)。
