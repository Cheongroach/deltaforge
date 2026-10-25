# DeltaForge 性能观测与实操手册

这一版先把学习和观测闭环做出来：打开 [Profiling Lab](../profiling-lab/profiling-lab.html)，不用 GPU、不用联网，也能探索真实测量、读已采集时间线、计算模拟结果和做判断练习。浏览器不会执行 PyTorch／CUDA；实际采集入口是 `profile.py`。

## 先玩十五分钟

1. 打开 **02 看真实测量**，选择 **910B4 / prefill / T=512**。基线／候选小于 1，表示候选更慢。查看七组原始样本，判断是否保留 eager。
2. 打开 **03 读时间线**，选择一个采集和一个阶段。点击矩形，看事件名称、开始时间和 duration。`cpu_op` 是主机算子调用，`kernel` 是 CUDA 执行。CPU-only 采集没有 GPU kernel。
3. 有 CUDA 采集时，对比 `decode_torch/0` 与 `decode_triton/0`。先数 kernel，再看 CPU runtime 与设备事件；不要用 profiler 的阶段延迟替代无采集基准。
4. 打开 **04 模拟优化**，先选发射开销，再选带宽与计算场景。只改变融合后的 kernel 数，保持总计算量不变；观察什么时候减少发射基本无用。
5. 将算子占比设为 20%、算子加速设为 5×，整模型的计算示例约为 1.19×。这不是 vLLM 实测。
6. 完成 **05 做三个判断**，导出练习 JSON。复跑清单也会记录在当前浏览器中；浏览器限制本地存储时，导出仍可用。

## 前一轮怎么测，这一轮怎么采

前一轮采用独立数值 oracle，加上 4 次预热、7 组重复、每组 5 次调用。每组同时记录同步 wall 时间与 device event 时间，编译成本单列。那一轮没有采集 NSYS／NCU，也不能从 5× 的 PyTorch 对照速度推断 HBM 或 SM 的具体利用率。

这一轮使用 `torch.profiler.profile`，选择 CPU 或 CPU＋CUDA activities，开启 `record_shapes=True`、`profile_memory=True`，并通过 `record_function` 标记每个阶段。导出 Chrome trace 和按算子聚合的表。首次导出／编译、四次预热和首次 backward 都位于 profiler 窗口外。[PyTorch 官方教程](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)

CUDA 教学采集在每个阶段内部同步，让 CPU 标签结束前对应 GPU 执行已完成。这会改变提交和重叠；它方便理解对应关系，不能证明在线并发、训练 overlap 或真实服务延迟。需要检验流水重叠时，应另采不在每个阶段同步的运行，并保留必要的跨线程／stream 关联。

### 采集对象

| 路径 | 输入与操作 | 能说明什么 |
|---|---|---|
| CPU GDN prefill eager | B=1,T=128,H=4,K=V=32,chunk=32 | 主机调用与嵌套结构 |
| CUDA prefill eager / compiled | 同一上述 shape 与 FP32 state | 编译路径的 kernel、runtime 与阶段关系 |
| decode PyTorch / Triton | B=4,T=1,H=16,K=V=128；CUDA 使用 BF16 输入＋FP32 state | 融合前后实际发射与设备事件 |
| PyTorch forward / backward | 上述 prefill 六输入均求梯度 | autograd 的算子事件；不是完整模型训练 |

这份 profile 的 prefill shape 与前一轮 K=V=128 的三个基准不同。不能把两套数据合成一个性能比。每份 `summary.json` 包含 source SHA-256、采集形状、版本、数值检查和采集边界。

本地开发机的 CPU trace 已导出，但进程退出时遇到 `/proc/.../comm` 线程命名的沙箱限制；这不构成 GPU 运行证据。H20 容器采集的终态与 CUDA kernel 是否存在，以 `profiling-lab/assets/profile-manifest.json` 为准。

## 自己执行一遍

在包含 `deltaforge/` 的父目录执行：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m deltaforge.profile \
  --device cpu --output ./my-profile-cpu --iterations 2

python -m deltaforge.profile \
  --device cuda --output ./my-profile-cuda --iterations 2
```

输出为 `summary.json`、`trace.json` 和 `trace.json.gz`。在页面第 03 页导入 `trace.json`，文件内容只在浏览器本地处理；教学导入上限为 25 MiB。大型 trace 应使用 Perfetto／Chrome tracing。导入的外部 trace 不会自动成为已经通过数值验收的基准数据。

接着运行不带 profiler 的测量：

```bash
python -m deltaforge.benchmark \
  --device cuda --output ./my-unprofiled-benchmark
```

先检查正确性，再比较相同输入、精度、state 与初始化条件下的原始样本。不要为了得到漂亮数字，把首次编译排除在一个方案之外，却计入另一个方案。

## 读 profile 时的五个陷阱

| 看到的现象 | 可以提出的假设 | 还不能声称什么 | 下一步 |
|---|---|---|---|
| CPU 多次发射、GPU 短 kernel 之间有间隙 | 提交成本可能影响执行 | 已证明 CPU launch 是唯一瓶颈 | 对比融合、CUDA Graph、batch；用 nsys 查看链路 |
| Self CPU 很大 | 某个算子自身占用主机时间 | GPU kernel 同样很慢 | 看对应设备事件，区分排队／同步 |
| 多行 CPU total 都大 | 存在嵌套调用 | 各行可以相加为总耗时 | 看 Self CPU 与调用时间区间 |
| GPU kernel 时长相加超过阶段时长 | stream／事件可能重叠 | GPU 覆盖率超过 100% | 合并区间，看实际 overlap 与同步 |
| 大 tile 比小 tile 慢 | 并行度、寄存器、冗余访存可能存在取舍 | 已确认寄存器 spill 或 HBM 饱和 | ncu 检查寄存器、spill、occupancy、DRAM 与 cache |

`gpu_interval_union_us` 是 kernel 时间区间的并集，不是 SM occupancy、Tensor Core 利用率或 HBM 利用率。算子表的 Self CPU 内存可以为负，表示释放；它不是 GPU 进程峰值。`record_shapes` 和内存采集本身也会增加开销。

本次还遇到同名 CPU `user_annotation` 与 GPU `gpu_user_annotation` 标签重复的问题。页面用 CPU 标签定义阶段，保留 GPU 事件；CUDA runtime 与 driver 都是主机侧 API 调用，不能把它们当作设备 kernel。算子表聚合整份采集，而时间线按当前阶段筛选，两者的范围不同。

## PyTorch profiler、nsys、ncu 各做什么

| 工具 | 更适合的问题 | 本版状态 |
|---|---|---|
| PyTorch profiler | 哪些 PyTorch 算子、shape、forward/backward 阶段在执行 | 已提供采集代码与实际 trace |
| Nsight Systems / nsys | CPU 提交、CUDA runtime、kernel、stream、同步、NVTX 时间线 | 当前没有实采 nsys；下一轮接入 |
| Nsight Compute / ncu | 选定 kernel 的寄存器、occupancy、DRAM、cache 与指令指标 | 当前没有实采 ncu；不能填造指标 |

不要在同一进程里随意叠加多个 CUPTI profiler。做 nsys／ncu 时，应使用单独的预热后工作负载，并检查该版本工具的参数和硬件权限。ncu 往往会重放 kernel；重放后的时间不能直接当作原应用 latency。

## 优化前有哪些候选

| 候选 | 优点 | 代价与不选的条件 | 当前验证 |
|---|---|---|---|
| token recurrence | 数学直观，适合独立 oracle | 多步发射，不适合拿它代表成熟 kernel | 已实现与验证 |
| 原生 triangular solve | 语义清楚，直接表达 chunk 求解 | 某些后端慢或 converter 不支持 | 两端测过；NPU 图中 eye 不支持 |
| 有限幂零倍增＋矩阵乘 | 利用领域契约，便于后端编译 | 小 GEMM 数增加，部分 shape 回归 | 两端编译与数值验收；T=512 NPU 回归 |
| Triton decode 融合 | 减少状态重复读写与多次发射 | tile/warps 改变并行度与资源压力 | CUDA 24 格配置扫描 |
| FLA / FlashQLA 成熟 kernel | 更强的真实竞争基线，含既有优化 | API、精度、状态布局和 backward 需统一 | 尚未安装与复现；不能宣称胜出 |
| 原生 AscendC / Cube＋Vector 设计 | 可以针对 NPU 数据搬运与片上资源 | 工程量、SDK 与数值验收要求高 | 尚未实现 |

除了算法候选，还有两个实际选择：**编译成本是否值得摊销**，以及 **特定 shape 是否保留 eager**。910B4 的 T=128 约 5% 收益需要约 88 万次调用摊销本轮编译；T=512 编译更慢，不回本。这些判断比统一“开启编译”更接近实际系统需求。

## 做过的人会特别注意什么

- 严格下三角矩阵才满足幂零契约。不能把一般三角求解都替换成有限多项式。
- 数学等价不等于浮点 bitwise 相同；state 的小误差可能在多步交接中累积。
- 使用 prefix 差再取 exp，避免直接相除的 underflow；mask 的无效上三角也不能先 overflow 再处理。
- `torch.device("npu")` 在 generated Python 中可能失败，固定 `eye` 的 converter 也可能缺失。兼容性修复必须保留旧错误并实机验证。
- 参数扫描的最优 tile 可能随 batch、CPU 提交成本、cache 和运行噪声改变。两轮最优配置不同，并不证明某个 schedule 永久最优。
- compiler 成功、profile 采集成功、数值通过、稳定计时通过，是四件事；每一项都要单独记录。

## 实际意义与 vLLM 的比较边界

当前最具体的用途是：研究 GDN 推理／训练路径的结构，发现某后端通用求解的适配问题，生成满足数学契约的可编译图，并按 shape 判断候选是否值得使用。

要证明在特定条件下比 vLLM 更有效，应把候选接入同一个模型与服务框架，在相同硬件、模型权重、dtype、batch／并发、输入输出长度、缓存命中率与状态管理条件下，测 TTFT、TPOT、吞吐、峰值显存和数值／任务质量。先检查 vLLM 已使用的 GDN kernel；用成熟内核作基线，而不是用 Python oracle 代替 vLLM。

可能值得验证的条件包括：固定 shape 高频重复、GDN 占比高、小 batch 的发射敏感场景、NPU 后端缺少可编译算子、训练中大图的保存激活与重算成本。它们是**待检验假设**，不是本项目已经成立的优势。

## 下一轮：进入训练框架与研究

参见 [研究与训练路线](TRAINING_RESEARCH_PLAN.md)。这一版尚未完成完整训练框架接入；页面中的 backward 是真实 PyTorch 算子夹具，模拟曲线不属于训练性能数据。下一轮先闭合成熟基线、解析梯度／保存激活、重算、真实模型训练步，再讨论多卡与论文贡献。
