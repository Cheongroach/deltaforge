# DeltaForge 完整开发实战

这篇文档演示的不是“最后代码长什么样”，而是一个 AI Infra 项目怎样从数学原型走到可安装后端、真实模型训练、性能测量和监控归档。

建议先读完 [一步步认识 DeltaForge 项目](一步步认识DeltaForge项目-friendly.md)。本文不再解释 embedding、Q/K/V、GPU、forward/backward 等基础知识。

每个关键步骤都回答五个问题：

1. 实际使用什么工具和命令？
2. 代码具体改在哪里？
3. 参数为什么这样选？
4. 怎样判断逻辑和功能正确？
5. 哪个文件留下可追溯证据？

## 先区分三类证据

| 标记 | 含义 |
|---|---|
| 本次实跑 | 2026-10-27 在当前 CPU 环境实际执行，命令和结果写入 [manifest](development-validation/20261027/manifest.json) |
| 历史实跑 | 2026-10-25 已在 H20／910B4 完成，本文从保存的 JSON 重算，不冒充本次新 GPU 运行 |
| 设计但未实跑 | 给出下一步工具和验收方法，但明确没有对应结果 |

本次固定公开仓库 main 提交：

```text
6ffc4df42fd4d78e2778edb0724f07f914b6d305
```

当前机器没有可用 CUDA 设备，所以本次新增的是 CPU 验证、构建、安装、证据重算和 CPU profile。所有 H20／910B4 数字仍来自历史终态。

---

## 第一阶段：冻结代码和运行环境

### 第 1 步：先建立隔离复现目录

实际命令：

```bash
mkdir -p /tmp/deltaforge-deep-lab-20261027
git clone --local --no-hardlinks \
  deltaforge-workspace/main \
  /tmp/deltaforge-deep-lab-20261027/repo
git -C /tmp/deltaforge-deep-lab-20261027/repo rev-parse HEAD
git -C /tmp/deltaforge-deep-lab-20261027/repo status --short
```

参数依据：

- `--local`：从已核验本地仓库复制 Git 对象，不依赖网络。
- `--no-hardlinks`：新目录拥有独立对象文件，后续试验不会通过硬链接影响原仓库。
- `/tmp/.../repo`：所有会重写汇总或产生 cache 的命令都在隔离目录执行。

实际输出：

```text
6ffc4df42fd4d78e2778edb0724f07f914b6d305
```

`git status --short` 为空。只有在“提交固定、工作树干净”时，后面的 source hash 和结果才可追溯。

---

### 第 2 步：用 `rg` 建立代码地图

不要从第一行读完整仓库。先搜索入口：

```bash
rg -n \
  "class GDN|def recurrent|block_inverse|class GDNBackend|def use_backend|def parameter_gate|def train_arm|torch.profiler" \
  deltaforge tools
```

代码地图：

| 文件 | 先看什么 |
|---|---|
| `deltaforge/ops.py` | recurrence、doubling、block、chunk forward |
| `deltaforge/backends.py` | 共同接口、后端选择、compiled cache、adapter |
| `deltaforge/validation.py` | fixture、误差定义、device correctness |
| `deltaforge/test_backends.py` | state、gradcheck、反例、真实层类 |
| `deltaforge/training_probe.py` | diagnostic、gate、train、kernel |
| `deltaforge/profile.py` | profiler、阶段标记、trace 汇总 |
| `tools/report_training.py` | 从终态 JSON 重新验收 |

选择 `rg` 而不是逐目录浏览，是为了先围绕“数学 → 接入 → 验证 → 观测”建立主路径。

---

### 第 3 步：读取包配置，不直接 `pip install -e .`

`pyproject.toml` 将依赖分成四层：

```toml
[project]
requires-python = ">=3.11"
dependencies = ["torch>=2.7"]

[project.optional-dependencies]
cuda = ["triton>=3.5"]
qwen = ["transformers==5.2.0", "peft==0.18.1",
        "accelerate==1.11.0", "safetensors==0.8.0"]
fla = ["fla-core==0.4.1", "einops==0.8.1"]
```

参数依据：

- 核心数学只依赖 PyTorch，不强迫 CPU 用户安装 FLA。
- Qwen adapter 锁定 Transformers 5.2.0，因为 callable 参数和模型层类型需要逐版本验收。
- FLA 锁定 0.4.1，因为实验门禁与 wheel hash 都对应这个版本。
- Triton 只属于 CUDA 推理支线。

先分层依赖，再决定安装范围，可以避免为了跑 CPU test 改坏 CUDA／NPU 厂商环境。

---

### 第 4 步：记录实际软件版本

本次执行：

```bash
python - <<'PY'
import importlib.metadata as m, platform, sys
print("python", sys.version.split()[0])
print("platform", platform.platform())
for name in ["torch", "transformers", "peft", "triton"]:
    print(name, m.version(name))
PY
```

实际环境：

| 组件 | 版本 |
|---|---|
| Python | 3.11.2 |
| PyTorch | 2.9.1 |
| Transformers | 5.2.0 |
| PEFT | 0.18.1 |
| Triton | 3.5.1 |
| CUDA device | 不可用 |

版本不是背景信息，而是实验输入。升级任一项后，compiled graph、模型 callable 或 FLA backward 都可能改变。

---

### 第 5 步：先处理真实的动态库冲突

第一次 CPU gate 在导入 torch 前失败：

```text
ImportError: .../libstdc++.so.6:
version `GLIBCXX_3.4.30' not found
```

检查命令：

```bash
printf 'LD_LIBRARY_PATH=%s\n' "$LD_LIBRARY_PATH"
ldd /usr/local/lib/python3.11/dist-packages/torch/lib/libtorch_python.so \
  | rg 'libstdc\+\+|not found'
```

根因：终端工具路径中的旧 `libstdc++.so.6` 排在系统库之前，而系统实际有：

```text
/usr/lib/x86_64-linux-gnu/libstdc++.so.6.0.30
```

最小修复：

```bash
env -u LD_LIBRARY_PATH python -c \
  'import torch; print(torch.__version__, torch.cuda.is_available())'
```

实际输出：

```text
2.9.1 False
```

这里不修改全局 shell 配置，也不复制动态库。只对需要加载 PyTorch 的验证命令移除注入变量，影响范围最小。

---

## 第二阶段：先写一个独立 recurrence 参照

### 第 6 步：固定张量契约

`ops.py` 顶部先明确：

```text
Q/K: [B,T,Hk,K]
V:   [B,T,Hv,V]
g,beta: [B,T,Hv]
state:  [B,Hv,K,V]
```

契约选择依据：

- sequence-major 与 Transformers 5.2 的 Qwen3.5 callable 一致。
- `Hv` 可以是 `Hk` 的整数倍，用 `repeat_interleave` 表达 GQA。
- state 固定为 `K×V` 布局，避免与外部 `V×K` 表示混写。
- 调用方拥有 normalization、gate 构造和 state slot 映射。

契约必须先于优化。否则 output 误差可能只是布局或 scale 不一致。

---

### 第 7 步：构造可重复 fixture

`validation.inputs` 使用：

```python
generator = torch.Generator(device="cpu").manual_seed(20261025)
q = normalize(randn(B, T, Hk, K))
k = normalize(randn(B, T, Hk, K))
v = randn(B, T, Hv, V)
g = -rand(B, T, Hv) * 0.15
beta = rand(B, T, Hv)
state = randn(B, Hv, K, V) * 0.05
```

参数依据：

- 固定 seed：失败可以重放。
- `g∈[-0.15,0]`：覆盖保留历史的常见区间。
- `beta∈[0,1]`：满足项目契约。
- 非零 initial state：不能只验证零状态特例。
- Q/K 预先归一化：把 normalization 与 core GDN 分开。

fixture 是数学／性能夹具，不是模型质量数据。

---

### 第 8 步：实现逐 token recurrence

核心代码：

```python
for token in range(T):
    state = state * exp(g_t)
    predicted = (k_t[..., None] * state).sum(-2)
    residual = (v_t - predicted) * beta_t[..., None]
    state = state + k_t[..., None] * residual[..., None, :]
    output = (q_t[..., None] * state).sum(-2) * K**-0.5
```

实现选择：

- 普通 Python loop 可读性高，适合 oracle。
- 除 double gradcheck 外，计算提升到 FP32。
- 不原地修改调用方 `initial_state`。
- 返回 output 和 final state。

recurrence 的目标是提供结构不同的参照，不追求最快。

---

### 第 9 步：定义统一误差指标

`validation.errors`：

```python
relative_l2 = ||actual - expected||₂ / max(||expected||₂, 1e-12)
max_abs = max(|actual - expected|)
finite = all(isfinite(actual))
```

为什么同时保留三项：

- relative L2 看整体相对偏差；
- max abs 发现少数极端元素；
- finite 阻止 NaN／Inf 被聚合值掩盖。

`verify_pair` 对 BF16 output 使用 `atol=5e-3, rtol=1e-2`，FP32 output 与 state 使用更严格容差。state 无论输入精度如何都按 FP32 验收。

---

## 第三阶段：把 recurrence 改写成 chunk

### 第 10 步：先统一 head-major 布局

`_prepare` 执行：

```python
q, k = repeat_interleave(...).transpose(1, 2)
v = v.transpose(1, 2)
g, beta = transpose(1, 2)
```

这样 chunk 主体使用：

```text
[B,H,T,D]
```

选择依据：同一 head 的连续 token 更方便组成 `T×T` 依赖矩阵。

padding token 的 Q/K/V/beta/g 置零，使它成为恒等 state transition，而不是只把 output mask 掉。

---

### 第 11 步：用 prefix 差构造 decay

代码：

```python
prefix = g.cumsum(-1)
decay = (prefix[..., :, None] - prefix[..., None, :]).clamp(max=0).exp()
decay = decay * causal_mask
```

不要写：

```text
exp(prefix_i) / exp(prefix_j)
```

长序列中分子和分母可能先 underflow。先做 prefix 差再 `exp`，数值范围更可靠。

上三角在取指数前 `clamp(max=0)`，避免无效位置先产生 overflow 再被 mask。

---

### 第 12 步：构造单位下三角系统

代码：

```python
lower = (
    beta[..., :, None]
    * (k @ k.transpose(-1, -2))
    * decay
).tril(-1)
```

`tril(-1)` 保证 `lower` 严格下三角；对角线由外部隐式加单位阵：

```text
(I + lower)Y = RHS
```

这个严格结构是 doubling 幂零展开合法的前提。一般三角矩阵不能直接套用。

---

### 第 13 步：合并两组 RHS

代码：

```python
rhs = torch.cat((
    beta[..., None] * v,
    (beta * prefix.exp())[..., None] * k,
), dim=-1)
```

求解后拆为：

```text
U = factors[..., :V]
W = factors[..., V:]
```

选择拼接的原因：`U/W` 使用同一个 `(I+L)`，合并 RHS 可以避免重复 inverse／solve。

---

### 第 14 步：实现三种求解方法

`GDN(..., inverse_method=...)` 支持：

| 参数 | 实现 | 用途 |
|---|---|---|
| `doubling` | `inverse_unit_lower(lower) @ rhs` | 历史算法和反例 |
| `block` | `block_inverse_unit_lower(lower) @ rhs` | 当前默认训练路径 |
| `triangular` | `torch.linalg.solve_triangular` | 独立参照／候选 |

默认选择 `block`，不是因为它在所有输入上有定理级误差上界，而是因为它在人工反例与真实层输入上修复了已观察故障，同时保留 autograd。

---

### 第 15 步：计算块内 output 和 final state

求解后：

```python
residual = U - W @ state
output = exp(prefix) * (Q @ state) + attention @ residual
state = exp(prefix_tail) * state + K_tail.T @ residual
```

每个 chunk 仍按顺序交接 state，但块内 token 已转成矩阵运算。

正确性验收同时比较：

- 所有 token output；
- final state；
- 对 Q/K/V/g/beta/initial_state 的梯度。

只检查 output 会漏掉下一段才暴露的 state 错误。

---

## 第四阶段：复现旧算法故障并实现修复

### 第 16 步：现场复现 inverse 抵消

本次实跑代码将：

```text
L = 0.999 × tril(ones, -1)
```

分别送入 doubling、block 和直接 triangular 参照。

实际结果：

| C | 方法 | inverse max abs | relative L2 |
|---:|---|---:|---:|
| 32 | doubling | 32 | 4.575531 |
| 32 | block | `5.96e-8` | `1.40e-8` |
| 64 | doubling | `2.5769803776e10` | `4.526666752e9` |
| 64 | block | `5.96e-8` | `1.70e-8` |

这一步证明的是 FP32 计算路径存在反例，不是 GDN 理论错误。

---

### 第 17 步：把 inverse 反例扩展到完整 GDN backward

本次使用：

```text
seed=23
T=64
K=4
V=3
共线 q/k
g=-1e-6
beta=0.999
chunk=32
```

实际结果：

| 方法 | output relative L2 | 六输入聚合 gradient relative L2 |
|---|---:|---:|
| doubling | 3.84459 | 2919.7446 |
| block | `1.99e-7` | `5.54e-7` |
| triangular | `1.82e-7` | `4.97e-7` |

参数选择依据：

- 共线 key 让 `K Kᵀ` 接近强相关下三角结构；
- `g≈0` 让历史长期保留；
- `beta≈1` 让更新接近完全写入；
- recurrence 提供不经过三角 inverse 的参照。

---

### 第 18 步：实现递归 block inverse

核心结构：

```python
M = [[A, 0],
     [B, D]]

M_inv = [[A_inv, 0],
         [-D_inv @ B @ A_inv, D_inv]]
```

实现细节：

- 将非 2 次幂大小 pad 到下一个 2 次幂；
- 从 `1×1` inverse 开始；
- 每轮把两个相邻子块合成更大块；
- 最后裁回原大小。

这样避免显式形成 `L²、L⁴、L⁸...` 的大矩阵幂，同时仍由 PyTorch matmul 组成，autograd 可直接追踪。

---

### 第 19 步：把反例写成永久回归

`test_collinear_keys_weak_decay_forward_backward` 同时检查：

1. block inverse 接近直接 triangular solve；
2. block GDN output 接近 recurrence；
3. 六个输入梯度接近 recurrence。

容差：

```text
forward/state: atol=2e-5, rtol=2e-4
gradient:      atol=2e-4, rtol=2e-3
```

gradient 容差更宽，因为它经过更多浮点运算，但阈值仍在看到新结果前固定。

---

## 第五阶段：接入 Transformers 后端

### 第 20 步：实现一个共同 `GDNBackend`

后端名称：

```text
torch / deltaforge / compiled / fla / recurrent
```

`default` 只在训练 probe 中表示“不替换原 callable”。

共同入口负责：

- shape、dtype、device 检查；
- normalization；
- scale；
- GQA repeat；
- state layout；
- unsupported 选项；
- backend stats。

只统一契约，不在一个 backend 失败时偷偷切换另一个 backend。

---

### 第 21 步：对齐 normalization 和 scale

Transformers 5.2 fallback 使用输入 dtype 算术和 epsilon `1e-6`：

```python
q = q * rsqrt((q*q).sum(-1, keepdim=True) + 1e-6)
k = k * rsqrt((k*k).sum(-1, keepdim=True) + 1e-6)
```

scale 默认：

```text
K^-0.5
```

adapter 在共同位置处理 normalization，并确保 scale 只施加一次。FLA 对照关闭其内部重复 normalization。

这是公平性配置，不是纯粹代码风格。

---

### 第 22 步：使用 functional FP32 state

state shape：

```text
[B,Hv,K,V]
```

规则：

- BF16／FP16／FP32 输入对应 FP32 state；
- double gradcheck 保留 FP64 state；
- 返回新 final state，不原地改 initial state；
- 外部 `V×K` 布局通过显式 transpose 适配。

functional state 让 checkpoint、跨 segment 梯度和错误恢复更可控。

---

### 第 23 步：用 context manager 临时替换层函数

代码入口：

```python
with use_backend(model, "deltaforge", chunk_size=32):
    loss = model(**batch, use_cache=False).loss
```

`use_backend`：

1. 验证 Transformers 恰为 5.2.0；
2. 找到所有 `Qwen3_5GatedDeltaNet`；
3. 保存原 chunk／recurrent callable；
4. 替换为共同 backend；
5. `finally` 中恢复。

测试还故意在 `with` 内抛异常，确认 callable 仍恢复。

---

### 第 24 步：给 compiled 后端建立实例 cache

cache key 包含：

```text
shape / stride / dtype / device / requires_grad
```

编译参数：

```python
torch.compile(operation, fullgraph=True, dynamic=False)
```

选择依据：

- `fullgraph=True`：遇到 graph break 直接失败，不把部分 eager 当编译成功；
- `dynamic=False`：首轮固定 256-token bucket，减少动态形状变量；
- `cache_size=8`：有限 LRU，避免无限保存编译实例。

这不是任意动态 shape 支持。

---

### 第 25 步：明确拒绝未支持选项

当前显式拒绝：

- packed `cu_seqlens`；
- 未知 beta transform；
- 未知 layout／head 选项；
- 非匹配 Transformers 版本；
- FLA 的非 CUDA FP16／BF16 路径。

失败要携带选项名称。silent fallback 会让报告无法判断实际执行了什么。

---

## 第六阶段：使用 export、FX、compile 和 Triton

### 第 26 步：演示历史 semantic solve lowering

历史路径：

```text
GDN(semantic_solve=True)
→ torch.export
→ FX 匹配 deltaforge::unit_lower_solve
→ 展开成 doubling matmul
→ Inductor 或 TorchAir
```

本次 CPU 实跑：

```text
lowered_domain_ops: 1
hoisted_factory_constants: 2
normalized_factory_devices: 7
FX nodes: 146 → 155
semantic op remaining: False
output relative L2: 1.37e-7
state relative L2: 1.32e-7
export/lower: 1.312 s
```

这个结果证明图改写执行成功。它仍是旧 doubling，不属于当前 block 训练支持。

---

### 第 27 步：演示当前 block 的直接 `torch.compile`

本次命令设置独立 cache：

```bash
env -u LD_LIBRARY_PATH \
TORCHINDUCTOR_CACHE_DIR=/tmp/deltaforge-deep-lab-20261027/inductor-cache \
python <block compile diagnostic>
```

配置：

```python
torch.compile(
    GDN(8, inverse_method="block"),
    fullgraph=True,
    dynamic=False,
)
```

CPU 小 fixture 实际结果：

```text
first call:  5.993 s
second call: 0.0429 s
output relative L2: 1.37e-7
state relative L2:  1.30e-7
```

这些时间只说明“首次编译与第二次复用差很多”，不是正式 CPU latency benchmark，更不能替代 H20 数据。

---

### 第 28 步：实现 Triton T=1 decode 支线

`cuda_decode.py` 的一个 program 处理：

```text
一个 batch/head × 一个 value tile
```

融合：

```text
decay
→ KᵀS
→ residual
→ rank-1 state update
→ QᵀS
```

调优参数：

```text
value_tile ∈ {8,16,32,64}
num_warps ∈ {4,8}
B ∈ {1,4,16}
```

选择依据：

- tile 小：program 数多、Q/K 重复读取少；
- tile 大：program 数少，但每个 program 资源压力更高；
- warps 影响并行度与寄存器压力；
- 三个 B 覆盖小 batch 与较高并行度。

这条路径没有训练 backward，不能用于 LoRA 结论。

---

### 第 29 步：为 910B4 适配图编译

历史 TorchAir 路径遇到两个实际错误：

1. generated Python 构造 NPU `torch.device` 失败；
2. `aten.eye` 没有 GE converter。

修复：

```text
device object → device string
固定 eye/ones → 图外 target-device buffer
```

每次修复后重新做 fullgraph 数值验证。不能把 eager fallback 计为 compiled。

---

## 第七阶段：建立分层正确性验证

### 第 30 步：先做语法检查

本次实跑：

```bash
python -m compileall -q deltaforge tools
```

实际结果：

```text
compileall passed
```

它只能发现 Python 语法／字节码编译问题，不能证明数学正确。

---

### 第 31 步：执行 14 项 CPU gate

修复动态库路径后，实际命令：

```bash
env -u LD_LIBRARY_PATH \
PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python -m deltaforge.cpu_gate \
  --output /tmp/deltaforge-deep-lab-20261027/cpu-gates.json \
  --sandbox-exit
```

参数依据：

- `PYTHONDONTWRITEBYTECODE=1`：不污染源码目录；
- OMP／MKL 线程设为 1：减少小测试的线程噪声；
- `--sandbox-exit`：写完 JSON 后绕过受限 `/proc` teardown；
- 输出路径在隔离目录。

实际结果：

```text
14 tests
0 failures
0 errors
0 skipped
2.977 s
```

证据 hash：

```text
9b81a0d24405629f5e4d32e73288ea30a56209676372d4fe85ae461767bac0f3
```

---

### 第 32 步：理解 14 项测试分别挡住什么

| 测试类型 | 防止的错误 |
|---|---|
| recurrence ↔ chunk | 主公式或 chunk 重组错误 |
| double gradcheck | autograd 梯度公式错误 |
| full six-input backward | 某个输入梯度遗漏 |
| state handoff | segment 交接不连续 |
| initial state immutable | 后端意外原地修改 |
| padding／GQA／K≠V | 只在规则 shape 正确 |
| scale／normalization | 重复缩放或比较条件不同 |
| collinear weak-decay | 随机测试漏掉抵消区域 |
| real Qwen layer class | adapter 参数或 callable 不匹配 |
| checkpoint／restore | 重算或异常后模型状态损坏 |
| unsupported inputs | 静默错误路径 |

测试金字塔先快后慢；CPU 失败时不应提交 GPU 任务。

---

### 第 33 步：验证模型资产与数据 hash

GPU 训练前：

```bash
python tools/verify_training_assets.py \
  --model /path/to/Qwen3.5-2B
```

它检查：

- `training-assets/data.json` SHA256；
- 九个模型文件 size；
- 九个模型文件 SHA256。

同名模型目录不够。镜像、重新序列化或移动 revision 都可能得到不同字节。

---

### 第 34 步：执行整模型 gate

历史 H20 命令：

```bash
python -m deltaforge.training_probe \
  --phase gate \
  --model /path/to/Qwen3.5-2B \
  --data training-assets/data.json \
  --output /tmp/deltaforge-model-gate
```

预设门槛：

| 指标 | 上限 |
|---|---:|
| model logits relative L2 | 3% |
| 参数梯度聚合 relative L2 | 8% |
| loss absolute difference | 0.05 |

第三轮历史结果：

```text
accepted: torch, deltaforge, compiled
rejected: default, fla
```

门槛失败后不自动发起训练。

---

### 第 35 步：用 fixed upstream 定位单层差异

历史命令：

```bash
python -m deltaforge.training_probe \
  --phase diagnostic \
  --model /path/to/Qwen3.5-2B \
  --data training-assets/data.json \
  --output /tmp/deltaforge-layer-diagnostic
```

诊断过程：

1. 用 torch 参考执行真实模型；
2. 截取层索引 0／12／22 的 Q/K/V/g/beta；
3. hook 保存该层 output 的 upstream gradient；
4. 在完全相同输入和 upstream 下替换局部实现；
5. 比较 output 与五个输入梯度。

固定 upstream 是为了排除前面层误差传播。

---

## 第八阶段：执行受控 LoRA 训练

### 第 36 步：固定训练配置

| 参数 | 值 | 选择依据 |
|---|---|---|
| 模型 | Qwen3.5-2B | 真实 GDN 层、单 H20 可运行 |
| dtype | BF16 | H20 常见训练精度；state 仍 FP32 |
| LoRA rank | 8 | 验证完整训练链，降低 optimizer 状态成本 |
| trainable params | 8,409,600 | 覆盖 attention／MLP／GDN 投影 |
| batch | 1 | 单卡受控比较 |
| bucket | 256 tokens | 固定 compile shape |
| optimizer | AdamW |
| lr | `1e-4` | 所有后端相同 |
| weight decay | `0.01` | 所有后端相同 |
| gradient clip | 1.0 | 非有限或异常梯度立即暴露 |
| checkpoint | non-reentrant | 验证 compile + backward + recompute |
| updates | 10 | 2 warmup + 8 measured |

这套预算用于后端兼容性，不用于模型收敛研究。

---

### 第 37 步：为每个后端启动独立进程

示例：

```bash
python -m deltaforge.training_probe \
  --phase train \
  --backend deltaforge \
  --repetition 0 \
  --model /path/to/Qwen3.5-2B \
  --data training-assets/data.json \
  --output /tmp/deltaforge-training-deltaforge-0
```

需要对：

```text
torch / deltaforge / compiled
× repetition 0 / 1
```

分别执行，共六个进程。第二轮反转后端顺序，降低固定顺序偏差。

两个 repetition 使用同 seed，不是 multi-seed。

---

### 第 38 步：记录完整训练 step

每步记录：

```json
{
  "step": 2,
  "warmup": false,
  "loss": 3.1,
  "gradient_norm": 1.7,
  "seconds": 0.40,
  "effective_tokens": 194,
  "peak_allocated_bytes": 4847000000,
  "peak_reserved_bytes": 6200000000
}
```

计时范围：

```text
forward → loss → backward → clip → AdamW → synchronize
```

`effective_tokens` 排除 padding，用于计算有效 tokens/s。

---

### 第 39 步：用参数 hash 确认真实更新

训练前后对所有 trainable parameter 按名称和 FP32 bytes 计算 SHA256：

```text
initial_adapter_sha256
final_adapter_sha256
```

验收：

- final 不能等于 initial；
- 同后端两个 repetition 的 final hash 要一致；
- 不要求不同后端 final hash 相同，因为浮点路径不同。

hash 变化证明整体参数内容变化，不证明模型质量提高。

---

### 第 40 步：从终态证据重新验收

本次实跑：

```bash
python tools/report_training.py \
  --output /tmp/deltaforge-deep-lab-20261027/reports/training-summary.json
```

实际输出：

```json
{
  "verified": true,
  "cpu_tests": 14,
  "training_arms": 6
}
```

工具重新检查：

- run `done`、Pod exit 0；
- CPU gate 通过；
- model gate 与固定阈值一致；
- 10 step、2 warmup、8 measured；
- data hash；
- 参数确实变化；
- repetition hash；
- final validation 对 torch ≤0.05。

输出 SHA256：

```text
1a4d18e79a873d52e39ab345174686c24ae0494570643143f08ac83e0129c543
```

---

## 第九阶段：正确测量性能

### 第 41 步：算子 benchmark 先数值、后计时

`benchmark.latency`：

```text
warmup = 4
repeats = 7
iterations per repeat = 5
```

每组同时记录：

- 同步 wall time；
- device event time；
- 七个原始样本；
- median、min、max。

选择中位数是为了降低偶发长尾对中心值的影响；原始样本仍保留。

---

### 第 42 步：扫描候选但保留负结果

prefill：

```text
T/H = 128/16, 512/16, 1024/4
chunk = 16,32,64
```

decode：

```text
B = 1,4,16
tile = 8,16,32,64
warps = 4,8
```

扫描 winner 只是本轮候选。没有独立确认前不能写成永久最优 schedule。

910B4 `T=512` compiled 比 eager 慢，dispatch plan 明确保留 eager。

---

### 第 43 步：重算历史算子汇总

本次实跑：

```bash
python tools/report_results.py
git diff --exit-code -- \
  artifacts/measurements.json \
  artifacts/measurements.csv \
  artifacts/summary.json \
  artifacts/dispatch-plan.json
```

实际结果：

```text
193 events
123 measurements
9 final comparisons
git diff: empty
```

空 diff 说明当前脚本从原始 JSONL 得到了与已发布文件完全相同的聚合结果。

---

### 第 44 步：完整 step 与单算子分开报告

历史完整 LoRA step：

| 后端 | 两次中位数 |
|---|---:|
| torch | 837.78／822.58 ms |
| DeltaForge block | 621.17／635.23 ms |
| compiled block | 380.44／406.48 ms |

同 repetition compiled／torch 比为 2.20×／2.02×。

历史随机 `T=2048` 单算子：

| 后端 | forward+backward 配对中位数 |
|---|---:|
| compiled | 23.22–27.80 ms |
| FLA | 1.92–2.00 ms |

两张表回答不同问题。不能把 2× 写成“胜过 FLA”。

---

### 第 45 步：编译首次成本单独记录

历史 H20：

```text
compiled T=2048 首个进程：105.99 s
共享磁盘 cache 第二进程：3.11 s
```

这些时间包含 dispatch、编译和执行，而且可能受到前面任务预热。

正确报告方式：

- 首次调用；
- warmup；
- 稳态样本；
- cache 是否共享；
- 是否独立冷环境。

不要从一个混合首次时间反推纯 compile cost。

---

## 第十阶段：设计并落地监控

### 第 46 步：先定义五个监控平面

| 平面 | 要回答的问题 | 当前实现 |
|---|---|---|
| 生命周期 | 任务是否真正结束？ | task start/complete、worker complete、run-status、Pod exit |
| 数值正确性 | output/state/logits/gradient 是否合格？ | CPU gate、model gate、layer diagnostic |
| 训练健康 | loss、gradient、参数是否更新？ | train_step、gradient norm、adapter hash、validation |
| 性能资源 | step、token/s、memory、kernel 如何？ | wall/device time、allocated/reserved、profiler |
| 可追溯性 | 结果属于哪版代码和数据？ | source/model/data SHA256、版本、原始 JSONL |

监控不是只画 GPU utilization。先保证任务身份和数值正确，再解释速度。

---

### 第 47 步：使用结构化 JSONL 事件

`training_probe.emit` 输出：

```text
DELTAFORGE {"event":"train_step", ...}
```

attempt-03 保存了：

```text
158 events
10 training_task_start
30 layer_diagnostic
5 model_gate
60 train_step
6 training_result
16 kernel_result
1 training_worker_complete
```

设计原则：

- 每行一个完整 JSON，便于流式采集；
- `event` 是稳定类型；
- backend、repetition、step 作为关联键；
- 原始事件保留，报告另行聚合；
- 一个失败不覆盖其他子任务证据。

---

### 第 48 步：为 step 设计健康指标

| 指标 | 正常判断 | 异常动作 |
|---|---|---|
| `loss` | finite | 立即失败并归档 batch/后端 |
| `gradient_norm` | finite；clip 前可观察 | 非有限立即失败 |
| `seconds` | 保存全部样本 | 长尾先查同步、编译、cache |
| `effective_tokens` | 与预算一致 | 不一致则拒绝吞吐比较 |
| `peak_allocated` | 同 scope 比较 | 只报告 allocator allocated |
| `peak_reserved` | 观察缓存池 | 不与 allocated 混写 |
| adapter hash | final ≠ initial | 相同则训练失败 |

这些是离线验收规则，不是已经部署的在线告警服务。

---

### 第 49 步：使用 `torch.profiler` 标记阶段

核心实现：

```python
with torch.profiler.profile(
    activities=activities,
    record_shapes=True,
    profile_memory=True,
    with_stack=False,
) as profiler:
    with torch.profiler.record_function(
        "deltaforge/prefill_eager/0"
    ):
        function()
```

参数依据：

- `record_shapes=True`：定位 shape 变化；
- `profile_memory=True`：观察分配／释放事件；
- `with_stack=False`：减少 trace 体积和采集开销；
- 编译和四次 warmup 放在 profiler 窗口外；
- CUDA 教学采集在阶段内同步，方便 CPU 标签与 GPU 工作对齐。

同步会改变 overlap，所以 profile duration 不作为正式 latency。

---

### 第 50 步：现场采集 CPU trace

本次实跑：

```bash
env -u LD_LIBRARY_PATH \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python -m deltaforge.profile \
  --device cpu \
  --output /tmp/deltaforge-deep-lab-20261027/profile-cpu \
  --iterations 2
```

实际结果：

```text
profile window: 55,369.916 us
phases: 6
CPU operator events: 7,794
timeline events: 7,800
GPU kernels: 0
trace.json: 4,060,953 bytes
trace.json.gz: 242,720 bytes
```

正确性：

```text
prefill output relative L2: 2.07e-7
prefill state relative L2:  1.66e-7
```

CPU capture 中 `gpuGetDeviceCount` 报错日志不构成采集失败；summary 明确记录 `gpu_trace_available=false`。

---

### 第 51 步：正确汇总时间线

`summarize_trace`：

1. 用 `user_annotation` 找 DeltaForge 采集窗口；
2. 只保留窗口内完整事件；
3. 分离 `cpu_op`、kernel、runtime、driver、memcpy；
4. 保留真实 start 和 duration；
5. 对 GPU kernel 区间求并集。

不能直接把所有 kernel duration 相加，因为不同 stream 可能重叠。

`gpu_interval_union_us` 也不等于 SM occupancy、Tensor Core 利用率或 HBM 带宽。

---

### 第 52 步：把 profile 和 benchmark 放进离线 Lab

本次实跑：

```bash
python tools/build_lab_data.py \
  --cpu /tmp/deltaforge-deep-lab-20261027/profile-cpu/summary.json \
  --cuda profiling-lab/assets/cuda-summary.json \
  --output /tmp/deltaforge-deep-lab-20261027/lab-data.js
```

实际结果：

```text
profiles: cpu, cuda
operator events: 193
final comparisons: 9
output: 2,669,600 bytes
```

工具只接受包含 timeline 和 phase 的真实 summary，不会生成虚构 benchmark 数字。

---

### 第 53 步：定义监控门槛与处置顺序

| 触发 | 处置 |
|---|---|
| run 非 done 或 Pod exit 非 0 | 不进入结果汇总 |
| CPU gate 失败 | 不提交 GPU 工作 |
| logits >3%／gradient >8%／loss >0.05 | 后端跳过训练 |
| 任意 NaN／Inf | 当前子任务失败，保留其他结果 |
| effective token 预算不一致 | 禁止吞吐比较 |
| compiled 慢于 eager | dispatch 选择 eager |
| cache miss 激增 | 检查 shape／stride／dtype key |
| 显存峰值上升 | 分开检查 allocated、reserved、saved tensors |

告警顺序是“任务身份 → 数值 → 训练 → 性能”，避免在错误结果上优化速度。

---

### 第 54 步：明确尚未落地的监控

当前没有：

- Prometheus／Grafana 在线训练告警；
- 当前 block 训练的 nsys trace；
- ncu 寄存器、occupancy、DRAM 与指令指标；
- 长周期 multi-seed 质量监控；
- 910B4 新 block 训练监控。

这些内容只能写成下一步：

```text
nsys：CPU 提交、runtime、stream、同步和 overlap
ncu：选定 kernel 的硬件计数器
线上监控：事件汇聚、时间序列、告警路由
```

不要把 Profiling Lab 教学页面当成生产监控平台。

---

## 第十一阶段：构建、安装和发布

### 第 55 步：构建 wheel

本次实跑：

```bash
python -m pip wheel \
  --no-deps \
  --wheel-dir /tmp/deltaforge-deep-lab-20261027/wheelhouse \
  .
```

为什么 `--no-deps`：

- wheel 构建不需要重新下载大型 PyTorch；
- 避免构建阶段改变已固定运行环境。

实际产物：

```text
deltaforge_gdn-0.3.0-py3-none-any.whl
size: 36,367 bytes
sha256: 2ef8bbdbc4d58550bedd4be81734d3390125de2383e2fecf7dd4764f0535cb2e
```

---

### 第 56 步：处理没有 `ensurepip` 的 Debian venv

第一次：

```bash
python -m venv --system-site-packages .../venv
```

失败：

```text
ensurepip is not available
```

不需要 sudo 安装系统包。改用：

```bash
python -m venv \
  --without-pip \
  --system-site-packages \
  /tmp/deltaforge-deep-lab-20261027/venv-nopip

python -m pip \
  --python /tmp/deltaforge-deep-lab-20261027/venv-nopip/bin/python \
  install --no-deps \
  /tmp/deltaforge-deep-lab-20261027/wheelhouse/deltaforge_gdn-0.3.0-py3-none-any.whl
```

这是局部修复，不修改系统 Python。

---

### 第 57 步：从仓库外做 installed smoke

在 `/tmp/deltaforge-deep-lab-20261027` 执行，避免 import 到源码目录：

```python
import deltaforge
print(deltaforge.__file__)
```

实际路径：

```text
.../venv-nopip/lib/python3.11/site-packages/deltaforge/__init__.py
```

再比较 `GDN(4)` 与 recurrence：

```text
output relative L2: 1.08e-7
state relative L2:  1.05e-7
```

这证明安装后的包可导入并执行，不代表 GPU wheel 或所有可选依赖已验收。

---

### 第 58 步：重绘证据图并逐字节比较

本次：

```bash
python tools/plot_evidence.py \
  --output /tmp/deltaforge-deep-lab-20261027/figures

cmp figures/model-checks.png \
    /tmp/deltaforge-deep-lab-20261027/figures/model-checks.png
```

三张 PNG 都逐字节一致：

```text
model-checks.png:
a83db327103438d5fb47ce5d538cc266fc9d893b16f3449687e5d2139b71de75

training-steps.png:
14b6d543d241e19f215d965e7b4acbae2466166ed98c2d8c52d9332a413e9b1a

kernel-baseline.png:
a3155ec9c85d544dd21326feb7fbc08fea4d5d3e8f1ea29222071d32c57232c7
```

绘图脚本只读取终态 JSON，没有重新跑模型。

---

### 第 59 步：最后做 Git 和 checksum 验收

发布前：

```bash
git diff --check
git status --short
sha256sum -c SHA256SUMS
```

然后：

```bash
git fetch origin main doc
git push --atomic origin main doc
git ls-remote origin refs/heads/main refs/heads/doc
```

规则：

- fetch 后确认本地基于最新远端；
- main/doc 原子 push，避免一支成功另一支失败；
- push 后独立读取远端 HEAD；
- `SHA256SUMS` 覆盖公开文件，不包含自身。

---

## 第十二阶段：按顺序完成一次复现

### 第 60 步：CPU 开发验收最短路径

```bash
# 1. 固定提交并检查环境
git rev-parse HEAD
python --version

# 2. 语法
python -m compileall -q deltaforge tools

# 3. CPU 数值与接口
env -u LD_LIBRARY_PATH \
PYTHONDONTWRITEBYTECODE=1 \
OMP_NUM_THREADS=1 \
MKL_NUM_THREADS=1 \
python -m deltaforge.cpu_gate \
  --output ./cpu-gates.json

# 4. 构建
python -m pip wheel --no-deps --wheel-dir ./dist .

# 5. 证据重算
python tools/report_training.py --output ./training-summary.json
python tools/plot_evidence.py --output ./figures-rebuilt

# 6. CPU profile
env -u LD_LIBRARY_PATH \
python -m deltaforge.profile \
  --device cpu \
  --output ./profile-cpu \
  --iterations 2
```

注意：`env -u LD_LIBRARY_PATH` 是本次终端环境的修复，不是所有机器都必须使用。

---

### 第 61 步：H20 设备验收顺序

在已经匹配 PyTorch／CUDA／Triton 的 H20 容器中：

```bash
# 1. CPU gate
python -m deltaforge.cpu_gate --output ./cpu-gates.json

# 2. 固定随机算子与编译
python -m deltaforge.benchmark \
  --device cuda \
  --output ./benchmark-cuda

# 3. 教学 profile，不能替代 benchmark latency
python -m deltaforge.profile \
  --device cuda \
  --output ./profile-cuda \
  --iterations 2

# 4. 模型资产
python tools/verify_training_assets.py \
  --model /path/to/Qwen3.5-2B

# 5. 单层诊断
python -m deltaforge.training_probe \
  --phase diagnostic \
  --model /path/to/Qwen3.5-2B \
  --data training-assets/data.json \
  --output ./diagnostic

# 6. 整模型门禁
python -m deltaforge.training_probe \
  --phase gate \
  --model /path/to/Qwen3.5-2B \
  --data training-assets/data.json \
  --output ./gate

# 7. 只对 gate 通过的后端训练
python -m deltaforge.training_probe \
  --phase train \
  --backend compiled \
  --repetition 0 \
  --model /path/to/Qwen3.5-2B \
  --data training-assets/data.json \
  --output ./train-compiled-0
```

第 7 步不会自动读取第 6 步并阻止误用。调用方必须先检查 `gate_passed`。

---

### 第 62 步：每次实验完成后的验收清单

| 检查 | 必须回答 |
|---|---|
| Source | commit、源码 SHA256 是否记录？ |
| Environment | Python、framework、driver／CANN、device 是否记录？ |
| Inputs | shape、dtype、seed、model/data hash 是否记录？ |
| Correctness | output、state、loss、gradient 是否分别检查？ |
| Completion | run done、exit 0、complete marker 是否同时存在？ |
| Performance | warmup、样本、同步、分母、首次成本是否明确？ |
| Training | 参数 hash 是否变化？有效 token 预算是否一致？ |
| Monitoring | 原始事件、trace、summary 是否可回查？ |
| Scope | 单算子、完整 step、模型质量是否分开？ |
| Failures | 失败轮次和负收益 shape 是否保留？ |

任何一项缺失，都应缩小结论，而不是补写一个更强的故事。

---

## 当前落地结果

本次新增 CPU 实跑证明：

1. 固定提交的 14 项 CPU gate 全部通过。
2. doubling 抵消故障与 block／triangular 修复可现场重现。
3. 历史 FX lowering 和当前 block direct compile 都能在 CPU fixture 上通过数值检查。
4. wheel 可以构建，并在仓库外隔离路径导入执行。
5. 历史训练和算子证据可以由原始 JSON 重新汇总，已发布聚合不变。
6. 三张证据图可以逐字节重建。
7. 新 CPU profiler trace 和离线 Lab 数据实际生成。

仍未新增：

- H20 或 910B4 运行；
- FLA 根因定位；
- 全参数／多卡／packed 训练；
- nsys／ncu 硬件机制证据；
- 线上监控和长期模型质量实验。

完整本次执行记录见 [development-validation/20261027/manifest.json](development-validation/20261027/manifest.json)。历史 H20／910B4 原始入口见 [证据索引](artifacts/README.md) 与 [训练结果](TRAINING_RESULTS.md)。
