# Qwen3.5 训练后端接入

本接口在 Transformers 5.2.0 中只替换 GDN callable，退出时恢复原实现。H20 的 block 和直接 compiled 路径已各完成两次十步 LoRA；参照为显式 torch adapter，原生默认／FLA 的真实模型梯度检查未通过。完整范围见 [结果报告](TRAINING_RESULTS.md)；第一次操作建议先读 [接入阅读版](TRAINING_INTEGRATION-friendly.md)。

下文把 adapter 指接入层，functional state 指返回新状态、保留训练计算图而不修改调用方状态的接口。compiled 指直接编译 GDN 计算，未编译整个模型。

## 安装与使用

在匹配硬件的 PyTorch 环境中，从独立仓库根目录执行：

```bash
python -m pip install --no-deps -e .
python -m pip install "transformers==5.2.0" "peft==0.18.1" \
  "accelerate==1.11.0" "safetensors==0.8.0"
# CUDA FLA 可选；先确认 torch / Triton 匹配。
python -m pip install --no-deps "fla-core==0.4.1" "einops==0.8.1"
```

```python
from deltaforge.backends import use_backend

optimizer.zero_grad()
with use_backend(model, "deltaforge", chunk_size=32) as selection:
    loss = model(**batch, use_cache=False).loss
    loss.backward()
    optimizer.step()
    print(selection.stats())
# 退出 context 后原 callable 恢复。
```

后端为 `torch`、`deltaforge`、`compiled`、`fla` 和 `recurrent`。训练 probe 另提供 `default`，保留模型原生 callable。attention、卷积、投影、norm 和优化器仍由原框架管理。

## 契约与限制

| 项目 | 本轮支持 |
|---|---|
| 输入 | Q/K `[B,T,H_k,K]`、V `[B,T,H_v,V]`；sequence-major，GQA repeat-interleave |
| gate | caller 构造的非正 log decay；beta `[0,1]`，无隐式 sigmoid |
| normalization | 显式开关；Transformers 5.2 fallback 算术，epsilon `1e-6`；scale 一次 |
| state | 功能式 FP32 `[B,H_v,K,V]`；optional initial/final，支持 V×K 外部布局 |
| 梯度 | PyTorch 路径保留跨 segment 梯度；double 路径用于有限差分 |
| compiled | shape/stride/dtype/device/grad 相关的实例 LRU；不是动态 shape 承诺 |
| 未支持 | packed、未知 gate transform、未知选项、未经核验的 Transformers 版本 |

`validate_inputs` 提供 gate 值和数值诊断；热路径的结构检查不逐元素同步到 CPU，caller 须满足 gate 范围。FLA 0.4.1 仅是本轮 CUDA FP16/BF16 基线，不宣称是最新版，也不作为 NPU 后端。FLA adapter 对照关闭内部重复 normalization，真正原生默认路径另做完整模型对照。

`GDN` 默认使用 `inverse_method="block"`；`doubling` 只作为已知不稳定的实验对照，`triangular` 是独立候选。`compiler.py` 的语义 solve lowering 仍使用旧 doubling，不纳入真实模型训练支持。

`unit_lower_solve` 注册仍未提供训练 autograd；本轮训练与 compiled 使用 `GDN(semantic_solve=False)` 的可微组合运算。Triton decode 仍是推理专用。

## 验证与复跑

下载对应 revision 的公开模型文件后先核对 lock（文件名与 hash 必须一致；HF 镜像不保证相同序列化 hash）。从公开独立仓库根目录执行：

```bash
python tools/verify_training_assets.py --model /path/to/Qwen3.5-2B
python -m deltaforge.cpu_gate --output /tmp/deltaforge-cpu.json
python -m deltaforge.training_probe --phase gate \
  --model /path/to/Qwen3.5-2B --data training-assets/data.json \
  --output /tmp/deltaforge-model-gate
python -m deltaforge.training_probe --phase kernel --output /tmp/deltaforge-kernel
python -m deltaforge.training_probe --phase train --backend deltaforge --repetition 0 \
  --model /path/to/Qwen3.5-2B --data training-assets/data.json \
  --output /tmp/deltaforge-training
```

14 项 CPU 测试覆盖数学、double 梯度、state/padding/GQA、scale、真实 Qwen 层输出与参数梯度、checkpoint 和异常恢复。开发机另有 `/proc/.../comm` 沙箱限制，断言与进程退出分别记录；第三轮 worker 已在安装 FLA 前以正常 exit 0 复核。

`--phase gate` 与 `--phase train` 是独立命令。手动复跑时先检查目标后端的 `gate_passed`，合格后再调用 train；不能把训练命令存在或退出成功作为数值门禁通过的证据。上述 `report_training.py` 重算的是仓库中保存的终态证据，新的输出需单独核验。

模型固定为公开 Qwen/Qwen3.5-2B，文件 revision/hash 见 `training-assets/lock.json`。数据为 ESCI 官方来源的 12 train＋4 validation，源 split hash 和 query 隔离已验证；仅用于 causal-LM 后端兼容性探针。

训练为 BF16、SDPA、LoRA rank 8、AdamW、非重入 checkpoint、batch 1、256-token bucket。每个后端 10 更新，2 warmup＋8 计时；两个独立进程重复，第二轮反转顺序。padding 不计入 loss／有效 token。加载必须没有 missing/unexpected/mismatched/error。

阈值预先固定：kernel output relative L2≤5%、kernel/model 梯度 relative L2≤8%、model logits relative L2≤3%、loss absolute difference≤0.05。保留原始误差；通过不等于逐元素一致。saved tensor 字节是 hooks 逻辑体积之和，不等于去重后的激活显存。

完整 step 包含 forward、loss、backward、clip、AdamW 和同步；首次 forward/backward、编译、warmup 和稳态分别记录。4 条 validation loss 不足以推出通用／业务质量或全参数训练结论。

小任务、失败、比选和 timeline 见 [小周报](progress/WEEKLY_2026-10-25.md)；阶段判据见 [迁移方案](MAINSTREAM_MIGRATION.md)。M1 的锁版本共同契约／随机强基线已完成；M2 对 torch oracle 的固定 LoRA 子范围通过，原生默认比较尚未闭合，M2 总体部分完成。公开证据可用 `python tools/report_training.py` 重新校验汇总。
