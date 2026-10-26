# 怎样自己更换后端并复跑

这篇按“准备环境 → 核对输入 → 检查数值 → 训练”的顺序说明操作。完整接口和限制见 [TRAINING_INTEGRATION.md](TRAINING_INTEGRATION.md)。命令从独立 GitHub 仓库根执行，该目录应包含 `pyproject.toml`、`deltaforge/` 和 `tools/`。

## 环境要与什么匹配

CPU 检查可以验证数学和接口，真实训练需要 GPU。本轮设备结果来自单张 NVIDIA H20、PyTorch 2.9.1、Triton 3.5.1。其他 GPU、软件版本和 NPU 不自动继承这份验收结果。

先使用匹配设备的 PyTorch 环境，再安装项目。`--no-deps` 表示不让项目安装过程自行更换底层依赖；后面的固定版本是本轮模型接入配置。

```bash
python -m pip install --no-deps -e .
python -m pip install "transformers==5.2.0" "peft==0.18.1" \
  "accelerate==1.11.0" "safetensors==0.8.0"
```

只有要复跑 FLA 对照时，才需要匹配 CUDA／Triton 的可选依赖：

```bash
python -m pip install --no-deps "fla-core==0.4.1" "einops==0.8.1"
```

这些命令不会自动准备 GPU 驱动、CANN 或模型文件。本轮安装验证复用了基础运行依赖，不是从空白机器重建整个软件栈。

## 为什么先核对模型与数据

比较实现之前，要确保模型权重、tokenizer 和数据相同。SHA256 是文件内容的指纹；相同名字不保证内容一致。[lock.json](../training-assets/lock.json) 保存本轮版本、来源和文件指纹。

把命令中的模型路径换成自己的模型目录，再运行：

```bash
python tools/verify_training_assets.py --model /path/to/Qwen3.5-2B
python -m deltaforge.cpu_gate --output ./cpu-gates.json
```

第一条检查九个模型文件及锁定数据，第二条运行 14 项 CPU 检查。CPU 通过说明测试覆盖的数学、状态、梯度和接口行为成立；GPU、真实模型和性能还要继续测。

## 数值检查与训练为什么分开

一个后端可能能运行、loss 看起来也正常，但参数梯度已经偏离参考。本轮先比较整模型输出打分、全部可训练参数梯度和损失，合格后才训练。

```bash
python -m deltaforge.training_probe --phase gate \
  --model /path/to/Qwen3.5-2B --data training-assets/data.json \
  --output ./model-gate
```

检查生成的 JSON 中每个后端的 `gate_passed`。本轮预设上限为 logits relative L2 3%、梯度 relative L2 8%、loss 绝对差 0.05；不要为了继续训练而改大阈值。gate 报告保留失败后端，不能只看命令退出码。

只有目标后端通过后，才手动执行：

```bash
python -m deltaforge.training_probe --phase train \
  --backend deltaforge --repetition 0 \
  --model /path/to/Qwen3.5-2B --data training-assets/data.json \
  --output ./training-deltaforge-0
```

这是独立调用；前面的命令不会替你阻止误用失败后端。`--backend compiled` 可复跑编译路径。要做完整对照，应按技术报告固定初始化、样本顺序、预算和重复编号，为各次运行保留不同输出目录。

## 放进自己的训练代码时，替换了什么

adapter 是把模型接口接到后端的连接层。下面的 context manager 只在代码块内替换 GDN callable，退出后恢复原实现，包括发生异常时：

```python
from deltaforge.backends import use_backend

optimizer.zero_grad()
with use_backend(model, "deltaforge", chunk_size=32) as selection:
    loss = model(**batch, use_cache=False).loss
    loss.backward()
    optimizer.step()
    print(selection.stats())
```

卷积、投影、普通 attention、模型参数和优化器仍由原框架管理。这个示例展示替换接口；它不负责下载模型、建立 optimizer 或运行完整训练配置。

state 是 GDN 传给下一段 token 的记忆。训练接口返回新 state，并保留跨段梯度；不能把推理专用、没有训练 backward 的 Triton decode 直接放进训练。

## 第一次复跑要保留什么

保留版本、资产 hash、数值 JSON、训练 step 样本、参数 hash、有效 token 数和显存。先确认输出与梯度合格、参数确实更新，再比较完整 step 的时间。

编译路径的第一次执行可能很慢，第二次可能复用缓存。记录首次成本和预热，不要把一个方案的首次成本删除，却留在另一个方案里。两个同 seed 进程也不等于多 seed 实验。

本轮只验收固定 Qwen3.5-2B／Transformers 5.2.0／LoRA／256 长度路径，packed 输入、动态形状承诺、全参数和多卡训练都还未完成。结果解释见 [训练结果阅读版](TRAINING_RESULTS-friendly.md)。
