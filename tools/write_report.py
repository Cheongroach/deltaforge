# SPDX-License-Identifier: Apache-2.0
"""Rebuild the historical report template separately from edited reading docs."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / ".doc-build/legacy",
                        help="Historical template output; does not replace edited docs")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output == ROOT.resolve():
        parser.error("Choose a separate output directory to preserve edited documents")
    (output / "artifacts").mkdir(parents=True, exist_ok=True)
    summaries = json.loads((ROOT/'artifacts/summary.json').read_text())
    events = json.loads((ROOT/'artifacts/measurements.json').read_text())
    index = json.loads((ROOT/'artifacts/run-index.json').read_text())
    prefill = [r for r in summaries if r['kind']=='prefill']
    decode = [r for r in summaries if r['kind']=='decode']
    assert len(prefill)==6 and len(decode)==3
    lines = ['# DeltaForge 双硬件实测报告', '', '实测日期：2026-10-25。工作负载为固定种子随机算子夹具，锚定 Qwen3.5-2B GDN 维度；未加载完整模型权重。', '',
        '## 已完成结果', '',
        '同一修复版本在 H20 / Inductor 与 910B4 / TorchAir 上各完成 3 组 fullgraph prefill 编译与数值验收；CUDA Triton decode 完成 24 组配置测量、GQA / 非规则维度校验与连续 128 步状态交接。两端最终任务均须同时具备 done、Pod exit=0 和 benchmark_complete，不能用提交、预检成功或仅有终止事件替代编译验收。', '',
        '所有速度均以同一进程、相同 shape / dtype / state 的对照计算。prefill 对照为同 chunk 的 portable eager；decode 对照为本项目 PyTorch token recurrence。这不是 FLA / FlashQLA 领先性或整模型收益证据。', '',
        '## 环境与方法', '', '| 后端 | 实采环境 | 资源 |', '|---|---|---|',
        '| H20 | NVIDIA H20 / SM90；PyTorch 2.9.1；Triton 3.5.1；TF32 matmul 关闭 | 单卡、20 CPU、230400 MB |',
        '| Ascend | Ascend910B4-1；PyTorch 2.7.1+cpu；torch-npu 2.7.1；TorchAir 可导入 | 单卡、15 CPU、128000 MB |', '',
        'TorchAir 无可用 distribution version；本轮未收集精确 CANN / driver build，因此当前版本标识并不覆盖整个软件栈。NPU 镜像中的 Triton 包不证明 Triton-Ascend backend 可用。当前 NPU 实作为 PyTorch / TorchAir 图路径。', '',
        '输入 Q/K 先归一化；g∈[-0.15,0]，β∈[0,1]，初始 state 非零。prefill 运算及 state 为 FP32；decode 输入 Q/K/V 为 BF16、state 为 FP32。每格先数值后计时：4 次预热、7 组重复、每组 5 次调用；表中为 wall 中位数，原始 7 组 wall / device event 样本都保留。event 会包含提交空洞，不能当作纯 kernel compute。', '',
        '本轮属于同一进程内的微基准配置扫描，候选选择与报告共用这些样本。尚未进行多次独立运行的候选确认或统计显著性检验。', '',
        '## Prefill：同 chunk 的编译性能比', '',
        'B=1，K=V=128；Hk=Hv=H；三个形状分别 T=128,H=16 / T=512,H=16 / T=1024,H=4。H=4 是小 head 数夹具，不是已运行 TP 的证据。', '',
        '| 后端 | T / H | chunk | eager ms | compiled ms | 性能比 | 导出+编译 s | 回本调用数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in prefill:
        s=row['shape'];label='H20 / Inductor' if row['run'].startswith('h20') else '910B4 / TorchAir'
        lines.append(f"| {label} | {s['T']} / {s['Hv']} | {row['chunk_size']} | {row['baseline_ms']:.4f} | {row['optimized_ms']:.4f} | {row['speedup']:.3f}x | {row['compile_seconds']:.3f} | {row['break_even_calls'] if row['break_even_calls'] is not None else '不回本'} |")
    lines += ['', '910B4 的 T=512 编译延迟高于 eager；dispatch-plan.json 对该形状建议 eager。T=128 的收益仅约 5%，须独立确认后再采用。这是离线配置建议，尚未实现在线 dispatch。\n\n编译列是 export/lower 加上首次编译与执行，不是稳定 cache miss 均值。回本调用数=ceil(该成本×1000 / 每调用节省毫秒)，为本轮估算；未实现在线编译 cache。形状改变会重新特化，不能无条件复用本次收益。', '',
        '原生 triangular solve 另保留测量，但其 GPU / NPU 路径、融合和 dtype 策略不同。不能只选慢原生 solve 的巨大比值来代表相对成熟 GDN kernel 的收益。', '',
        '## Decode：CUDA Triton 融合与配置扫描', '',
        'T=1，Hk=Hv=16，K=V=128。候选 tile=8/16/32/64，warps=4/8；每个 B 测量 8 组候选。表中包括函数调用和输出/state 分配。', '',
        '| B | tile / warps | PyTorch us | Triton us | 加速 |', '|---:|---:|---:|---:|---:|']
    for row in decode:
        lines.append(f"| {row['shape']['B']} | {row['value_tile']} / {row['num_warps']} | {row['baseline_ms']*1000:.2f} | {row['optimized_ms']*1000:.2f} | {row['speedup']:.3f}x |")
    lines += ['', 'kernel 将 decay、KᵀS、误差、rank-1 更新和 QᵀS 融合，state 按 value tile 只读写一次。增大 tile 的并行度与 Q/K 冗余存在取舍。当前没有 nsys / ncu 采样，不能据延迟断言 HBM 饱和、具体 occupancy 或 Tensor Core 使用比例。', '',
        '## 数值与梯度', '',
        '两端最终容器均通过 8 项 CPU 回归（含 double gradcheck、完整六输入 autograd、state handoff 与 FX 改写）；设备分别用 FP32/BF16 各验证 7 组小形状/边界夹具。compiled prefill 的独立 oracle 在 CPU 按 token 执行。', '',
        'FP32 output 门禁 atol=5e-5、rtol=5e-4；BF16 output 为 atol=5e-3、rtol=1e-2；所有 FP32 state 统一 atol=5e-5、rtol=5e-4。混合容差不代表每个测试能覆盖全部输入分布。', '',
        '| 路径 | output max abs | state max abs |', '|---|---:|---:|']
    for run in ['h20-constants','npu-constants']:
        selected=[r for r in prefill if r['run']==run]
        lines.append(f"| {run} compiled prefill（3 形状最大值） | {max(r['correctness']['output']['max_abs'] for r in selected):.6g} | {max(r['correctness']['state']['max_abs'] for r in selected):.6g} |")
    handoff=next(r for r in events if r['run']=='h20-constants' and r['event']=='decode_handoff_validation')
    lines.append(f"| Triton 128 步 handoff | {handoff['correctness']['output']['max_abs']:.6g} | {handoff['correctness']['state']['max_abs']:.6g} |")
    lines += ['', '连续 handoff 检查 FP32 state 不累计 BF16 截断。非规则 decode 另检查 B=2、Hk=2、Hv=4、K=37、V=53。padding 的无效 token 作为恒等状态转移。纯 PyTorch backward 已验证；语义 custom op 和 Triton kernel 未实现融合训练 backward。', '',
        '## 编译失败与修复证据', '',
        '| 轮次 | 观察 | 修复或处置 |', '|---|---|---|',
        '| npu-initial | Dynamo Unsupported：generated Python 构造 NPU torch.device | 将 factory device 规范化为字符串 |',
        '| npu-device-fix | BackendCompilerFailed：aten.eye.default ge_converter is not implemented | 将固定 eye / ones 提升为图外 target-device buffer |',
        '| h20 / npu-stage-blocked | 只有设备 probe；execute-script 需要缺失的 WebShell SSO cookie | 停止这两项等待传源码的任务，改为入口携带完整源码 |',
        '| h20 / npu-constants | 实机编译后重新数值验收与计时 | 不把 eager fallback 记作 compiled |', '',
        'FX 只匹配领域 unit_lower_solve；GDN 构造严格下三角 L，确保 L^C=0。有限倍增是实数域精确改写，浮点误差仍需测量。当前每个 prefill 图降低 1 个 batched domain op，并提升 2 个固定 factory 常量。CPU 小形状示例的 before/after FX 图位于 artifacts/ir-example/；它不是设备生成 kernel / GE IR 的导出。', '',
        '## 任务闭合与证据索引', '', '| 别名 | 状态 | 完整 benchmark | Pod exit |', '|---|---|---|---|']
    for row in index:
        lines.append(f"| {row['alias']} | {row['state']} | {row['benchmark_complete']} | {row['pod_exit_status']} |")
    lines += ['', '平台请求先核对 schema / dry-run，再将同一最终 payload 包装到 check-items。data.status=error 时停止；warn/hint 展示后可继续；成功 sid 保留式合并到 request_context.precheck_sid，并在 30 分钟内提交。所有本项目新任务均为单设备、3600 秒硬上限、不重启。已停止的只限本项目等待传输任务。', '',
        'run-index.json 仅保留实验别名、终态、退出码与资源信息；原始平台响应和签名 URL 留在私有临时目录。每个真实 worker 的 source_sha256 与启动归档逐文件核对。首轮 H20 / NPU 和后续修复轮次的源码都保留，避免把不同代码版本混成一次实验。', '',
        '文件：[原始事件与快照](artifacts/README.md) · [聚合 JSON](artifacts/measurements.json) · [CSV](artifacts/measurements.csv) · [比较与回本 JSON](artifacts/summary.json)。', '',
        '## 实用边界与下一步', '',
        '已完成可复跑算子/编译原型。NPU 尚未手写 AscendC kernel；未做多卡、packed serving、动态 cache、fused backward 或完整 Qwen 模型接入。下一阶段先引入锁版本 FLA / FlashQLA 基线及 profiler，再做单层和整模型误差 / TTFT / TPOT / 显存测试。算子速度不能直接写成任务吞吐或模型推理加速。', '']
    (output/'RESULTS.md').write_text('\n'.join(lines))
    counts={'events':len(events),'measurements':sum(r['event']=='measurement' for r in events),'runs':len(index)}
    (output/'artifacts/README.md').write_text(f'''# 实测证据索引

{counts['runs']} 轮任务记录、{counts['events']} 条算子事件、{counts['measurements']} 格测量；每格保留 7 组 wall 与 device event 样本。

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
''')
    print('Historical template created in', output)


if __name__=='__main__':
    main()
