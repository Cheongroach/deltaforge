# SPDX-License-Identifier: Apache-2.0
"""Render documentation figures from terminal evidence, without rerunning a model."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


COLORS = {
    "torch": "#697381",
    "deltaforge": "#E69F00",
    "compiled": "#0072B2",
    "default": "#9CA3AF",
    "fla": "#AA5FA6",
}
LABELS = {
    "torch": "torch 参考",
    "deltaforge": "DeltaForge\n分块",
    "compiled": "DeltaForge\n编译",
    "default": "框架默认",
    "fla": "FLA",
}


def load(path):
    return json.loads(path.read_text())


def save(fig, output, name):
    fig.savefig(output / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    svg = output / f"{name}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    # Matplotlib emits whitespace at path line ends; normalize without changing geometry.
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def clean(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#E5E7EB", linewidth=0.7, zorder=0)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path)
    parser.add_argument("--font", type=Path)
    args = parser.parse_args()
    root = args.root
    output = args.output or root / "figures"
    output.mkdir(parents=True, exist_ok=True)
    font = args.font or root / "profiling-lab/_shared/fonts/wqy-microhei.ttc"
    font_manager.fontManager.addfont(str(font))
    family = font_manager.FontProperties(fname=str(font)).get_name()
    plt.rcParams.update({
        "font.family": family, "font.size": 11, "axes.titlesize": 12,
        "axes.unicode_minus": False, "svg.fonttype": "path",
        "svg.hashsalt": "deltaforge-evidence", "figure.dpi": 120,
    })
    evidence = root / "training-results/attempt-03/evidence"
    paths = [evidence / "model-gate/model-gates.json", root / "training-assets/lock.json"]
    gates = load(paths[0])
    protocol = load(paths[1])["protocol"]
    comparisons = [r for r in gates if r["backend"] != "torch"]
    names = [r["backend"] for r in comparisons]

    # Insight 1: close model outputs do not establish matching gradients.
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), sharey=True)
    fig.subplots_adjust(left=0.22, right=0.98, bottom=0.18, top=0.72, wspace=0.22)
    fig.suptitle("模型输出接近，梯度仍可能不合格", fontsize=16, y=0.98)
    fig.text(0.5, 0.88, "Qwen3.5-2B · 单 H20 · BF16 · 同一次参数初始化与输入 · 对照 torch 参考",
             ha="center", fontsize=10)
    specifications = [
        ("模型打分误差", "logits", protocol["model_logits_relative_l2_max"] * 100, 3.7),
        ("参数梯度误差", "parameter_gradient_relative_l2",
         protocol["gradient_relative_l2_max"] * 100, 27),
    ]
    values = {}
    for ax, (title, key, threshold, limit) in zip(axes, specifications):
        errors = [
            (r["checks"]["logits"]["relative_l2"] if key == "logits" else r["checks"][key]) * 100
            for r in comparisons
        ]
        values[key] = dict(zip(names, errors))
        ax.barh(range(len(names)), errors, color=[COLORS[n] for n in names], height=0.52)
        for y, value in enumerate(errors):
            ax.text(value + limit * 0.022, y, f"{value:.2f}%", va="center", fontsize=10,
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
        ax.axvline(threshold, color="#B32632", linestyle="--", linewidth=1.3)
        ax.text(threshold, 1.025, f"预设上限 {threshold:g}%", color="#B32632",
                transform=ax.get_xaxis_transform(), ha="center", fontsize=10)
        ax.set_xlim(0, limit)
        ax.set_title(title, pad=29)
        ax.set_xlabel("relative L2（%）")
        clean(ax)
    axes[0].set_yticks(range(len(names)), [LABELS[n].replace("\n", " ") for n in names])
    axes[0].invert_yaxis()
    fig.text(0.5, 0.025, "误差按相对范数聚合；虚线是预设门禁，不是置信区间。只有梯度也合格的后端进入训练。",
             ha="center", fontsize=9)
    save(fig, output, "model-checks")

    # Insight 2: report all eight measured steps in both same-seed processes.
    training = {}
    medians = {}
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.8), sharey=True)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.23, top=0.72, wspace=0.16)
    fig.suptitle("固定 LoRA 任务的完整训练步耗时", fontsize=16, y=0.98)
    fig.text(0.5, 0.89, "Qwen3.5-2B · 单 H20 · BF16 · batch=1 · 256-token bucket · 相同 seed",
             ha="center", fontsize=10)
    backends = ["torch", "deltaforge", "compiled"]
    for repetition, ax in enumerate(axes):
        for x, backend in enumerate(backends):
            path = evidence / f"train-{backend}-{repetition}/training-{backend}-{repetition}.json"
            paths.append(path)
            row = load(path)
            steps = [s for s in row["steps"] if not s["warmup"]]
            seconds = [s["seconds"] for s in steps]
            assert len(steps) == 8
            assert sum(s["effective_tokens"] for s in steps) == 1552
            median = statistics.median(seconds)
            training[f"{backend}-{repetition}"] = seconds
            medians[f"{backend}-{repetition}"] = median
            ax.scatter(x + np.linspace(-0.12, 0.12, 8), seconds,
                       s=32, color=COLORS[backend], alpha=0.85, zorder=3)
            ax.plot([x - 0.18, x + 0.18], [median, median], color="#111827", linewidth=1.6)
            ax.text(x + 0.2, median + 0.08, f"{median:.3f}s", fontsize=9)
        ax.set_title(f"独立进程 {repetition + 1}")
        ax.set_xticks(range(3), [LABELS[b] for b in backends])
        ax.set_xlim(-0.4, 2.65)
        ax.set_ylim(0, 2.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#E5E7EB", linewidth=0.7)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("一个完整 step 的耗时（秒）")
    fig.text(0.5, 0.025,
             "点：每次进程的 8 个计时 step；短横线：中位数。已排除 2 个 warmup 更新，进程共享磁盘编译缓存。",
             ha="center", fontsize=9)
    save(fig, output, "training-steps")

    # Insight 3: the mature FLA kernel is faster on the random T=2048 fixture.
    kernels = []
    for repetition in range(2):
        path = evidence / f"kernel-{repetition}/kernel-results.json"
        paths.append(path)
        for row in load(path):
            if row["length"] != 2048:
                continue
            assert row["gate_passed"] and len(row["samples"]) == 10
            paired = [s["forward_ms"] + s["backward_ms"] for s in row["samples"]]
            kernels.append({"backend": row["backend"], "repetition": repetition,
                            "paired_median_ms": statistics.median(paired), "samples": paired})
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    fig.subplots_adjust(left=0.23, right=0.96, bottom=0.21, top=0.70)
    fig.suptitle("随机 GDN 算子：FLA 耗时更短", fontsize=16, y=0.98)
    fig.text(0.5, 0.88, "单 H20 · BF16 输入 / FP32 state · B=1, T=2048, H=16, K=V=128",
             ha="center", fontsize=10)
    kernel_backends = backends + ["fla"]
    for y, backend in enumerate(kernel_backends):
        points = [r["paired_median_ms"] for r in kernels if r["backend"] == backend]
        positions = [y - 0.07, y + 0.07]
        ax.plot(points, positions, color=COLORS[backend], linewidth=1.5)
        for repetition, value in enumerate(points):
            ax.scatter(value, positions[repetition], color=COLORS[backend], s=55,
                       marker="o" if repetition == 0 else "^", zorder=3)
        ax.text(max(points) + 3, y, f"{min(points):.2f}–{max(points):.2f} ms",
                va="center", fontsize=10)
    ax.set_yticks(range(4), [LABELS[b].replace("\n", " ") for b in kernel_backends])
    ax.invert_yaxis()
    ax.set_xlim(0, 116)
    ax.set_xlabel("前向＋反向耗时（毫秒，配对样本的中位数）")
    clean(ax)
    ax.legend(handles=[
        Line2D([], [], marker="o", color="#333333", linestyle="None", label="进程 1"),
        Line2D([], [], marker="^", color="#333333", linestyle="None", label="进程 2"),
    ], loc="lower right", frameon=False)
    fig.text(0.5, 0.025, "每个点来自一个进程的 10 次计时；连线仅连接两个中位数，不表示置信区间。输入为随机夹具。",
             ha="center", fontsize=9)
    save(fig, output, "kernel-baseline")

    ratios = [medians[f"torch-{i}"] / medians[f"compiled-{i}"] for i in range(2)]
    assert round(min(ratios), 2) == 2.02 and round(max(ratios), 2) == 2.20
    metadata = {
        "scope": "Replot existing terminal evidence; no new GPU experiment",
        "seed_repetitions": "two processes, same seed, shared disk caches; not multi-seed",
        "sources_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                           for p in paths},
        "model_error_percent": values, "training_step_seconds": training,
        "training_step_medians_seconds": medians, "compiled_speedup_vs_torch": ratios,
        "kernel_t2048_paired_samples": kernels,
        "figure_meanings": {
            "model-checks": "Forward similarity does not establish gradient compatibility",
            "training-steps": "Measured step samples, including observed tails",
            "kernel-baseline": "FLA is faster on this random single-operator workload",
        },
    }
    (output / "evidence.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "figures": 3, "compiled_speedup": ratios,
                      "kernel_paired_medians": [{k: v for k, v in row.items() if k != "samples"}
                                                for row in kernels]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
