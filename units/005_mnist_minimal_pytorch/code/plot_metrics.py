import json
from pathlib import Path

import matplotlib.pyplot as plt


def load_metrics(path: Path) -> dict:
    return json.loads(path.read_text())


def main() -> None:
    model_dir = Path("../model")
    plot_dir = Path("../plot")
    plot_dir.mkdir(parents=True, exist_ok=True)

    comparisons = [
        (
            "Linear smoke",
            load_metrics(model_dir / "metrics.json"),
            load_metrics(model_dir / "metrics_cuda_smoke.json"),
        ),
        (
            "Deep MLP",
            load_metrics(model_dir / "metrics_deep_mlp_cpu.json"),
            load_metrics(model_dir / "metrics_deep_mlp_amd_gpu.json"),
        ),
    ]

    labels = [name for name, _, _ in comparisons]
    cpu_throughput = [cpu["examples_per_s"] for _, cpu, _ in comparisons]
    gpu_throughput = [gpu["examples_per_s"] for _, _, gpu in comparisons]
    speedups = [gpu / cpu for cpu, gpu in zip(cpu_throughput, gpu_throughput, strict=True)]

    x = range(len(comparisons))
    width = 0.35
    colors = {"cpu": "#32746d", "gpu": "#b45f06", "speedup": "#4c5f99"}

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    axes[0].bar([i - width / 2 for i in x], cpu_throughput, width, label="CPU", color=colors["cpu"])
    axes[0].bar([i + width / 2 for i in x], gpu_throughput, width, label="AMD GPU", color=colors["gpu"])
    axes[0].set_title("Throughput")
    axes[0].set_ylabel("examples/sec")
    axes[0].set_xticks(list(x), labels)
    axes[0].legend()

    axes[1].bar(labels, speedups, color=colors["speedup"])
    axes[1].axhline(1.0, color="black", linewidth=1, alpha=0.5)
    axes[1].set_title("GPU Speedup")
    axes[1].set_ylabel("GPU / CPU throughput")

    for ax in axes:
        ax.grid(axis="y", linestyle=":", alpha=0.35)

    for idx, value in enumerate(cpu_throughput):
        axes[0].text(idx - width / 2, value, f"{value:g}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(gpu_throughput):
        axes[0].text(idx + width / 2, value, f"{value:g}", ha="center", va="bottom", fontsize=8)
    for idx, value in enumerate(speedups):
        axes[1].text(idx, value, f"{value:.2f}x", ha="center", va="bottom")

    fig.suptitle("MNIST CPU vs AMD GPU Benchmarks")
    fig.tight_layout()
    output_path = plot_dir / "cpu_vs_amd_gpu.svg"
    fig.savefig(output_path)
    print(f"Saved {output_path}")

    learning = load_metrics(model_dir / "metrics_deep_mlp_learning_amd_gpu.json")
    history = learning["history"]
    epochs = [row["epoch"] for row in history]
    train_loss = [row["train_loss"] for row in history]
    val_loss = [row["val_loss"] for row in history]
    val_accuracy = [row["val_accuracy"] for row in history]

    fig, ax_loss = plt.subplots(figsize=(7.5, 4.2))
    ax_acc = ax_loss.twinx()

    ax_loss.plot(epochs, train_loss, marker="o", label="train loss", color="#32746d")
    ax_loss.plot(epochs, val_loss, marker="o", label="val loss", color="#b45f06")
    ax_acc.plot(epochs, val_accuracy, marker="s", label="val accuracy", color="#4c5f99")

    ax_loss.set_title("Deep MLP Learning Curve on AMD GPU")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_acc.set_ylabel("validation accuracy")
    ax_loss.set_xticks(epochs)
    ax_loss.grid(axis="y", linestyle=":", alpha=0.35)

    lines = ax_loss.get_lines() + ax_acc.get_lines()
    labels = [line.get_label() for line in lines]
    ax_loss.legend(lines, labels, loc="center right")

    fig.tight_layout()
    output_path = plot_dir / "deep_mlp_learning_curve.svg"
    fig.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
