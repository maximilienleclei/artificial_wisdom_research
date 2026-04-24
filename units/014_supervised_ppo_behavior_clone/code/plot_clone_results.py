from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    unit_dir = Path(__file__).resolve().parents[1]
    comparison = json.loads((unit_dir / "plot" / "benchmark_comparison.json").read_text())
    episodes = pd.read_csv(unit_dir / "plot" / "episode_metrics.csv")
    history = pd.DataFrame(comparison["train_history"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 9))
    axes[0].plot(history["epoch"], history["train_accuracy"], color="#32746d")
    axes[0].set_title("Unit 13 Supervised PPO Behavior Clone")
    axes[0].set_ylabel("train accuracy")
    axes[0].grid(axis="y", linestyle=":", alpha=0.35)

    axes[1].plot(episodes["seed"], episodes["return"], color="#b45f06", label="clone return")
    axes[1].axhline(comparison["reference"]["return_mean"], color="black", linestyle="--", alpha=0.45, label="ppo return mean")
    axes[1].set_ylabel("episode return")
    axes[1].grid(axis="y", linestyle=":", alpha=0.35)
    axes[1].legend()

    axes[2].bar(
        ["return mean", "length mean", "action=1", "switch rate"],
        [
            comparison["deltas"]["return_mean_abs"],
            comparison["deltas"]["length_mean_abs"],
            comparison["deltas"]["action_one_rate_mean_abs"],
            comparison["deltas"]["action_switch_rate_mean_abs"],
        ],
        color=["#4c5f99", "#7f8c8d", "#32746d", "#b45f06"],
    )
    axes[2].set_ylabel("absolute delta vs PPO")
    axes[2].grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()
    out = unit_dir / "plot" / "clone_results.svg"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
