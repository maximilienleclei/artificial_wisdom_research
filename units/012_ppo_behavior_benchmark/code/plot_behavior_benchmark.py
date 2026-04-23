from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    unit_dir = Path(__file__).resolve().parents[1]
    episodes = pd.read_csv(unit_dir / "plot" / "episode_metrics.csv")
    summary = json.loads((unit_dir / "plot" / "summary.json").read_text())
    survival = pd.DataFrame(summary["survival_curve"])

    fig, axes = plt.subplots(3, 1, figsize=(8, 9))

    axes[0].plot(episodes["seed"], episodes["return"], color="#32746d")
    axes[0].axhline(summary["return_mean"], color="black", linestyle="--", alpha=0.45)
    axes[0].set_title("Unit 12 PPO Closed-Loop Behavior Benchmark")
    axes[0].set_ylabel("episode return")
    axes[0].grid(axis="y", linestyle=":", alpha=0.35)

    axes[1].plot(episodes["seed"], episodes["action_one_rate"], color="#b45f06", label="action=1 rate")
    axes[1].plot(episodes["seed"], episodes["action_switch_rate"], color="#4c5f99", label="action switch rate")
    axes[1].set_ylabel("action behavior")
    axes[1].grid(axis="y", linestyle=":", alpha=0.35)
    axes[1].legend()

    axes[2].plot(survival["step"], survival["alive_fraction"], color="#7f8c8d")
    axes[2].set_xlabel("step")
    axes[2].set_ylabel("alive fraction")
    axes[2].grid(axis="y", linestyle=":", alpha=0.35)

    fig.tight_layout()
    out = unit_dir / "plot" / "behavior_benchmark.svg"
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
