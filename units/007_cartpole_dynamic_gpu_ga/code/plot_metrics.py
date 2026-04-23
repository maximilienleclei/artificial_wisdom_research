from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    df = pd.read_csv("../plot/metrics.csv")
    output_path = Path("../plot/curve.svg")

    fig, ax_reward = plt.subplots(figsize=(8, 4.5))
    ax_hidden = ax_reward.twinx()

    ax_reward.plot(df["generation"], df["best"], label="best reward", color="#32746d")
    ax_reward.plot(df["generation"], df["mean"], label="mean reward", color="#b45f06")
    ax_hidden.plot(df["generation"], df["mean_hidden"], label="mean hidden nodes", color="#4c5f99", alpha=0.75)

    ax_reward.set_title("GPU-Batched Dynamic Network GA on CartPole-v1")
    ax_reward.set_xlabel("generation")
    ax_reward.set_ylabel("reward")
    ax_hidden.set_ylabel("mean hidden nodes")
    ax_reward.grid(axis="y", linestyle=":", alpha=0.35)

    lines = ax_reward.get_lines() + ax_hidden.get_lines()
    labels = [line.get_label() for line in lines]
    ax_reward.legend(lines, labels, loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
