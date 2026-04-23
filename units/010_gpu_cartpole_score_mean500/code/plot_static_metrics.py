from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    df = pd.read_csv("../plot/static_metrics.csv")
    out = Path("../plot/static_curve.svg")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(df["elapsed_s"], df["best"], label="best reward", color="#32746d")
    ax.plot(df["elapsed_s"], df["mean"], label="mean reward", color="#b45f06")
    ax.axhline(500.0, color="black", linestyle="--", alpha=0.5)
    ax.set_title("Unit 10 Static Networks on GPU-Native CartPole")
    ax.set_xlabel("elapsed seconds")
    ax.set_ylabel("reward")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
