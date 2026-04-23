from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from torch_cartpole import step_cartpole


def run_gym_rollout(initial_state: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    env = gym.make("CartPole-v1")
    try:
        env.reset(seed=0)
        base_env = env.unwrapped
        base_env.state = initial_state.astype(np.float64).copy()
        base_env.steps_beyond_terminated = None
        states = []
        for action in actions:
            _, _, terminated, truncated, _ = env.step(int(action))
            states.append(base_env.state.astype(np.float64).copy())
            if terminated or truncated:
                break
        return np.stack(states), actions[: len(states)]
    finally:
        env.close()


def run_torch_rollout(
    initial_state: np.ndarray,
    actions: np.ndarray,
    device: torch.device,
    dtype: torch.dtype,
) -> np.ndarray:
    state = torch.as_tensor(initial_state, dtype=dtype, device=device).unsqueeze(0)
    states = []
    with torch.no_grad():
        for action in actions:
            action_tensor = torch.tensor([int(action)], device=device)
            state, _, _ = step_cartpole(state, action_tensor)
            states.append(state.squeeze(0).detach().cpu().to(torch.float64).numpy())
    return np.stack(states)


def parity(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)
    device = torch.device(args.device)
    plot_dir = Path("../plot")
    plot_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    candidate_names = ["torch_cpu_float64", "torch_gpu_float64", "torch_gpu_float32"]
    max_abs_by_step = {name: np.zeros(args.steps) for name in candidate_names}
    step_counts = np.zeros(args.steps, dtype=np.int64)
    start = time.perf_counter()

    for rollout in range(args.rollouts):
        initial_state = rng.uniform(low=-0.05, high=0.05, size=(4,))
        actions = rng.integers(0, 2, size=args.steps)
        gym_states, valid_actions = run_gym_rollout(initial_state, actions)
        step_counts[: len(valid_actions)] += 1
        candidates = {
            "torch_cpu_float64": run_torch_rollout(initial_state, valid_actions, torch.device("cpu"), torch.float64),
            "torch_gpu_float64": run_torch_rollout(initial_state, valid_actions, device, torch.float64),
            "torch_gpu_float32": run_torch_rollout(initial_state, valid_actions, device, torch.float32),
        }
        for name, states in candidates.items():
            abs_err = np.abs(states - gym_states)
            step_err = abs_err.max(axis=1)
            max_abs_by_step[name][: len(step_err)] = np.maximum(max_abs_by_step[name][: len(step_err)], step_err)
            rows.append(
                {
                    "candidate": name,
                    "rollout": rollout,
                    "compared_steps": len(valid_actions),
                    "max_abs_error": float(abs_err.max()),
                    "final_abs_error": float(step_err[-1]),
                    "mean_abs_error": float(abs_err.mean()),
                }
            )

    metrics_path = plot_dir / "metrics.csv"
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    for name in max_abs_by_step:
        candidate_rows = [row for row in rows if row["candidate"] == name]
        summary_rows.append(
            {
                "candidate": name,
                "max_abs_error": max(row["max_abs_error"] for row in candidate_rows),
                "max_final_abs_error": max(row["final_abs_error"] for row in candidate_rows),
                "mean_abs_error": float(np.mean([row["mean_abs_error"] for row in candidate_rows])),
                "min_compared_steps": min(row["compared_steps"] for row in candidate_rows),
                "max_compared_steps": max(row["compared_steps"] for row in candidate_rows),
                "elapsed_s": round(time.perf_counter() - start, 3),
            }
        )

    summary_path = plot_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in max_abs_by_step.items():
        valid = step_counts > 0
        ax.plot(np.arange(1, args.steps + 1)[valid], values[valid], label=name)
    ax.set_yscale("log")
    ax.set_title("Torch CartPole Physics Drift vs Gymnasium")
    ax.set_xlabel("step")
    ax.set_ylabel("max absolute state error")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    output_path = plot_dir / "parity_drift.svg"
    fig.savefig(output_path)

    print(f"Saved {metrics_path}")
    print(f"Saved {summary_path}")
    print(f"Saved {output_path}")
    for row in summary_rows:
        print(
            "{candidate}: max_abs_error={max_abs_error:.3e} "
            "max_final_abs_error={max_final_abs_error:.3e} mean_abs_error={mean_abs_error:.3e}".format(**row)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollouts", type=int, default=128)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    return parser.parse_args()


if __name__ == "__main__":
    parity(parse_args())
