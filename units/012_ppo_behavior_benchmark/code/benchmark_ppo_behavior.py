from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

UNIT8_CODE = Path(__file__).resolve().parents[2] / "008_torch_cartpole_physics_parity" / "code"
if str(UNIT8_CODE) not in sys.path:
    sys.path.insert(0, str(UNIT8_CODE))

from torch_cartpole import step_cartpole  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--time-budget-s", type=float, default=8.0)
    parser.add_argument("--num-episodes", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--base-seed", type=int, default=1000)
    parser.add_argument(
        "--target-agent-path",
        default=str(
            Path(__file__).resolve().parents[2]
            / "003_cartpole_sb3_ppo_action_imitation"
            / "model"
            / "ppo-CartPole-v1.zip"
        ),
    )
    parser.add_argument("--rollouts-path", default="../data/reference_rollouts.csv")
    parser.add_argument("--episode-metrics-path", default="../plot/episode_metrics.csv")
    parser.add_argument("--summary-path", default="../plot/summary.json")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def rollout_episode(
    model: PPO,
    seed: int,
    max_steps: int,
    device: torch.device,
    deadline: float,
) -> tuple[list[dict[str, float | int]], dict[str, float | int]] | None:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    state = torch.empty(1, 4, device=device, dtype=torch.float32).uniform_(-0.05, 0.05, generator=g)
    rows: list[dict[str, float | int]] = []
    actions: list[int] = []
    total_reward = 0.0

    for step in range(max_steps):
        if time.perf_counter() >= deadline:
            return None
        state_np = state.detach().cpu().numpy()
        action_arr, _ = model.predict(state_np, deterministic=True)
        action = int(np.asarray(action_arr).reshape(-1)[0])
        next_state, reward_t, terminated_t = step_cartpole(
            state,
            torch.tensor([action], dtype=torch.int64, device=device),
        )
        reward = float(reward_t.item())
        terminated = bool(terminated_t.item())
        row = {
            "seed": seed,
            "step": step,
            "x": float(state[0, 0].item()),
            "x_dot": float(state[0, 1].item()),
            "theta": float(state[0, 2].item()),
            "theta_dot": float(state[0, 3].item()),
            "action": action,
            "reward": reward,
            "done": int(terminated),
        }
        rows.append(row)
        actions.append(action)
        total_reward += reward
        state = next_state
        if terminated:
            break

    length = len(rows)
    switch_rate = 0.0
    if length > 1:
        switches = sum(int(a != b) for a, b in zip(actions[:-1], actions[1:]))
        switch_rate = switches / (length - 1)
    action_one_rate = float(sum(actions) / length) if length else 0.0
    episode = {
        "seed": seed,
        "return": total_reward,
        "length": length,
        "action_one_rate": action_one_rate,
        "action_switch_rate": switch_rate,
    }
    return rows, episode


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def build_summary(episode_rows: list[dict], rollout_rows: list[dict], requested_episodes: int, elapsed_s: float, device: torch.device) -> dict:
    returns = np.asarray([row["return"] for row in episode_rows], dtype=np.float64)
    lengths = np.asarray([row["length"] for row in episode_rows], dtype=np.float64)
    action_one_rates = np.asarray([row["action_one_rate"] for row in episode_rows], dtype=np.float64)
    action_switch_rates = np.asarray([row["action_switch_rate"] for row in episode_rows], dtype=np.float64)
    states = np.asarray(
        [[row["x"], row["x_dot"], row["theta"], row["theta_dot"]] for row in rollout_rows],
        dtype=np.float64,
    )
    actions = np.asarray([row["action"] for row in rollout_rows], dtype=np.float64)
    state_mean = states.mean(axis=0)
    state_std = states.std(axis=0)
    state_cov = np.cov(states, rowvar=False)
    max_length = int(lengths.max()) if len(lengths) else 0
    survival = []
    for step in range(max_length):
        alive = float(np.mean(lengths > step))
        survival.append({"step": step, "alive_fraction": alive})
    return {
        "episodes_completed": int(len(episode_rows)),
        "episodes_requested": int(requested_episodes),
        "elapsed_s": round(float(elapsed_s), 3),
        "device": str(device),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "action_one_rate_mean": float(action_one_rates.mean()),
        "action_one_rate_std": float(action_one_rates.std()),
        "action_switch_rate_mean": float(action_switch_rates.mean()),
        "action_switch_rate_std": float(action_switch_rates.std()),
        "global_action_one_rate": float(actions.mean()),
        "state_mean": state_mean.tolist(),
        "state_std": state_std.tolist(),
        "state_cov": state_cov.tolist(),
        "survival_curve": survival,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    model = PPO.load(args.target_agent_path, device="cpu")
    rollouts_path = Path(args.rollouts_path)
    episode_metrics_path = Path(args.episode_metrics_path)
    summary_path = Path(args.summary_path)
    start = time.perf_counter()
    deadline = start + args.time_budget_s

    rollout_rows: list[dict[str, float | int]] = []
    episode_rows: list[dict[str, float | int]] = []
    for episode_idx in range(args.num_episodes):
        seed = args.base_seed + episode_idx
        result = rollout_episode(model, seed, args.max_steps, device, deadline)
        if result is None:
            break
        rows, episode = result
        for row in rows:
            rollout_rows.append({"episode": episode_idx, **row})
        episode_rows.append(episode)
        write_csv(rollouts_path, rollout_rows)
        write_csv(episode_metrics_path, episode_rows)
        print(
            "episode={episode:03d} seed={seed} return={ret:.1f} length={length} action_one_rate={a1:.3f}".format(
                episode=episode_idx,
                seed=seed,
                ret=episode["return"],
                length=episode["length"],
                a1=episode["action_one_rate"],
            )
        )
    if not episode_rows:
        raise RuntimeError("Time budget expired before completing one PPO benchmark episode.")

    summary = build_summary(
        episode_rows=episode_rows,
        rollout_rows=rollout_rows,
        requested_episodes=args.num_episodes,
        elapsed_s=time.perf_counter() - start,
        device=device,
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "episodes={episodes_completed}/{episodes_requested} return_mean={return_mean:.2f} "
        "return_std={return_std:.2f} length_mean={length_mean:.2f} length_std={length_std:.2f}".format(**summary)
    )


if __name__ == "__main__":
    main()
