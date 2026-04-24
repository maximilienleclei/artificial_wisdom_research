from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3 import PPO
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

UNIT8_CODE = Path(__file__).resolve().parents[2] / "008_torch_cartpole_physics_parity" / "code"
UNIT_DIR = Path(__file__).resolve().parents[1]
UNIT12_DIR = Path(__file__).resolve().parents[2] / "012_ppo_behavior_benchmark"
UNIT3_MODEL_PATH = (
    Path(__file__).resolve().parents[2]
    / "003_cartpole_sb3_ppo_action_imitation"
    / "model"
    / "ppo-CartPole-v1.zip"
)
if str(UNIT8_CODE) not in sys.path:
    sys.path.insert(0, str(UNIT8_CODE))

from torch_cartpole import step_cartpole  # noqa: E402


class ClonePolicy(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def parameter_count(hidden_size: int) -> int:
    return (4 * hidden_size + hidden_size) + (hidden_size * hidden_size + hidden_size) + (hidden_size * 2 + 2)


def forward_flops_per_example(hidden_size: int) -> int:
    return 2 * ((4 * hidden_size) + (hidden_size * hidden_size) + (hidden_size * 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--time-budget-s", type=float, default=6.0)
    parser.add_argument("--reuse-dataset", action="store_true")
    parser.add_argument("--train-episodes", type=int, default=32)
    parser.add_argument("--train-base-seed", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--min-lr-scale", type=float, default=0.05)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--collect-fraction", type=float, default=0.1)
    parser.add_argument("--val-interval-s", type=float, default=30.0)
    parser.add_argument("--eval-summary-path", default=str(UNIT12_DIR / "plot" / "summary.json"))
    parser.add_argument("--eval-episodes-path", default=str(UNIT12_DIR / "plot" / "episode_metrics.csv"))
    parser.add_argument("--target-agent-path", default=str(UNIT3_MODEL_PATH))
    parser.add_argument("--dataset-path", default=str(UNIT_DIR / "data" / "ppo_train_dataset.csv"))
    parser.add_argument("--model-path", default=str(UNIT_DIR / "model" / "clone_policy.pt"))
    parser.add_argument("--metrics-path", default=str(UNIT_DIR / "model" / "metrics.json"))
    parser.add_argument("--comparison-path", default=str(UNIT_DIR / "plot" / "benchmark_comparison.json"))
    parser.add_argument("--episode-metrics-path", default=str(UNIT_DIR / "plot" / "episode_metrics.csv"))
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def summarize_partial_history(history: list[dict]) -> dict:
    if not history:
        return {"epochs_completed": 0}
    return {
        "epochs_completed": int(len(history)),
        "latest_train_loss": float(history[-1]["train_loss"]),
        "latest_train_accuracy": float(history[-1]["train_accuracy"]),
        "latest_val_loss": float(history[-1]["val_loss"]),
        "latest_val_accuracy": float(history[-1]["val_accuracy"]),
        "best_val_accuracy": float(max(row["val_accuracy"] for row in history)),
        "elapsed_s": float(history[-1]["elapsed_s"]),
    }


def write_progress_snapshot(metrics_path: Path, comparison_path: Path, payload: dict) -> None:
    write_json(metrics_path, payload)
    write_json(comparison_path, payload)


def scheduled_lr_scale(progress: float, warmup_fraction: float, min_lr_scale: float) -> float:
    clamped = min(max(progress, 0.0), 1.0)
    if warmup_fraction > 0.0 and clamped < warmup_fraction:
        return max(min_lr_scale, clamped / warmup_fraction)
    if warmup_fraction >= 1.0:
        return 1.0
    cosine_progress = (clamped - warmup_fraction) / max(1e-8, 1.0 - warmup_fraction)
    cosine_value = 0.5 * (1.0 + np.cos(np.pi * cosine_progress))
    return min_lr_scale + (1.0 - min_lr_scale) * cosine_value


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def collect_training_data(
    model: PPO,
    train_episodes: int,
    train_base_seed: int,
    max_steps: int,
    device: torch.device,
    deadline: float,
) -> list[dict]:
    rows: list[dict] = []
    for episode_idx in range(train_episodes):
        if time.perf_counter() >= deadline:
            break
        seed = train_base_seed + episode_idx
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        state = torch.empty(1, 4, device=device, dtype=torch.float32).uniform_(-0.05, 0.05, generator=g)
        for step in range(max_steps):
            if time.perf_counter() >= deadline:
                break
            action_arr, _ = model.predict(state.detach().cpu().numpy(), deterministic=True)
            action = int(np.asarray(action_arr).reshape(-1)[0])
            rows.append(
                {
                    "episode": episode_idx,
                    "seed": seed,
                    "step": step,
                    "x": float(state[0, 0].item()),
                    "x_dot": float(state[0, 1].item()),
                    "theta": float(state[0, 2].item()),
                    "theta_dot": float(state[0, 3].item()),
                    "action": action,
                }
            )
            state, _, terminated = step_cartpole(state, torch.tensor([action], dtype=torch.int64, device=device))
            if bool(terminated.item()):
                break
    return rows


@torch.no_grad()
def evaluate_loader(
    policy: ClonePolicy,
    loader: DataLoader,
    device: torch.device,
    loss_fn: nn.Module,
) -> tuple[float, float]:
    policy.eval()
    total = 0
    correct = 0
    loss_total = 0.0
    for batch_x, batch_y in loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = policy(batch_x)
        loss = loss_fn(logits, batch_y)
        loss_total += float(loss.item()) * int(batch_y.numel())
        correct += int((logits.argmax(dim=1) == batch_y).sum().item())
        total += int(batch_y.numel())
    if total == 0:
        return 0.0, 0.0
    return correct / total, loss_total / total


def build_loaders(
    train_rows: list[dict],
    val_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    x = torch.tensor([[r["x"], r["x_dot"], r["theta"], r["theta_dot"]] for r in train_rows], dtype=torch.float32)
    y = torch.tensor([r["action"] for r in train_rows], dtype=torch.int64)
    total_count = int(y.shape[0])
    val_count = max(1, min(total_count - 1, int(total_count * val_fraction)))
    train_count = total_count - val_count
    train_x = x[:train_count]
    train_y = y[:train_count]
    val_x = x[train_count:]
    val_y = y[train_count:]
    return train_x, train_y, val_x, val_y, train_count, val_count


def train_for_slice(
    policy: ClonePolicy,
    optimizer: torch.optim.Optimizer,
    loss_fn: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    batch_size: int,
    lr: float,
    warmup_fraction: float,
    min_lr_scale: float,
    train_examples: int,
    val_examples: int,
    train_start: float,
    total_deadline: float,
    val_interval_s: float,
    metrics_path: Path,
    comparison_path: Path,
    model_path: Path,
    episode_metrics_path: Path,
    history: list[dict],
    best_val_accuracy: float,
    forward_example_evals: int,
    eval_seeds: list[int],
    max_steps: int,
    reference_summary: dict,
) -> tuple[list[dict], float, int, dict | None]:
    train_loader = DataLoader(TensorDataset(train_x, train_y), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_x, val_y), batch_size=batch_size, shuffle=False)
    next_val_at = train_start
    current_lr = lr
    latest_behavior: dict | None = None
    while time.perf_counter() < total_deadline:
        epoch_loss = 0.0
        correct = 0
        total = 0
        policy.train()
        for batch_x, batch_y in train_loader:
            if time.perf_counter() >= total_deadline:
                break
            train_progress = (time.perf_counter() - train_start) / max(1e-8, total_deadline - train_start)
            current_lr = lr * scheduled_lr_scale(train_progress, warmup_fraction, min_lr_scale)
            set_optimizer_lr(optimizer, current_lr)
            batch_x = batch_x.to(policy.net[0].weight.device)
            batch_y = batch_y.to(policy.net[0].weight.device)
            optimizer.zero_grad(set_to_none=True)
            logits = policy(batch_x)
            loss = loss_fn(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.item()) * int(batch_y.numel())
            correct += int((logits.argmax(dim=1) == batch_y).sum().item())
            total += int(batch_y.numel())
            forward_example_evals += int(batch_y.numel())
        if total == 0:
            break
        now = time.perf_counter()
        if now >= next_val_at:
            val_accuracy, val_loss = evaluate_loader(policy, val_loader, policy.net[0].weight.device, loss_fn)
            forward_example_evals += val_examples
            rollout_rows, episode_rows = evaluate_clone(
                policy=policy,
                eval_seeds=eval_seeds,
                max_steps=max_steps,
                device=policy.net[0].weight.device,
                deadline=total_deadline,
            )
            history_row = {
                "epoch": len(history) + 1,
                "train_loss": epoch_loss / total,
                "train_accuracy": correct / total,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "lr": current_lr,
                "elapsed_s": round(now - train_start, 3),
            }
            if episode_rows:
                write_csv(episode_metrics_path, episode_rows)
                clone_summary = summarize_rollouts(rollout_rows, episode_rows)
                latest_behavior = {
                    "episodes_completed": int(clone_summary["episodes_completed"]),
                    "return_mean": float(clone_summary["return_mean"]),
                    "return_std": float(clone_summary["return_std"]),
                    "action_one_rate_mean": float(clone_summary["action_one_rate_mean"]),
                    "action_switch_rate_mean": float(clone_summary["action_switch_rate_mean"]),
                    "return_mean_abs": abs(clone_summary["return_mean"] - reference_summary["return_mean"]),
                    "action_switch_rate_mean_abs": abs(
                        clone_summary["action_switch_rate_mean"] - reference_summary["action_switch_rate_mean"]
                    ),
                }
                history_row["closed_loop_episodes_completed"] = latest_behavior["episodes_completed"]
                history_row["closed_loop_return_mean"] = latest_behavior["return_mean"]
                history_row["closed_loop_switch_delta"] = latest_behavior["action_switch_rate_mean_abs"]
            history.append(history_row)
            if val_accuracy >= best_val_accuracy:
                best_val_accuracy = val_accuracy
                model_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(policy.state_dict(), model_path)
            write_progress_snapshot(
                metrics_path,
                comparison_path,
                {
                    "stage": "training",
                    "elapsed_s": round(now - train_start, 3),
                    "train_examples": train_examples,
                    "val_examples": val_examples,
                    "parameter_count": parameter_count(policy.net[0].out_features),
                    "optimization_forward_example_evals": int(forward_example_evals),
                    "scheduler": {
                        "type": "linear_warmup_cosine_decay",
                        "base_lr": lr,
                        "warmup_fraction": warmup_fraction,
                        "min_lr_scale": min_lr_scale,
                        "current_lr": current_lr,
                    },
                    "history_summary": summarize_partial_history(history),
                    "latest_behavior": latest_behavior,
                    "train_history": history,
                },
            )
            next_val_at = now + val_interval_s
    if not history:
        val_accuracy, val_loss = evaluate_loader(policy, val_loader, policy.net[0].weight.device, loss_fn)
        history.append(
            {
                "epoch": 1,
                "train_loss": epoch_loss / max(1, total),
                "train_accuracy": correct / max(1, total),
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "lr": current_lr,
                "elapsed_s": round(time.perf_counter() - train_start, 3),
            }
        )
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(policy.state_dict(), model_path)
    return history, best_val_accuracy, forward_example_evals, latest_behavior


@torch.no_grad()
def evaluate_clone(
    policy: ClonePolicy,
    eval_seeds: list[int],
    max_steps: int,
    device: torch.device,
    deadline: float,
) -> tuple[list[dict], list[dict]]:
    rollout_rows: list[dict] = []
    episode_rows: list[dict] = []
    for episode_idx, seed in enumerate(eval_seeds):
        if time.perf_counter() >= deadline:
            break
        g = torch.Generator(device=device)
        g.manual_seed(int(seed))
        state = torch.empty(1, 4, device=device, dtype=torch.float32).uniform_(-0.05, 0.05, generator=g)
        actions: list[int] = []
        total_reward = 0.0
        for step in range(max_steps):
            if time.perf_counter() >= deadline:
                break
            logits = policy(state)
            action = int(logits.argmax(dim=1).item())
            rollout_rows.append(
                {
                    "episode": episode_idx,
                    "seed": int(seed),
                    "step": step,
                    "x": float(state[0, 0].item()),
                    "x_dot": float(state[0, 1].item()),
                    "theta": float(state[0, 2].item()),
                    "theta_dot": float(state[0, 3].item()),
                    "action": action,
                }
            )
            actions.append(action)
            state, reward_t, terminated_t = step_cartpole(state, torch.tensor([action], dtype=torch.int64, device=device))
            total_reward += float(reward_t.item())
            if bool(terminated_t.item()):
                break
        if not actions:
            break
        switch_rate = 0.0
        if len(actions) > 1:
            switch_rate = sum(int(a != b) for a, b in zip(actions[:-1], actions[1:])) / (len(actions) - 1)
        episode_rows.append(
            {
                "seed": int(seed),
                "return": total_reward,
                "length": len(actions),
                "action_one_rate": sum(actions) / len(actions),
                "action_switch_rate": switch_rate,
            }
        )
    return rollout_rows, episode_rows


def summarize_rollouts(rollout_rows: list[dict], episode_rows: list[dict]) -> dict:
    returns = np.asarray([row["return"] for row in episode_rows], dtype=np.float64)
    lengths = np.asarray([row["length"] for row in episode_rows], dtype=np.float64)
    action_one_rates = np.asarray([row["action_one_rate"] for row in episode_rows], dtype=np.float64)
    action_switch_rates = np.asarray([row["action_switch_rate"] for row in episode_rows], dtype=np.float64)
    states = np.asarray([[row["x"], row["x_dot"], row["theta"], row["theta_dot"]] for row in rollout_rows], dtype=np.float64)
    actions = np.asarray([row["action"] for row in rollout_rows], dtype=np.float64)
    return {
        "episodes_completed": int(len(episode_rows)),
        "return_mean": float(returns.mean()),
        "return_std": float(returns.std()),
        "length_mean": float(lengths.mean()),
        "length_std": float(lengths.std()),
        "action_one_rate_mean": float(action_one_rates.mean()),
        "action_one_rate_std": float(action_one_rates.std()),
        "action_switch_rate_mean": float(action_switch_rates.mean()),
        "action_switch_rate_std": float(action_switch_rates.std()),
        "global_action_one_rate": float(actions.mean()),
        "state_mean": states.mean(axis=0).tolist(),
        "state_std": states.std(axis=0).tolist(),
    }


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    start = time.perf_counter()
    total_deadline = start + args.time_budget_s
    collect_deadline = start + args.time_budget_s * args.collect_fraction
    dataset_path = Path(args.dataset_path)
    metrics_path = Path(args.metrics_path)
    comparison_path = Path(args.comparison_path)
    model_path = Path(args.model_path)
    if args.reuse_dataset and dataset_path.exists():
        dataset_rows = pd.read_csv(dataset_path).to_dict("records")
    else:
        ppo = PPO.load(args.target_agent_path, device="cpu")
        dataset_rows = collect_training_data(
            model=ppo,
            train_episodes=args.train_episodes,
            train_base_seed=args.train_base_seed,
            max_steps=args.max_steps,
            device=device,
            deadline=collect_deadline,
        )
        if not dataset_rows:
            raise RuntimeError("Time budget expired before collecting clone training data.")
        write_csv(dataset_path, dataset_rows)
    write_progress_snapshot(
        metrics_path,
        comparison_path,
        {
            "stage": "dataset_ready",
            "elapsed_s": round(time.perf_counter() - start, 3),
            "dataset_steps": int(len(dataset_rows)),
            "train_episodes_collected": int(len({row["episode"] for row in dataset_rows})),
            "dataset_reused": bool(args.reuse_dataset and dataset_path.exists()),
            "scheduler": {
                "type": "linear_warmup_cosine_decay",
                "base_lr": args.lr,
                "warmup_fraction": args.warmup_fraction,
                "min_lr_scale": args.min_lr_scale,
            },
        },
    )
    train_x, train_y, val_x, val_y, train_count, val_count = build_loaders(dataset_rows, args.val_fraction)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)
    policy = ClonePolicy(args.hidden_size).to(device)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    train_history: list[dict] = []
    best_val_accuracy = -1.0
    optimization_forward_example_evals = 0
    eval_episode_df = pd.read_csv(args.eval_episodes_path)
    eval_seeds = [int(v) for v in eval_episode_df["seed"].tolist()]
    reference_summary = json.loads(Path(args.eval_summary_path).read_text())
    train_history, best_val_accuracy, optimization_forward_example_evals, latest_behavior = train_for_slice(
        policy=policy,
        optimizer=optimizer,
        loss_fn=loss_fn,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        batch_size=args.batch_size,
        lr=args.lr,
        warmup_fraction=args.warmup_fraction,
        min_lr_scale=args.min_lr_scale,
        train_examples=train_count,
        val_examples=val_count,
        train_start=start,
        total_deadline=total_deadline,
        val_interval_s=args.val_interval_s,
        metrics_path=metrics_path,
        comparison_path=comparison_path,
        model_path=model_path,
        episode_metrics_path=Path(args.episode_metrics_path),
        history=train_history,
        best_val_accuracy=best_val_accuracy,
        forward_example_evals=optimization_forward_example_evals,
        eval_seeds=eval_seeds,
        max_steps=args.max_steps,
        reference_summary=reference_summary,
    )
    if not train_history:
        raise RuntimeError("Time budget expired before clone training completed one epoch.")
    closed_loop_evaluated = latest_behavior is not None
    clone_summary = latest_behavior if latest_behavior is not None else {"episodes_completed": 0}
    comparison = {
        "device": str(device),
        "time_budget_s": args.time_budget_s,
        "dataset_steps": len(dataset_rows),
        "train_episodes_collected": len({row["episode"] for row in dataset_rows}),
        "train_examples": train_count,
        "val_examples": val_count,
        "parameter_count": parameter_count(args.hidden_size),
        "optimization_forward_example_evals": int(optimization_forward_example_evals),
        "optimization_forward_flops_estimate": int(optimization_forward_example_evals * forward_flops_per_example(args.hidden_size)),
        "dataset_reused": bool(args.reuse_dataset and dataset_path.exists()),
        "scheduler": {
            "type": "linear_warmup_cosine_decay",
            "base_lr": args.lr,
            "warmup_fraction": args.warmup_fraction,
            "min_lr_scale": args.min_lr_scale,
            "final_lr": float(train_history[-1]["lr"]),
        },
        "train_accuracy": float(train_history[-1]["train_accuracy"]),
        "best_val_accuracy": float(best_val_accuracy),
        "elapsed_s": round(time.perf_counter() - start, 3),
        "closed_loop_evaluated": closed_loop_evaluated,
        "reference": {
            "return_mean": reference_summary["return_mean"],
            "return_std": reference_summary["return_std"],
            "length_mean": reference_summary["length_mean"],
            "length_std": reference_summary["length_std"],
            "action_one_rate_mean": reference_summary["action_one_rate_mean"],
            "action_switch_rate_mean": reference_summary["action_switch_rate_mean"],
        },
        "clone": clone_summary,
        "deltas": (
            {
                "return_mean_abs": clone_summary["return_mean_abs"],
                "action_switch_rate_mean_abs": clone_summary["action_switch_rate_mean_abs"],
            }
            if closed_loop_evaluated
            else None
        ),
        "train_history": train_history,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(policy.state_dict(), args.model_path)
    write_progress_snapshot(metrics_path, comparison_path, comparison)
    print(
        "dataset_steps={dataset_steps} train_acc={train_accuracy:.4f} val_acc={val_accuracy:.4f} "
        "closed_loop_evaluated={closed_loop_evaluated} clone_return_mean={clone_return}".format(
            dataset_steps=comparison["dataset_steps"],
            train_accuracy=comparison["train_accuracy"],
            val_accuracy=comparison["best_val_accuracy"],
            closed_loop_evaluated=str(comparison["closed_loop_evaluated"]).lower(),
            clone_return=(
                f"{comparison['clone']['return_mean']:.2f}"
                if comparison["closed_loop_evaluated"]
                else "n/a"
            ),
        )
    )


if __name__ == "__main__":
    args = parse_args()
    main(args)
