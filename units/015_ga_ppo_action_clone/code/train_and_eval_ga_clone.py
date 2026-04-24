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

UNIT8_CODE = Path(__file__).resolve().parents[2] / "008_torch_cartpole_physics_parity" / "code"
UNIT_DIR = Path(__file__).resolve().parents[1]
UNIT12_DIR = Path(__file__).resolve().parents[2] / "012_ppo_behavior_benchmark"
UNIT13_DIR = Path(__file__).resolve().parents[2] / "013_supervised_ppo_behavior_clone"
if str(UNIT8_CODE) not in sys.path:
    sys.path.insert(0, str(UNIT8_CODE))

from torch_cartpole import step_cartpole  # noqa: E402


class StaticPopulation:
    def __init__(
        self,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        w3: torch.Tensor,
        b3: torch.Tensor,
        mutation_std: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.w3 = w3
        self.b3 = b3
        self.mutation_std = mutation_std
        self.device = device

    @classmethod
    def random(cls, pop: int, hidden: int, device: torch.device, seed: int) -> "StaticPopulation":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        scale = 0.5
        dtype = torch.float32
        return cls(
            torch.randn(pop, 4, hidden, generator=g, device=device, dtype=dtype) * scale,
            torch.randn(pop, 1, hidden, generator=g, device=device, dtype=dtype) * scale,
            torch.randn(pop, hidden, hidden, generator=g, device=device, dtype=dtype) * scale,
            torch.randn(pop, 1, hidden, generator=g, device=device, dtype=dtype) * scale,
            torch.randn(pop, hidden, 2, generator=g, device=device, dtype=dtype) * scale,
            torch.randn(pop, 1, 2, generator=g, device=device, dtype=dtype) * scale,
            torch.full((pop,), 0.03, device=device, dtype=dtype),
            device,
        )

    @property
    def population_size(self) -> int:
        return int(self.w1.shape[0])

    def forward_logits_batch(self, observations: torch.Tensor) -> torch.Tensor:
        x = observations.unsqueeze(0).expand(self.population_size, -1, -1)
        h1 = torch.tanh(torch.bmm(x, self.w1) + self.b1)
        h2 = torch.tanh(torch.bmm(h1, self.w2) + self.b2)
        return torch.bmm(h2, self.w3) + self.b3

    def select(self, idx: torch.Tensor) -> "StaticPopulation":
        return StaticPopulation(
            self.w1[idx].clone(),
            self.b1[idx].clone(),
            self.w2[idx].clone(),
            self.b2[idx].clone(),
            self.w3[idx].clone(),
            self.b3[idx].clone(),
            self.mutation_std[idx].clone(),
            self.device,
        )

    def reproduce(self, parent_idx: torch.Tensor, mutation_scale_std: float, seed: int) -> "StaticPopulation":
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        parents = self.select(parent_idx)
        log_std = torch.log(torch.clamp(parents.mutation_std, min=1e-4))
        log_std = log_std + torch.randn(log_std.shape, generator=g, device=self.device) * mutation_scale_std
        child_std = torch.clamp(torch.exp(log_std), min=1e-4, max=1.0)

        def mutate(x: torch.Tensor) -> torch.Tensor:
            child = x.clone()
            std = child_std.view(-1, *([1] * (child.dim() - 1)))
            child += torch.randn(child.shape, generator=g, device=self.device) * std
            return child

        return StaticPopulation(
            mutate(parents.w1),
            mutate(parents.b1),
            mutate(parents.w2),
            mutate(parents.b2),
            mutate(parents.w3),
            mutate(parents.b3),
            child_std,
            self.device,
        )

    def best_json(self, index: int, fitness: float) -> dict:
        return {
            "fitness": fitness,
            "w1": self.w1[index].detach().cpu().tolist(),
            "b1": self.b1[index].detach().cpu().tolist(),
            "w2": self.w2[index].detach().cpu().tolist(),
            "b2": self.b2[index].detach().cpu().tolist(),
            "w3": self.w3[index].detach().cpu().tolist(),
            "b3": self.b3[index].detach().cpu().tolist(),
            "mutation_std": float(self.mutation_std[index].detach().cpu().item()),
        }

    @classmethod
    def from_json(cls, payload: dict, device: torch.device) -> "StaticPopulation":
        def to_t(value: list) -> torch.Tensor:
            return torch.tensor([value], dtype=torch.float32, device=device)

        return cls(
            to_t(payload["w1"]),
            to_t(payload["b1"]),
            to_t(payload["w2"]),
            to_t(payload["b2"]),
            to_t(payload["w3"]),
            to_t(payload["b3"]),
            torch.tensor([payload["mutation_std"]], dtype=torch.float32, device=device),
            device,
        )


def parameter_count(hidden_size: int) -> int:
    return (4 * hidden_size + hidden_size) + (hidden_size * hidden_size + hidden_size) + (hidden_size * 2 + 2)


def forward_flops_per_example(hidden_size: int) -> int:
    return 2 * ((4 * hidden_size) + (hidden_size * hidden_size) + (hidden_size * 2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--time-budget-s", type=float, default=6.0)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--parent-count", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--initial-mutation-std", type=float, default=0.03)
    parser.add_argument("--mutation-scale-std", type=float, default=0.1)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-interval-s", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dataset-path", default=str(UNIT13_DIR / "data" / "ppo_train_dataset.csv"))
    parser.add_argument("--eval-summary-path", default=str(UNIT12_DIR / "plot" / "summary.json"))
    parser.add_argument("--eval-episodes-path", default=str(UNIT12_DIR / "plot" / "episode_metrics.csv"))
    parser.add_argument("--metrics-path", default=str(UNIT_DIR / "model" / "metrics.json"))
    parser.add_argument("--best-path", default=str(UNIT_DIR / "model" / "best_genome.json"))
    parser.add_argument("--history-path", default=str(UNIT_DIR / "plot" / "ga_history.csv"))
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


def write_progress_snapshot(metrics_path: Path, comparison_path: Path, payload: dict) -> None:
    write_json(metrics_path, payload)
    write_json(comparison_path, payload)


def load_dataset(dataset_path: str, val_fraction: float, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    df = pd.read_csv(dataset_path)
    obs = torch.tensor(df[["x", "x_dot", "theta", "theta_dot"]].to_numpy(), dtype=torch.float32, device=device)
    actions = torch.tensor(df["action"].to_numpy(), dtype=torch.int64, device=device)
    total = int(actions.shape[0])
    val_count = max(1, min(total - 1, int(total * val_fraction)))
    train_count = total - val_count
    return obs[:train_count], actions[:train_count], obs[train_count:], actions[train_count:]


@torch.no_grad()
def fitness_on_dataset(population: StaticPopulation, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    logits = population.forward_logits_batch(observations)
    probs = torch.softmax(logits, dim=-1)
    target_idx = actions.view(1, -1, 1).expand(population.population_size, -1, 1)
    correct_probs = probs.gather(dim=-1, index=target_idx).squeeze(-1)
    return correct_probs.mean(dim=1)


@torch.no_grad()
def accuracy_for_genome(population: StaticPopulation, observations: torch.Tensor, actions: torch.Tensor, index: int) -> float:
    logits = population.forward_logits_batch(observations)[index]
    pred = logits.argmax(dim=1)
    return float((pred == actions).float().mean().item())


@torch.no_grad()
def evaluate_policy(population: StaticPopulation, seed: int, max_steps: int, device: torch.device) -> tuple[float, int, float, float]:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    state = torch.empty(1, 4, device=device, dtype=torch.float32).uniform_(-0.05, 0.05, generator=g)
    actions: list[int] = []
    total_reward = 0.0
    for _ in range(max_steps):
        action = int(population.forward_logits_batch(state).argmax(dim=-1).item())
        actions.append(action)
        state, reward_t, terminated_t = step_cartpole(state, torch.tensor([action], dtype=torch.int64, device=device))
        total_reward += float(reward_t.item())
        if bool(terminated_t.item()):
            break
    switch_rate = 0.0
    if len(actions) > 1:
        switch_rate = sum(int(a != b) for a, b in zip(actions[:-1], actions[1:])) / (len(actions) - 1)
    return total_reward, len(actions), sum(actions) / len(actions), switch_rate


def evaluate_population_best(
    best_json: dict,
    eval_seeds: list[int],
    max_steps: int,
    device: torch.device,
    deadline: float,
) -> list[dict]:
    best_single = StaticPopulation.from_json(best_json, device)
    episode_rows: list[dict] = []
    for seed in eval_seeds:
        if time.perf_counter() >= deadline:
            break
        ret, length, action_one_rate, switch_rate = evaluate_policy(best_single, seed, max_steps, device)
        episode_rows.append(
            {
                "seed": seed,
                "return": ret,
                "length": length,
                "action_one_rate": action_one_rate,
                "action_switch_rate": switch_rate,
            }
        )
    return episode_rows


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    train_x, train_y, val_x, val_y = load_dataset(args.dataset_path, args.val_fraction, device)
    start = time.perf_counter()
    deadline = start + args.time_budget_s
    metrics_path = Path(args.metrics_path)
    comparison_path = Path(args.comparison_path)
    eval_episode_df = pd.read_csv(args.eval_episodes_path)
    eval_seeds = [int(v) for v in eval_episode_df["seed"].tolist()]
    reference = json.loads(Path(args.eval_summary_path).read_text())

    population = StaticPopulation.random(args.population_size, args.hidden_size, device, args.seed)
    population.mutation_std.fill_(args.initial_mutation_std)
    history: list[dict] = []
    best_json = None
    generation = 0
    best_val_accuracy = 0.0
    optimization_forward_example_evals = 0
    selection_generator = torch.Generator(device=device)
    selection_generator.manual_seed(args.seed + 10_000)
    latest_behavior: dict | None = None
    next_val_at = start
    while time.perf_counter() < deadline:
        fitness = fitness_on_dataset(population, train_x, train_y)
        optimization_forward_example_evals += int(population.population_size * train_x.shape[0])
        parent_fit, parent_idx = torch.topk(fitness, k=args.parent_count, largest=True, sorted=True)
        idx0 = int(parent_idx[0].item())
        train_accuracy = accuracy_for_genome(population, train_x, train_y, idx0)
        val_accuracy = None
        if time.perf_counter() >= next_val_at:
            val_accuracy = accuracy_for_genome(population, val_x, val_y, idx0)
            optimization_forward_example_evals += int(val_x.shape[0])
            current_json = population.best_json(idx0, float(parent_fit[0].item()))
            episode_rows = evaluate_population_best(
                best_json=current_json,
                eval_seeds=eval_seeds,
                max_steps=args.max_steps,
                device=device,
                deadline=deadline,
            )
            if episode_rows:
                write_csv(Path(args.episode_metrics_path), episode_rows)
                clone_df = pd.DataFrame(episode_rows)
                latest_behavior = {
                    "episodes_completed": int(len(episode_rows)),
                    "return_mean": float(clone_df["return"].mean()),
                    "return_std": float(clone_df["return"].std(ddof=0)),
                    "action_one_rate_mean": float(clone_df["action_one_rate"].mean()),
                    "action_switch_rate_mean": float(clone_df["action_switch_rate"].mean()),
                    "return_mean_abs": abs(float(clone_df["return"].mean()) - reference["return_mean"]),
                    "action_switch_rate_mean_abs": abs(
                        float(clone_df["action_switch_rate"].mean()) - reference["action_switch_rate_mean"]
                    ),
                }
            next_val_at = time.perf_counter() + args.val_interval_s
        row = {
            "generation": generation,
            "best_fitness": round(float(parent_fit[0].item()), 6),
            "mean_fitness": round(float(fitness.mean().item()), 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_accuracy": round(val_accuracy, 6) if val_accuracy is not None else None,
            "best_mutation_std": round(float(population.mutation_std[idx0].item()), 6),
            "mean_mutation_std": round(float(population.mutation_std.mean().item()), 6),
            "elapsed_s": round(time.perf_counter() - start, 3),
        }
        if latest_behavior is not None:
            row["closed_loop_episodes_completed"] = latest_behavior["episodes_completed"]
            row["closed_loop_return_mean"] = round(latest_behavior["return_mean"], 6)
            row["closed_loop_switch_delta"] = round(latest_behavior["action_switch_rate_mean_abs"], 6)
        history.append(row)
        write_csv(Path(args.history_path), history)
        if val_accuracy is not None and val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_json = population.best_json(idx0, float(parent_fit[0].item()))
            Path(args.best_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.best_path).write_text(json.dumps(best_json, indent=2) + "\n")
        write_progress_snapshot(
            metrics_path,
            comparison_path,
            {
                "stage": "training",
                "elapsed_s": round(time.perf_counter() - start, 3),
                "train_examples": int(train_x.shape[0]),
                "val_examples": int(val_x.shape[0]),
                "parameter_count": parameter_count(args.hidden_size),
                "optimization_forward_example_evals": int(optimization_forward_example_evals),
                "best_val_accuracy": float(best_val_accuracy) if best_val_accuracy >= 0 else None,
                "best_mutation_std": float(population.mutation_std[idx0].item()),
                "mean_mutation_std": float(population.mutation_std.mean().item()),
                "latest_behavior": latest_behavior,
                "history": history,
            },
        )
        sampled_parent_idx = parent_idx[torch.randint(0, args.parent_count, (args.population_size,), generator=selection_generator, device=device)]
        population = population.reproduce(sampled_parent_idx, args.mutation_scale_std, args.seed + generation + 1)
        generation += 1

    if not history or best_json is None:
        raise RuntimeError("Time budget expired before GA clone completed one generation.")
    closed_loop_evaluated = latest_behavior is not None
    comparison = {
        "device": str(device),
        "time_budget_s": args.time_budget_s,
        "train_examples": int(train_x.shape[0]),
        "val_examples": int(val_x.shape[0]),
        "parameter_count": parameter_count(args.hidden_size),
        "optimization_forward_example_evals": int(optimization_forward_example_evals),
        "optimization_forward_flops_estimate": int(optimization_forward_example_evals * forward_flops_per_example(args.hidden_size)),
        "best_train_accuracy": float(history[-1]["train_accuracy"]),
        "best_val_accuracy": float(best_val_accuracy),
        "closed_loop_evaluated": closed_loop_evaluated,
        "ga_strategy": {
            "selection": "top_k_parent_sampling",
            "parent_count": args.parent_count,
            "self_adaptive_mutation_std": True,
            "initial_mutation_std": args.initial_mutation_std,
            "mutation_scale_std": args.mutation_scale_std,
            "final_best_mutation_std": float(best_json["mutation_std"]),
            "final_mean_mutation_std": float(population.mutation_std.mean().item()),
        },
        "elapsed_s": round(time.perf_counter() - start, 3),
        "reference": {
            "return_mean": reference["return_mean"],
            "return_std": reference["return_std"],
            "length_mean": reference["length_mean"],
            "length_std": reference["length_std"],
            "action_one_rate_mean": reference["action_one_rate_mean"],
            "action_switch_rate_mean": reference["action_switch_rate_mean"],
        },
        "clone": (
            latest_behavior
            if closed_loop_evaluated
            else {"episodes_completed": 0}
        ),
        "deltas": (
            {
                "return_mean_abs": latest_behavior["return_mean_abs"],
                "action_switch_rate_mean_abs": latest_behavior["action_switch_rate_mean_abs"],
            }
            if closed_loop_evaluated
            else None
        ),
        "history": history,
    }
    write_progress_snapshot(metrics_path, comparison_path, comparison)
    print(
        "best_val_acc={val_acc:.4f} closed_loop_evaluated={closed_loop_evaluated} clone_return_mean={clone_return}".format(
            val_acc=comparison["best_val_accuracy"],
            closed_loop_evaluated=str(comparison["closed_loop_evaluated"]).lower(),
            clone_return=(
                f"{comparison['clone']['return_mean']:.2f}"
                if comparison["closed_loop_evaluated"]
                else "n/a"
            ),
        )
    )


if __name__ == "__main__":
    main(parse_args())
