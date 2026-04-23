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
        device: torch.device,
    ) -> None:
        self.w1 = w1
        self.b1 = b1
        self.w2 = w2
        self.b2 = b2
        self.w3 = w3
        self.b3 = b3
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
            self.device,
        )

    def reproduce(self, elite_idx: torch.Tensor, mutation_std: float, seed: int) -> "StaticPopulation":
        elites = self.select(elite_idx)
        elite_count = int(elite_idx.shape[0])
        if elite_count == self.population_size:
            return elites
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        child_count = self.population_size - elite_count
        parent = torch.randint(0, elite_count, (child_count,), generator=g, device=self.device)

        def mutate(x: torch.Tensor) -> torch.Tensor:
            child = x[parent].clone()
            child += torch.randn(child.shape, generator=g, device=self.device) * mutation_std
            return child

        return StaticPopulation(
            torch.cat([elites.w1, mutate(elites.w1)], dim=0),
            torch.cat([elites.b1, mutate(elites.b1)], dim=0),
            torch.cat([elites.w2, mutate(elites.w2)], dim=0),
            torch.cat([elites.b2, mutate(elites.b2)], dim=0),
            torch.cat([elites.w3, mutate(elites.w3)], dim=0),
            torch.cat([elites.b3, mutate(elites.b3)], dim=0),
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
            device,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--time-budget-s", type=float, default=6.0)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--mutation-std", type=float, default=0.03)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--train-fraction", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--dataset-path", default="../../013_supervised_ppo_behavior_clone/data/ppo_train_dataset.csv")
    parser.add_argument("--eval-summary-path", default="../../012_ppo_behavior_benchmark/plot/summary.json")
    parser.add_argument("--eval-episodes-path", default="../../012_ppo_behavior_benchmark/plot/episode_metrics.csv")
    parser.add_argument("--metrics-path", default="../model/metrics.json")
    parser.add_argument("--best-path", default="../model/best_genome.json")
    parser.add_argument("--history-path", default="../plot/ga_history.csv")
    parser.add_argument("--comparison-path", default="../plot/benchmark_comparison.json")
    parser.add_argument("--episode-metrics-path", default="../plot/episode_metrics.csv")
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


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    train_x, train_y, val_x, val_y = load_dataset(args.dataset_path, args.val_fraction, device)
    start = time.perf_counter()
    deadline = start + args.time_budget_s
    fit_deadline = start + args.time_budget_s * args.train_fraction

    population = StaticPopulation.random(args.population_size, args.hidden_size, device, args.seed)
    history: list[dict] = []
    best_json = None
    generation = 0
    best_population = population
    best_index = 0
    best_val_accuracy = 0.0

    while time.perf_counter() < fit_deadline:
        fitness = fitness_on_dataset(population, train_x, train_y)
        elite_fit, elite_idx = torch.topk(fitness, k=args.elite_count, largest=True, sorted=True)
        idx0 = int(elite_idx[0].item())
        train_accuracy = accuracy_for_genome(population, train_x, train_y, idx0)
        val_accuracy = accuracy_for_genome(population, val_x, val_y, idx0)
        row = {
            "generation": generation,
            "best_fitness": round(float(elite_fit[0].item()), 6),
            "mean_fitness": round(float(fitness.mean().item()), 6),
            "train_accuracy": round(train_accuracy, 6),
            "val_accuracy": round(val_accuracy, 6),
            "elapsed_s": round(time.perf_counter() - start, 3),
        }
        history.append(row)
        write_csv(Path(args.history_path), history)
        if val_accuracy >= best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_index = idx0
            best_population = population
            best_json = population.best_json(idx0, float(elite_fit[0].item()))
            Path(args.best_path).parent.mkdir(parents=True, exist_ok=True)
            Path(args.best_path).write_text(json.dumps(best_json, indent=2) + "\n")
        if time.perf_counter() >= fit_deadline:
            break
        population = population.reproduce(elite_idx, args.mutation_std, args.seed + generation + 1)
        generation += 1

    if not history or best_json is None:
        raise RuntimeError("Time budget expired before GA clone completed one generation.")

    eval_episode_df = pd.read_csv(args.eval_episodes_path)
    eval_seeds = [int(v) for v in eval_episode_df["seed"].tolist()]
    best_single = StaticPopulation.from_json(best_json, device)
    episode_rows: list[dict] = []
    for seed in eval_seeds:
        if time.perf_counter() >= deadline:
            break
        ret, length, action_one_rate, switch_rate = evaluate_policy(best_single, seed, args.max_steps, device)
        episode_rows.append(
            {
                "seed": seed,
                "return": ret,
                "length": length,
                "action_one_rate": action_one_rate,
                "action_switch_rate": switch_rate,
            }
        )
    if not episode_rows:
        raise RuntimeError("Time budget expired before GA clone evaluation completed one episode.")
    write_csv(Path(args.episode_metrics_path), episode_rows)

    reference = json.loads(Path(args.eval_summary_path).read_text())
    clone_df = pd.DataFrame(episode_rows)
    comparison = {
        "device": str(device),
        "time_budget_s": args.time_budget_s,
        "train_examples": int(train_x.shape[0]),
        "val_examples": int(val_x.shape[0]),
        "best_train_accuracy": float(history[-1]["train_accuracy"]),
        "best_val_accuracy": float(best_val_accuracy),
        "elapsed_s": round(time.perf_counter() - start, 3),
        "reference": {
            "return_mean": reference["return_mean"],
            "return_std": reference["return_std"],
            "length_mean": reference["length_mean"],
            "length_std": reference["length_std"],
            "action_one_rate_mean": reference["action_one_rate_mean"],
            "action_switch_rate_mean": reference["action_switch_rate_mean"],
        },
        "clone": {
            "episodes_completed": int(len(episode_rows)),
            "return_mean": float(clone_df["return"].mean()),
            "return_std": float(clone_df["return"].std(ddof=0)),
            "length_mean": float(clone_df["length"].mean()),
            "length_std": float(clone_df["length"].std(ddof=0)),
            "action_one_rate_mean": float(clone_df["action_one_rate"].mean()),
            "action_one_rate_std": float(clone_df["action_one_rate"].std(ddof=0)),
            "action_switch_rate_mean": float(clone_df["action_switch_rate"].mean()),
            "action_switch_rate_std": float(clone_df["action_switch_rate"].std(ddof=0)),
        },
        "deltas": {
            "return_mean_abs": abs(float(clone_df["return"].mean()) - reference["return_mean"]),
            "return_std_abs": abs(float(clone_df["return"].std(ddof=0)) - reference["return_std"]),
            "length_mean_abs": abs(float(clone_df["length"].mean()) - reference["length_mean"]),
            "length_std_abs": abs(float(clone_df["length"].std(ddof=0)) - reference["length_std"]),
            "action_one_rate_mean_abs": abs(float(clone_df["action_one_rate"].mean()) - reference["action_one_rate_mean"]),
            "action_switch_rate_mean_abs": abs(float(clone_df["action_switch_rate"].mean()) - reference["action_switch_rate_mean"]),
        },
        "history": history,
    }
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics_path).write_text(json.dumps(comparison, indent=2) + "\n")
    Path(args.comparison_path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.comparison_path).write_text(json.dumps(comparison, indent=2) + "\n")
    print(
        "best_val_acc={val_acc:.4f} clone_return_mean={clone_return:.2f} return_delta={return_delta:.2f} switch_delta={switch_delta:.4f}".format(
            val_acc=comparison["best_val_accuracy"],
            clone_return=comparison["clone"]["return_mean"],
            return_delta=comparison["deltas"]["return_mean_abs"],
            switch_delta=comparison["deltas"]["action_switch_rate_mean_abs"],
        )
    )


if __name__ == "__main__":
    main(parse_args())
