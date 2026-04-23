from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

UNIT8_CODE = Path(__file__).resolve().parents[2] / "008_torch_cartpole_physics_parity" / "code"
if str(UNIT8_CODE) not in sys.path:
    sys.path.insert(0, str(UNIT8_CODE))

from torch_cartpole import step_cartpole  # noqa: E402

MAX_INCOMING = 3


class DynamicTorchPopulation:
    def __init__(
        self,
        sources: torch.Tensor,
        weights: torch.Tensor,
        active: torch.Tensor,
        input_size: int,
        output_size: int,
        passes: int,
        device: torch.device,
    ) -> None:
        self.sources = sources
        self.weights = weights
        self.active = active
        self.input_size = input_size
        self.output_size = output_size
        self.passes = passes
        self.device = device

    @classmethod
    def random(
        cls,
        population_size: int,
        input_size: int,
        output_size: int,
        max_hidden: int,
        initial_scale: float,
        passes: int,
        device: torch.device,
        seed: int,
    ) -> "DynamicTorchPopulation":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        mutable_count = output_size + max_hidden
        sources = torch.randint(0, input_size, (population_size, mutable_count, MAX_INCOMING), generator=g, device=device)
        weights = torch.randn(population_size, mutable_count, MAX_INCOMING, generator=g, device=device) * initial_scale
        active = torch.zeros(population_size, mutable_count, dtype=torch.bool, device=device)
        active[:, :output_size] = True
        return cls(sources, weights, active, input_size, output_size, passes, device)

    @property
    def population_size(self) -> int:
        return int(self.sources.shape[0])

    @property
    def mutable_count(self) -> int:
        return int(self.sources.shape[1])

    @property
    def total_nodes(self) -> int:
        return self.input_size + self.mutable_count

    def hidden_counts(self) -> torch.Tensor:
        return self.active[:, self.output_size :].sum(dim=1)

    def forward_logits(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        values = torch.zeros(batch_size, self.total_nodes, device=self.device)
        values[:, : self.input_size] = observations
        mutable_indices = self.input_size + torch.arange(self.mutable_count, device=self.device)
        for _ in range(self.passes):
            previous = values
            gathered = previous.gather(1, self.sources.reshape(batch_size, -1)).reshape(batch_size, self.mutable_count, MAX_INCOMING)
            node_values = torch.tanh((gathered * self.weights).sum(dim=2))
            node_values = torch.where(self.active, node_values, torch.zeros_like(node_values))
            values = values.clone()
            values[:, mutable_indices] = node_values
        return values[:, self.input_size : self.input_size + self.output_size]

    def select(self, indices: torch.Tensor) -> "DynamicTorchPopulation":
        return DynamicTorchPopulation(
            self.sources[indices].clone(),
            self.weights[indices].clone(),
            self.active[indices].clone(),
            self.input_size,
            self.output_size,
            self.passes,
            self.device,
        )

    def reproduce(
        self,
        elite_indices: torch.Tensor,
        mutation_std: float,
        grow_probability: float,
        prune_probability: float,
        rewire_probability: float,
        seed: int,
    ) -> "DynamicTorchPopulation":
        elites = self.select(elite_indices)
        elite_count = int(elite_indices.shape[0])
        child_count = self.population_size - elite_count
        if child_count <= 0:
            return elites
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        parent_choice = torch.randint(0, elite_count, (child_count,), generator=g, device=self.device)
        child_sources = elites.sources[parent_choice].clone()
        child_weights = elites.weights[parent_choice].clone()
        child_active = elites.active[parent_choice].clone()
        child_weights += torch.randn(child_weights.shape, generator=g, device=self.device) * mutation_std

        rewire_mask = torch.rand(child_sources.shape, generator=g, device=self.device) < rewire_probability
        new_sources = torch.randint(0, self.total_nodes, child_sources.shape, generator=g, device=self.device)
        child_sources = torch.where(rewire_mask, new_sources, child_sources)
        child_sources = self._sanitize_sources(child_sources)

        grow_draw = torch.rand(child_count, generator=g, device=self.device)
        for child_idx in torch.nonzero(grow_draw < grow_probability, as_tuple=False).flatten().tolist():
            inactive = torch.nonzero(~child_active[child_idx, self.output_size :], as_tuple=False).flatten()
            if inactive.numel() == 0:
                continue
            hidden_slot = self.output_size + int(inactive[0].item())
            child_active[child_idx, hidden_slot] = True
            child_sources[child_idx, hidden_slot] = torch.randint(
                0, self.total_nodes, (MAX_INCOMING,), generator=g, device=self.device
            )
            child_weights[child_idx, hidden_slot] = torch.randn(MAX_INCOMING, generator=g, device=self.device) * mutation_std
            target = int(torch.randint(0, self.mutable_count, (1,), generator=g, device=self.device).item())
            slot = int(torch.randint(0, MAX_INCOMING, (1,), generator=g, device=self.device).item())
            child_sources[child_idx, target, slot] = self.input_size + hidden_slot

        prune_draw = torch.rand(child_count, generator=g, device=self.device)
        for child_idx in torch.nonzero(prune_draw < prune_probability, as_tuple=False).flatten().tolist():
            active_hidden = torch.nonzero(child_active[child_idx, self.output_size :], as_tuple=False).flatten()
            if active_hidden.numel() == 0:
                continue
            hidden_slot = self.output_size + int(active_hidden[0].item())
            child_active[child_idx, hidden_slot] = False
            node_index = self.input_size + hidden_slot
            replacement = torch.randint(0, self.input_size, child_sources[child_idx].shape, generator=g, device=self.device)
            child_sources[child_idx] = torch.where(child_sources[child_idx] == node_index, replacement, child_sources[child_idx])

        child_sources = self._sanitize_sources(child_sources)
        return DynamicTorchPopulation(
            torch.cat([elites.sources, child_sources], dim=0),
            torch.cat([elites.weights, child_weights], dim=0),
            torch.cat([elites.active, child_active], dim=0),
            self.input_size,
            self.output_size,
            self.passes,
            self.device,
        )

    def _sanitize_sources(self, sources: torch.Tensor) -> torch.Tensor:
        node_indices = self.input_size + torch.arange(self.mutable_count, device=self.device)
        self_sources = node_indices.reshape(1, self.mutable_count, 1)
        invalid = (sources < 0) | (sources >= self.total_nodes) | (sources == self_sources)
        fallback = torch.remainder(sources.abs(), self.input_size)
        return torch.where(invalid, fallback, sources)

    def best_json(self, index: int, fitness: float) -> dict:
        return {
            "fitness": fitness,
            "hidden_count": int(self.hidden_counts()[index].item()),
            "passes": self.passes,
            "sources": self.sources[index].detach().cpu().tolist(),
            "weights": self.weights[index].detach().cpu().tolist(),
            "active": self.active[index].detach().cpu().tolist(),
        }


def evaluate_population(
    population: DynamicTorchPopulation,
    num_episodes: int,
    max_steps: int,
    seed: int,
    deadline: float,
) -> np.ndarray:
    pop = population.population_size
    total_rewards = torch.zeros(pop, device=population.device)
    for episode_idx in range(num_episodes):
        if time.perf_counter() >= deadline:
            break
        g = torch.Generator(device=population.device)
        g.manual_seed(seed + episode_idx)
        state = torch.empty(pop, 4, device=population.device).uniform_(-0.05, 0.05, generator=g)
        done = torch.zeros(pop, dtype=torch.bool, device=population.device)
        for _ in range(max_steps):
            if time.perf_counter() >= deadline:
                break
            logits = population.forward_logits(state)
            action = logits.argmax(dim=1)
            next_state, reward, terminated = step_cartpole(state, action)
            total_rewards += reward * (~done).float()
            done |= terminated
            state = next_state
    completed_episodes = max(1, num_episodes)
    return (total_rewards / completed_episodes).detach().cpu().numpy()


def write_metrics(path: Path, history: list[dict]) -> None:
    if not history:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def run(args: argparse.Namespace) -> None:
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    metrics_path = Path(args.metrics_path)
    best_path = Path(args.best_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    population = DynamicTorchPopulation.random(
        population_size=args.population_size,
        input_size=4,
        output_size=2,
        max_hidden=args.max_hidden,
        initial_scale=args.initial_scale,
        passes=args.passes,
        device=device,
        seed=args.seed,
    )

    start = time.perf_counter()
    deadline = start + args.time_budget_s
    history: list[dict] = []
    best_json = None
    generation = 0

    while generation < args.generations and time.perf_counter() < deadline:
        fitness = evaluate_population(
            population,
            num_episodes=args.episodes_per_individual,
            max_steps=args.max_steps,
            seed=args.seed + generation * 10_000,
            deadline=deadline,
        )
        fitness_tensor = torch.as_tensor(fitness, dtype=torch.float32, device=device)
        elite_fitness, elite_indices = torch.topk(fitness_tensor, k=args.elite_count, largest=True, sorted=True)
        hidden_counts = population.hidden_counts().detach().cpu().numpy()
        row = {
            "generation": generation,
            "best": round(float(elite_fitness[0].item()), 4),
            "mean": round(float(np.mean(fitness)), 4),
            "median": round(float(np.median(fitness)), 4),
            "mean_hidden": round(float(np.mean(hidden_counts)), 3),
            "max_hidden": int(np.max(hidden_counts)),
            "elapsed_s": round(time.perf_counter() - start, 3),
            "time_budget_s": args.time_budget_s,
            "device": str(device),
            "population_size": args.population_size,
            "passes": args.passes,
        }
        history.append(row)
        best_index = int(elite_indices[0].item())
        best_json = population.best_json(best_index, float(elite_fitness[0].item()))
        write_metrics(metrics_path, history)
        best_path.write_text(json.dumps(best_json, indent=2) + "\n")
        print(
            "generation={generation:03d} best={best:.2f} mean={mean:.2f} "
            "mean_hidden={mean_hidden:.2f} elapsed_s={elapsed_s:.3f} device={device}".format(**row)
        )
        if row["mean"] >= 500.0 or time.perf_counter() >= deadline:
            break
        population = population.reproduce(
            elite_indices,
            mutation_std=args.mutation_std,
            grow_probability=args.grow_probability,
            prune_probability=args.prune_probability,
            rewire_probability=args.rewire_probability,
            seed=args.seed + generation + 1,
        )
        generation += 1

    if not history:
        raise RuntimeError("Time budget expired before completing one generation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--time-budget-s", type=float, default=8.0)
    parser.add_argument("--generations", type=int, default=100000)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--episodes-per-individual", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--mutation-std", type=float, default=0.08)
    parser.add_argument("--initial-scale", type=float, default=0.5)
    parser.add_argument("--grow-probability", type=float, default=0.25)
    parser.add_argument("--prune-probability", type=float, default=0.03)
    parser.add_argument("--rewire-probability", type=float, default=0.02)
    parser.add_argument("--max-hidden", type=int, default=24)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metrics-path", default="../plot/dynamic_metrics.csv")
    parser.add_argument("--best-path", default="../model/dynamic_best.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
