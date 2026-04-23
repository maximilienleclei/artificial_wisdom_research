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


class StaticPopulation:
    def __init__(
        self,
        input_hidden: torch.Tensor,
        hidden_bias: torch.Tensor,
        hidden_output: torch.Tensor,
        output_bias: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.input_hidden = input_hidden
        self.hidden_bias = hidden_bias
        self.hidden_output = hidden_output
        self.output_bias = output_bias
        self.device = device

    @classmethod
    def random(
        cls,
        population_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        device: torch.device,
        seed: int,
    ) -> "StaticPopulation":
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        dtype = torch.float32
        scale = 0.5
        return cls(
            input_hidden=torch.randn(population_size, input_size, hidden_size, generator=g, device=device, dtype=dtype)
            * scale,
            hidden_bias=torch.randn(population_size, 1, hidden_size, generator=g, device=device, dtype=dtype) * scale,
            hidden_output=torch.randn(
                population_size, hidden_size, output_size, generator=g, device=device, dtype=dtype
            )
            * scale,
            output_bias=torch.randn(population_size, 1, output_size, generator=g, device=device, dtype=dtype) * scale,
            device=device,
        )

    @property
    def population_size(self) -> int:
        return int(self.input_hidden.shape[0])

    def forward_logits(self, observations: torch.Tensor) -> torch.Tensor:
        hidden = torch.bmm(observations, self.input_hidden) + self.hidden_bias
        hidden = torch.tanh(hidden)
        return torch.bmm(hidden, self.hidden_output) + self.output_bias

    def select(self, indices: torch.Tensor) -> "StaticPopulation":
        return StaticPopulation(
            self.input_hidden[indices].clone(),
            self.hidden_bias[indices].clone(),
            self.hidden_output[indices].clone(),
            self.output_bias[indices].clone(),
            self.device,
        )

    def reproduce(self, elite_indices: torch.Tensor, mutation_std: float, seed: int) -> "StaticPopulation":
        elites = self.select(elite_indices)
        elite_count = int(elite_indices.shape[0])
        if elite_count == self.population_size:
            return elites
        g = torch.Generator(device=self.device)
        g.manual_seed(seed)
        child_count = self.population_size - elite_count
        parent_choice = torch.randint(0, elite_count, (child_count,), generator=g, device=self.device)
        child_input_hidden = elites.input_hidden[parent_choice].clone()
        child_hidden_bias = elites.hidden_bias[parent_choice].clone()
        child_hidden_output = elites.hidden_output[parent_choice].clone()
        child_output_bias = elites.output_bias[parent_choice].clone()
        child_input_hidden += torch.randn(child_input_hidden.shape, generator=g, device=self.device) * mutation_std
        child_hidden_bias += torch.randn(child_hidden_bias.shape, generator=g, device=self.device) * mutation_std
        child_hidden_output += torch.randn(child_hidden_output.shape, generator=g, device=self.device) * mutation_std
        child_output_bias += torch.randn(child_output_bias.shape, generator=g, device=self.device) * mutation_std
        return StaticPopulation(
            torch.cat([elites.input_hidden, child_input_hidden], dim=0),
            torch.cat([elites.hidden_bias, child_hidden_bias], dim=0),
            torch.cat([elites.hidden_output, child_hidden_output], dim=0),
            torch.cat([elites.output_bias, child_output_bias], dim=0),
            self.device,
        )

    def best_json(self, index: int, fitness: float) -> dict:
        return {
            "fitness": fitness,
            "input_hidden": self.input_hidden[index].detach().cpu().tolist(),
            "hidden_bias": self.hidden_bias[index].detach().cpu().tolist(),
            "hidden_output": self.hidden_output[index].detach().cpu().tolist(),
            "output_bias": self.output_bias[index].detach().cpu().tolist(),
        }


def evaluate_population(
    population: StaticPopulation,
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
            logits = population.forward_logits(state.unsqueeze(1)).squeeze(1)
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

    population = StaticPopulation.random(
        population_size=args.population_size,
        input_size=4,
        hidden_size=args.hidden_size,
        output_size=2,
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
        row = {
            "generation": generation,
            "best": round(float(elite_fitness[0].item()), 4),
            "mean": round(float(np.mean(fitness)), 4),
            "median": round(float(np.median(fitness)), 4),
            "elapsed_s": round(time.perf_counter() - start, 3),
            "time_budget_s": args.time_budget_s,
            "device": str(device),
            "population_size": args.population_size,
            "hidden_size": args.hidden_size,
        }
        history.append(row)
        best_index = int(elite_indices[0].item())
        best_json = population.best_json(best_index, float(elite_fitness[0].item()))
        write_metrics(metrics_path, history)
        best_path.write_text(json.dumps(best_json, indent=2) + "\n")
        print(
            "generation={generation:03d} best={best:.2f} mean={mean:.2f} "
            "elapsed_s={elapsed_s:.3f} device={device}".format(**row)
        )
        if row["mean"] >= 500.0 or time.perf_counter() >= deadline:
            break
        population = population.reproduce(elite_indices, mutation_std=args.mutation_std, seed=args.seed + generation + 1)
        generation += 1

    if not history:
        raise RuntimeError("Time budget expired before completing one generation.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--time-budget-s", type=float, default=10.0)
    parser.add_argument("--generations", type=int, default=100000)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--elite-count", type=int, default=8)
    parser.add_argument("--episodes-per-individual", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--mutation-std", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--metrics-path", default="../plot/static_metrics.csv")
    parser.add_argument("--best-path", default="../model/static_best.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
