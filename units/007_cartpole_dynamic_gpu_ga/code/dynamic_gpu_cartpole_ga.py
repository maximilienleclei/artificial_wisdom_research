from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch


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
    ) -> DynamicTorchPopulation:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        mutable_count = output_size + max_hidden
        sources = torch.randint(
            low=0,
            high=input_size,
            size=(population_size, mutable_count, MAX_INCOMING),
            generator=generator,
            device=device,
        )
        weights = torch.randn(
            population_size,
            mutable_count,
            MAX_INCOMING,
            generator=generator,
            device=device,
        ) * initial_scale
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
            gathered = previous.gather(
                dim=1,
                index=self.sources.reshape(batch_size, -1),
            ).reshape(batch_size, self.mutable_count, MAX_INCOMING)
            node_values = torch.tanh((gathered * self.weights).sum(dim=2))
            node_values = torch.where(self.active, node_values, torch.zeros_like(node_values))
            values = values.clone()
            values[:, mutable_indices] = node_values
        return values[:, self.input_size : self.input_size + self.output_size]

    def select(self, indices: torch.Tensor) -> DynamicTorchPopulation:
        return DynamicTorchPopulation(
            sources=self.sources[indices].clone(),
            weights=self.weights[indices].clone(),
            active=self.active[indices].clone(),
            input_size=self.input_size,
            output_size=self.output_size,
            passes=self.passes,
            device=self.device,
        )

    def reproduce(
        self,
        elite_indices: torch.Tensor,
        mutation_std: float,
        grow_probability: float,
        prune_probability: float,
        rewire_probability: float,
        seed: int,
    ) -> DynamicTorchPopulation:
        elites = self.select(elite_indices)
        elite_count = int(elite_indices.shape[0])
        child_count = self.population_size - elite_count
        if child_count <= 0:
            return elites

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        parent_choice = torch.randint(
            low=0,
            high=elite_count,
            size=(child_count,),
            generator=generator,
            device=self.device,
        )
        child_sources = elites.sources[parent_choice].clone()
        child_weights = elites.weights[parent_choice].clone()
        child_active = elites.active[parent_choice].clone()

        child_weights += torch.randn(
            child_weights.shape,
            generator=generator,
            device=self.device,
        ) * mutation_std

        rewire_mask = torch.rand(
            child_sources.shape,
            generator=generator,
            device=self.device,
        ) < rewire_probability
        new_sources = torch.randint(
            low=0,
            high=self.total_nodes,
            size=child_sources.shape,
            generator=generator,
            device=self.device,
        )
        child_sources = torch.where(rewire_mask, new_sources, child_sources)
        child_sources = self._sanitize_sources(child_sources)

        grow_draw = torch.rand(child_count, generator=generator, device=self.device)
        for child_idx in torch.nonzero(grow_draw < grow_probability, as_tuple=False).flatten().tolist():
            inactive = torch.nonzero(~child_active[child_idx, self.output_size :], as_tuple=False).flatten()
            if inactive.numel() == 0:
                continue
            hidden_slot = self.output_size + int(inactive[0].item())
            child_active[child_idx, hidden_slot] = True
            child_sources[child_idx, hidden_slot] = torch.randint(
                low=0,
                high=self.total_nodes,
                size=(MAX_INCOMING,),
                generator=generator,
                device=self.device,
            )
            child_weights[child_idx, hidden_slot] = (
                torch.randn(MAX_INCOMING, generator=generator, device=self.device) * mutation_std
            )
            target = int(torch.randint(0, self.mutable_count, (1,), generator=generator, device=self.device).item())
            slot = int(torch.randint(0, MAX_INCOMING, (1,), generator=generator, device=self.device).item())
            child_sources[child_idx, target, slot] = self.input_size + hidden_slot

        prune_draw = torch.rand(child_count, generator=generator, device=self.device)
        for child_idx in torch.nonzero(prune_draw < prune_probability, as_tuple=False).flatten().tolist():
            active_hidden = torch.nonzero(child_active[child_idx, self.output_size :], as_tuple=False).flatten()
            if active_hidden.numel() == 0:
                continue
            hidden_slot = self.output_size + int(active_hidden[0].item())
            child_active[child_idx, hidden_slot] = False
            node_index = self.input_size + hidden_slot
            replacement = torch.randint(
                low=0,
                high=self.input_size,
                size=child_sources[child_idx].shape,
                generator=generator,
                device=self.device,
            )
            child_sources[child_idx] = torch.where(
                child_sources[child_idx] == node_index,
                replacement,
                child_sources[child_idx],
            )

        child_sources = self._sanitize_sources(child_sources)

        return DynamicTorchPopulation(
            sources=torch.cat([elites.sources, child_sources], dim=0),
            weights=torch.cat([elites.weights, child_weights], dim=0),
            active=torch.cat([elites.active, child_active], dim=0),
            input_size=self.input_size,
            output_size=self.output_size,
            passes=self.passes,
            device=self.device,
        )

    def _sanitize_sources(self, sources: torch.Tensor) -> torch.Tensor:
        node_indices = self.input_size + torch.arange(self.mutable_count, device=self.device)
        self_sources = node_indices.reshape(1, self.mutable_count, 1)
        invalid = (sources < 0) | (sources >= self.total_nodes) | (sources == self_sources)
        fallback = torch.remainder(sources.abs(), self.input_size)
        return torch.where(invalid, fallback, sources)

    def to_jsonable_best(self, index: int, fitness: float) -> dict:
        return {
            "fitness": fitness,
            "hidden_count": int(self.hidden_counts()[index].item()),
            "passes": self.passes,
            "sources": self.sources[index].detach().cpu().tolist(),
            "weights": self.weights[index].detach().cpu().tolist(),
            "active": self.active[index].detach().cpu().tolist(),
        }


def make_vector_env(env_name: str, population_size: int) -> gym.vector.VectorEnv:
    return gym.vector.SyncVectorEnv([lambda en=env_name: gym.make(en) for _ in range(population_size)])


def write_metrics(metrics_path: Path, history: list[dict]) -> None:
    if not history:
        return
    with metrics_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def evaluate_population(
    env: gym.vector.VectorEnv,
    population: DynamicTorchPopulation,
    max_steps: int,
    seed: int,
    deadline: float,
) -> np.ndarray:
    obs, _ = env.reset(seed=[seed + i for i in range(population.population_size)])
    fitness = np.zeros(population.population_size, dtype=np.float64)
    done = np.zeros(population.population_size, dtype=bool)

    for _ in range(max_steps):
        if time.perf_counter() >= deadline:
            break
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=population.device)
        with torch.no_grad():
            actions = population.forward_logits(obs_tensor).argmax(dim=1).detach().cpu().numpy()
        obs, rewards, terminated, truncated, _ = env.step(actions)
        fitness += rewards * (~done)
        done |= terminated | truncated
    return fitness


def run(args: argparse.Namespace) -> None:
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    env = make_vector_env("CartPole-v1", args.population_size)
    metrics_path = Path(args.metrics_path)
    best_path = Path(args.best_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        single_env = gym.make("CartPole-v1")
        input_size = int(single_env.observation_space.shape[0])
        output_size = int(single_env.action_space.n)
        single_env.close()

        population = DynamicTorchPopulation.random(
            population_size=args.population_size,
            input_size=input_size,
            output_size=output_size,
            max_hidden=args.max_hidden,
            initial_scale=args.initial_scale,
            passes=args.passes,
            device=device,
            seed=args.seed,
        )

        history = []
        start = time.perf_counter()
        deadline = start + args.time_budget_s
        best_json = None

        for generation in range(args.generations):
            if time.perf_counter() >= deadline:
                break
            fitness_runs = [
                evaluate_population(
                    env,
                    population,
                    args.max_steps,
                    args.seed + generation * 1000 + ep * 100_000,
                    deadline,
                )
                for ep in range(args.episodes_per_individual)
            ]
            fitness = np.mean(np.stack(fitness_runs), axis=0)
            fitness_tensor = torch.as_tensor(fitness, dtype=torch.float32, device=device)
            elite_fitness, elite_indices = torch.topk(
                fitness_tensor,
                k=args.elite_count,
                largest=True,
                sorted=True,
            )
            hidden_counts = population.hidden_counts().detach().cpu().numpy()
            best_index = int(elite_indices[0].item())
            best = float(elite_fitness[0].item())
            row = {
                "generation": generation,
                "best": round(best, 4),
                "mean": round(float(np.mean(fitness)), 4),
                "median": round(float(np.median(fitness)), 4),
                "mean_hidden": round(float(np.mean(hidden_counts)), 3),
                "max_hidden": int(np.max(hidden_counts)),
                "elapsed_s": round(time.perf_counter() - start, 3),
                "time_budget_s": args.time_budget_s,
                "device": str(device),
            }
            history.append(row)
            best_json = population.to_jsonable_best(best_index, best)
            write_metrics(metrics_path, history)
            best_path.write_text(json.dumps(best_json, indent=2) + "\n")
            print(
                "generation={generation:03d} best={best:.2f} mean={mean:.2f} "
                "mean_hidden={mean_hidden:.2f} max_hidden={max_hidden} device={device}".format(**row)
            )
            if time.perf_counter() >= deadline:
                break
            population = population.reproduce(
                elite_indices=elite_indices,
                mutation_std=args.mutation_std,
                grow_probability=args.grow_probability,
                prune_probability=args.prune_probability,
                rewire_probability=args.rewire_probability,
                seed=args.seed + generation + 1,
            )

        if not history:
            raise RuntimeError("Time budget expired before completing one generation.")

        write_metrics(metrics_path, history)
        best_path.write_text(json.dumps(best_json, indent=2) + "\n")
    finally:
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=100)
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
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--time-budget-s", type=float, default=30.0)
    parser.add_argument("--metrics-path", default="../plot/metrics.csv")
    parser.add_argument("--best-path", default="../model/best_genome.json")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
