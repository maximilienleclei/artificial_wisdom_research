from __future__ import annotations

import argparse

import gymnasium as gym
import numpy as np

from awr.config import CartPoleGAConfig
from awr.evolution import ScoredIndividual, initialize_population, reproduce, truncation_selection
from awr.plateau import SmoothedPlateauTracker


def evaluate_genome(
    env: gym.Env,
    genome,
    episodes: int,
    max_steps: int,
    rng: np.random.Generator,
) -> float:
    returns = []
    for _ in range(episodes):
        observation, _ = env.reset(seed=int(rng.integers(0, 1_000_000)))
        total_reward = 0.0
        for _ in range(max_steps):
            action = genome.act(np.asarray(observation, dtype=np.float64))
            observation, reward, terminated, truncated, _ = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        returns.append(total_reward)
    return float(np.mean(returns))


def run_experiment(config: CartPoleGAConfig) -> None:
    rng = np.random.default_rng(config.seed)
    env = gym.make("CartPole-v1")
    mutation_std = config.mutation_std
    stop_tracker = SmoothedPlateauTracker(config.plateau_stop_smoothing_window)

    try:
        input_size = int(env.observation_space.shape[0])
        output_size = int(env.action_space.n)
        population = initialize_population(rng, config, input_size, output_size)

        for generation in range(config.generations):
            scored_population = [
                ScoredIndividual(
                    fitness=evaluate_genome(
                        env,
                        genome,
                        episodes=config.episodes_per_individual,
                        max_steps=config.max_steps_per_episode,
                        rng=rng,
                    ),
                    genome=genome,
                )
                for genome in population
            ]
            elites = truncation_selection(scored_population, config.elite_count)
            best = elites[0].fitness
            mean = float(np.mean([item.fitness for item in scored_population]))
            smoothed_mean = stop_tracker.update(mean)
            mean_mutation_std = float(
                np.mean([genome.mean_mutation_std(mutation_std) for genome in population])
            )

            print(
                f"generation={generation:03d} best={best:.2f} mean={mean:.2f} mutation_std={mean_mutation_std:.4f}"
            )
            population = reproduce(
                rng,
                elites,
                config,
                mutation_std=mutation_std,
            )
            if (
                config.plateau_stop_patience is not None
                and stop_tracker.stagnant_updates >= config.plateau_stop_patience
            ):
                print(
                    f"stopping_early generation={generation:03d} reason=smoothed_mean_fitness_plateau patience={config.plateau_stop_patience} window={config.plateau_stop_smoothing_window} smoothed_mean={smoothed_mean:.4f}"
                )
                break
    finally:
        env.close()


def parse_args() -> CartPoleGAConfig:
    parser = argparse.ArgumentParser(description="Run static-network GA on CartPole.")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--episodes-per-individual", type=int, default=3)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--mutation-std", type=float, default=0.03)
    parser.add_argument("--sigma-sigma", type=float, default=0.1)
    parser.add_argument("--plateau-stop-patience", type=int, default=None)
    parser.add_argument("--plateau-stop-smoothing-window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    return CartPoleGAConfig(
        generations=args.generations,
        population_size=args.population_size,
        episodes_per_individual=args.episodes_per_individual,
        hidden_size=args.hidden_size,
        mutation_std=args.mutation_std,
        sigma_sigma=None if args.sigma_sigma <= 0.0 else args.sigma_sigma,
        plateau_stop_patience=args.plateau_stop_patience,
        plateau_stop_smoothing_window=args.plateau_stop_smoothing_window,
        seed=args.seed,
    )


if __name__ == "__main__":
    run_experiment(parse_args())
