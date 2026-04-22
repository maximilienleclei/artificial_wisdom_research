from __future__ import annotations

import argparse
import itertools

import numpy as np

from awr.behavior_prediction import ActionPredictionConfig, ActionPredictionEvaluator
from awr.config import CartPoleGAConfig
from awr.models import TorchStaticMLPPopulation
from awr.plateau import SmoothedPlateauTracker
from awr.reporting import GenerationMetrics, write_action_prediction_svg, write_generation_metrics_csv


def run_experiment(
    config: CartPoleGAConfig,
    eval_config: ActionPredictionConfig,
    target_agent_path: str,
    target_agent_algo: str,
    metrics_output_path: str | None = None,
    plot_output_path: str | None = None,
) -> None:
    rng = np.random.default_rng(config.seed)
    evaluator = ActionPredictionEvaluator.from_sb3(
        config=eval_config,
        target_agent_path=target_agent_path,
        target_agent_algo=target_agent_algo,
        num_workers=config.population_size,
    )
    mutation_std = config.mutation_std
    stop_tracker = SmoothedPlateauTracker(config.plateau_stop_smoothing_window)
    metrics_rows: list[GenerationMetrics] = []

    try:
        print(f"evaluation_device={evaluator.device}")
        input_size, output_size = evaluator.retrieve_num_inputs_outputs()
        population = TorchStaticMLPPopulation.random(
            population_size=config.population_size,
            input_size=input_size,
            hidden_size=config.hidden_size,
            output_size=output_size,
            initial_mutation_std=config.mutation_std,
            sigma_sigma=config.sigma_sigma,
            device=evaluator.device,
            seed=config.seed,
        )
        generation_indices = range(config.generations) if config.generations is not None else itertools.count()

        for generation in generation_indices:
            fitness = evaluator.evaluate(population, generation=generation)
            elite_indices, elite_fitness = population.topk(fitness, config.elite_count)
            best = float(elite_fitness[0])
            mean = float(np.mean(fitness))
            smoothed_mean = stop_tracker.update(mean)
            mean_mutation_std = population.mean_mutation_std(mutation_std)

            print(
                f"generation={generation:03d} best={best:.4f} mean={mean:.4f} mutation_std={mean_mutation_std:.4f}"
            )
            metrics_rows.append(
                GenerationMetrics(
                    generation=generation,
                    best=best,
                    mean=mean,
                    mutation_std=mean_mutation_std,
                    annealed=False,
                )
            )
            population = population.reproduce(
                elite_indices=elite_indices,
                mutation_std=mutation_std,
                seed=config.seed + generation + 1,
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
        evaluator.close()

    if metrics_output_path is not None:
        write_generation_metrics_csv(metrics_rows, metrics_output_path)
    if plot_output_path is not None:
        write_action_prediction_svg(
            metrics_rows,
            plot_output_path,
            title="Imitating SB3 PPO CartPole Policy",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run action-prediction GA against an SB3 checkpoint.")
    parser.add_argument("--target-agent-path", required=True)
    parser.add_argument("--target-agent-algo", required=True, choices=["ppo", "sac", "td3", "a2c", "dqn"])
    parser.add_argument("--env-name", default="CartPole-v1")
    parser.add_argument("--generations", type=int, default=None)
    parser.add_argument("--population-size", type=int, default=256)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--mutation-std", type=float, default=0.03)
    parser.add_argument("--sigma-sigma", type=float, default=0.1)
    parser.add_argument("--plateau-stop-patience", type=int, default=None)
    parser.add_argument("--plateau-stop-smoothing-window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--device", default=None)
    parser.add_argument("--metrics-output", default="reports/act_pred_sb3_cartpole_metrics.csv")
    parser.add_argument("--plot-output", default="reports/act_pred_sb3_cartpole_curve.svg")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ga_config = CartPoleGAConfig(
        generations=args.generations,
        population_size=args.population_size,
        hidden_size=args.hidden_size,
        mutation_std=args.mutation_std,
        sigma_sigma=None if args.sigma_sigma <= 0.0 else args.sigma_sigma,
        plateau_stop_patience=args.plateau_stop_patience,
        plateau_stop_smoothing_window=args.plateau_stop_smoothing_window,
        seed=args.seed,
    )
    eval_config = ActionPredictionConfig(
        env_name=args.env_name,
        max_steps=args.max_steps,
        base_seed=args.seed,
        device=args.device,
    )
    run_experiment(
        config=ga_config,
        eval_config=eval_config,
        target_agent_path=args.target_agent_path,
        target_agent_algo=args.target_agent_algo,
        metrics_output_path=args.metrics_output or None,
        plot_output_path=args.plot_output or None,
    )
