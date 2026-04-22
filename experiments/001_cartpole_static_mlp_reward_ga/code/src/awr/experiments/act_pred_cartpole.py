from __future__ import annotations

import argparse

import numpy as np

from awr.behavior_prediction import ActionPredictionConfig, ActionPredictionEvaluator
from awr.config import CartPoleGAConfig
from awr.evolution import ScoredIndividual, initialize_population, reproduce, truncation_selection
from awr.plateau import SmoothedPlateauTracker
from awr.reporting import GenerationMetrics, write_action_prediction_svg, write_generation_metrics_csv


class ConstantZeroPolicy:
    def predict(
        self,
        observations: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        return np.zeros(len(observations), dtype=np.int64), None


def run_experiment(
    config: CartPoleGAConfig,
    eval_config: ActionPredictionConfig,
    metrics_output_path: str | None = None,
    plot_output_path: str | None = None,
) -> None:
    rng = np.random.default_rng(config.seed)
    evaluator = ActionPredictionEvaluator(
        config=eval_config,
        target_policy=ConstantZeroPolicy(),
        num_workers=config.population_size,
    )
    mutation_std = config.mutation_std
    stop_tracker = SmoothedPlateauTracker(config.plateau_stop_smoothing_window)
    metrics_rows: list[GenerationMetrics] = []

    try:
        input_size, output_size = evaluator.retrieve_num_inputs_outputs()
        population = initialize_population(rng, config, input_size, output_size)

        for generation in range(config.generations):
            fitness = evaluator.evaluate(population, generation=generation)
            scored_population = [
                ScoredIndividual(fitness=float(score), genome=genome)
                for genome, score in zip(population, fitness, strict=True)
            ]
            elites = truncation_selection(scored_population, config.elite_count)
            best = elites[0].fitness
            mean = float(np.mean(fitness))
            smoothed_mean = stop_tracker.update(mean)
            mean_mutation_std = float(
                np.mean([genome.mean_mutation_std(mutation_std) for genome in population])
            )

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
            population = reproduce(rng, elites, config, mutation_std=mutation_std)
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
            title="Imitating Constant-Zero CartPole Policy",
        )


def parse_args() -> tuple[CartPoleGAConfig, ActionPredictionConfig]:
    parser = argparse.ArgumentParser(description="Run action-prediction GA on CartPole.")
    parser.add_argument("--generations", type=int, default=40)
    parser.add_argument("--population-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--mutation-std", type=float, default=0.03)
    parser.add_argument("--sigma-sigma", type=float, default=0.1)
    parser.add_argument("--plateau-stop-patience", type=int, default=None)
    parser.add_argument("--plateau-stop-smoothing-window", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--metrics-output", default="reports/act_pred_constant_zero_metrics.csv")
    parser.add_argument("--plot-output", default="reports/act_pred_constant_zero_curve.svg")
    args = parser.parse_args()

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
        env_name="CartPole-v1",
        max_steps=args.max_steps,
        base_seed=args.seed,
    )
    return ga_config, eval_config, args.metrics_output, args.plot_output


if __name__ == "__main__":
    ga_config, eval_config, metrics_output, plot_output = parse_args()
    run_experiment(
        ga_config,
        eval_config,
        metrics_output_path=metrics_output,
        plot_output_path=plot_output,
    )
