from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from awr.config import CartPoleGAConfig
from awr.models import StaticMLPGenome


@dataclass(slots=True)
class ScoredIndividual:
    fitness: float
    genome: StaticMLPGenome


def initialize_population(
    rng: np.random.Generator,
    config: CartPoleGAConfig,
    input_size: int,
    output_size: int,
) -> list[StaticMLPGenome]:
    return [
        StaticMLPGenome.random(
            rng,
            input_size,
            config.hidden_size,
            output_size,
            initial_mutation_std=config.mutation_std,
            sigma_sigma=config.sigma_sigma,
        )
        for _ in range(config.population_size)
    ]


def truncation_selection(
    scored_population: list[ScoredIndividual],
    elite_count: int,
) -> list[ScoredIndividual]:
    ranked = sorted(scored_population, key=lambda item: item.fitness, reverse=True)
    return ranked[:elite_count]


def reproduce(
    rng: np.random.Generator,
    elites: list[ScoredIndividual],
    config: CartPoleGAConfig,
    mutation_std: float,
) -> list[StaticMLPGenome]:
    next_population = [elite.genome.clone() for elite in elites]
    while len(next_population) < config.population_size:
        parent = elites[rng.integers(0, len(elites))].genome
        child = parent.mutate(rng, mutation_std, sigma_sigma=config.sigma_sigma)
        next_population.append(child)
    return next_population
