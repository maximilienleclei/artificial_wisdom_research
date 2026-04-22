import numpy as np

from awr.config import CartPoleGAConfig
from awr.evolution import ScoredIndividual, initialize_population, reproduce, truncation_selection
from awr.models import StaticMLPGenome


def test_truncation_selection_keeps_top_half():
    scored = [
        ScoredIndividual(fitness=float(score), genome=_dummy_genome())
        for score in [1, 5, 3, 4]
    ]

    elites = truncation_selection(scored, elite_count=2)

    assert [elite.fitness for elite in elites] == [5.0, 4.0]


def test_reproduce_restores_population_size():
    rng = np.random.default_rng(0)
    config = CartPoleGAConfig(population_size=6)
    elites = [ScoredIndividual(fitness=1.0, genome=_dummy_genome()) for _ in range(3)]

    population = reproduce(rng, elites, config, mutation_std=0.03)

    assert len(population) == 6


def test_initialize_population_shapes():
    rng = np.random.default_rng(0)
    config = CartPoleGAConfig(population_size=4, hidden_size=8)

    population = initialize_population(rng, config, input_size=4, output_size=2)

    assert len(population) == 4
    assert population[0].input_hidden.shape == (4, 8)
    assert population[0].hidden_output.shape == (8, 2)
    assert population[0].input_hidden_sigma is not None


def test_self_adaptive_mutation_updates_sigmas():
    rng = np.random.default_rng(0)
    genome = StaticMLPGenome.random(
        rng,
        input_size=4,
        hidden_size=2,
        output_size=2,
        initial_mutation_std=0.03,
        sigma_sigma=0.1,
    )

    child = genome.mutate(rng, mutation_std=0.03, sigma_sigma=0.1)

    assert child.input_hidden_sigma is not None
    assert not np.allclose(child.input_hidden_sigma, genome.input_hidden_sigma)


def _dummy_genome() -> StaticMLPGenome:
    return StaticMLPGenome(
        input_hidden=np.zeros((4, 2)),
        hidden_bias=np.zeros((2,)),
        hidden_output=np.zeros((2, 2)),
        output_bias=np.zeros((2,)),
    )
