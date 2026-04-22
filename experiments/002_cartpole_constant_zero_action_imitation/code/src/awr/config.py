from dataclasses import dataclass


@dataclass(slots=True)
class CartPoleGAConfig:
    population_size: int = 256
    elite_fraction: float = 0.5
    episodes_per_individual: int = 3
    generations: int | None = None
    max_steps_per_episode: int = 500
    hidden_size: int = 32
    mutation_std: float = 0.03
    sigma_sigma: float | None = 0.1
    plateau_stop_patience: int | None = None
    plateau_stop_smoothing_window: int = 10
    seed: int = 7

    @property
    def elite_count(self) -> int:
        return max(1, int(self.population_size * self.elite_fraction))
