from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor


@dataclass
class StaticMLPGenome:
    input_hidden: np.ndarray
    hidden_bias: np.ndarray
    hidden_output: np.ndarray
    output_bias: np.ndarray
    input_hidden_sigma: np.ndarray | None = None
    hidden_bias_sigma: np.ndarray | None = None
    hidden_output_sigma: np.ndarray | None = None
    output_bias_sigma: np.ndarray | None = None

    @classmethod
    def random(
        cls,
        rng: np.random.Generator,
        input_size: int,
        hidden_size: int,
        output_size: int,
        initial_mutation_std: float,
        sigma_sigma: float | None,
    ) -> "StaticMLPGenome":
        scale = 0.5
        if sigma_sigma is None:
            input_hidden_sigma = None
            hidden_bias_sigma = None
            hidden_output_sigma = None
            output_bias_sigma = None
        else:
            input_hidden_sigma = np.full((input_size, hidden_size), initial_mutation_std)
            hidden_bias_sigma = np.full((hidden_size,), initial_mutation_std)
            hidden_output_sigma = np.full((hidden_size, output_size), initial_mutation_std)
            output_bias_sigma = np.full((output_size,), initial_mutation_std)
        return cls(
            input_hidden=rng.normal(0.0, scale, size=(input_size, hidden_size)),
            hidden_bias=rng.normal(0.0, scale, size=(hidden_size,)),
            hidden_output=rng.normal(0.0, scale, size=(hidden_size, output_size)),
            output_bias=rng.normal(0.0, scale, size=(output_size,)),
            input_hidden_sigma=input_hidden_sigma,
            hidden_bias_sigma=hidden_bias_sigma,
            hidden_output_sigma=hidden_output_sigma,
            output_bias_sigma=output_bias_sigma,
        )

    def clone(self) -> "StaticMLPGenome":
        return StaticMLPGenome(
            input_hidden=self.input_hidden.copy(),
            hidden_bias=self.hidden_bias.copy(),
            hidden_output=self.hidden_output.copy(),
            output_bias=self.output_bias.copy(),
            input_hidden_sigma=None if self.input_hidden_sigma is None else self.input_hidden_sigma.copy(),
            hidden_bias_sigma=None if self.hidden_bias_sigma is None else self.hidden_bias_sigma.copy(),
            hidden_output_sigma=None if self.hidden_output_sigma is None else self.hidden_output_sigma.copy(),
            output_bias_sigma=None if self.output_bias_sigma is None else self.output_bias_sigma.copy(),
        )

    def mutate(
        self,
        rng: np.random.Generator,
        mutation_std: float,
        sigma_sigma: float | None = None,
    ) -> "StaticMLPGenome":
        child = self.clone()
        sigma_pairs = (
            ("input_hidden", "input_hidden_sigma"),
            ("hidden_bias", "hidden_bias_sigma"),
            ("hidden_output", "hidden_output_sigma"),
            ("output_bias", "output_bias_sigma"),
        )
        for field_name, sigma_field_name in sigma_pairs:
            weights = getattr(child, field_name)
            sigma_values = getattr(child, sigma_field_name)
            if sigma_sigma is not None and sigma_values is not None:
                sigma_noise = rng.normal(0.0, sigma_sigma, size=sigma_values.shape)
                sigma_values = np.clip(sigma_values * (1.0 + sigma_noise), a_min=1e-6, a_max=None)
                setattr(child, sigma_field_name, sigma_values)
                noise_std = sigma_values
            else:
                noise_std = mutation_std
            noise = rng.normal(0.0, noise_std, size=weights.shape)
            setattr(child, field_name, weights + noise)
        return child

    def mean_mutation_std(self, fallback_mutation_std: float) -> float:
        sigma_values = [
            self.input_hidden_sigma,
            self.hidden_bias_sigma,
            self.hidden_output_sigma,
            self.output_bias_sigma,
        ]
        present = [float(value.mean()) for value in sigma_values if value is not None]
        if not present:
            return fallback_mutation_std
        return float(np.mean(present))

    def forward_logits(self, observation: np.ndarray) -> np.ndarray:
        hidden = np.tanh(observation @ self.input_hidden + self.hidden_bias)
        return hidden @ self.hidden_output + self.output_bias

    def act(self, observation: np.ndarray) -> int:
        logits = self.forward_logits(observation)
        return int(np.argmax(logits))


@dataclass(slots=True)
class BatchedMLPPopulation:
    input_hidden: torch.Tensor
    hidden_bias: torch.Tensor
    hidden_output: torch.Tensor
    output_bias: torch.Tensor
    device: torch.device

    @classmethod
    def from_genomes(
        cls,
        genomes: list[StaticMLPGenome],
        device: torch.device,
    ) -> "BatchedMLPPopulation":
        if not genomes:
            raise ValueError("Cannot batch an empty population.")
        dtype = _torch_dtype_from_numpy(genomes[0].input_hidden.dtype)
        return cls(
            input_hidden=torch.stack(
                [torch.as_tensor(genome.input_hidden, dtype=dtype, device=device) for genome in genomes]
            ),
            hidden_bias=torch.stack(
                [torch.as_tensor(genome.hidden_bias, dtype=dtype, device=device) for genome in genomes]
            ).unsqueeze(1),
            hidden_output=torch.stack(
                [torch.as_tensor(genome.hidden_output, dtype=dtype, device=device) for genome in genomes]
            ),
            output_bias=torch.stack(
                [torch.as_tensor(genome.output_bias, dtype=dtype, device=device) for genome in genomes]
            ).unsqueeze(1),
            device=device,
        )

    def forward_logits_batch(self, observations: np.ndarray) -> torch.Tensor:
        x = torch.as_tensor(observations, dtype=self.input_hidden.dtype, device=self.device)
        x = x.unsqueeze(0).expand(self.input_hidden.shape[0], -1, -1)
        hidden = torch.bmm(x, self.input_hidden) + self.hidden_bias
        hidden = torch.tanh(hidden)
        logits = torch.bmm(hidden, self.hidden_output) + self.output_bias
        return logits


@dataclass(slots=True)
class TorchStaticMLPPopulation:
    input_hidden: Tensor
    hidden_bias: Tensor
    hidden_output: Tensor
    output_bias: Tensor
    input_hidden_sigma: Tensor | None
    hidden_bias_sigma: Tensor | None
    hidden_output_sigma: Tensor | None
    output_bias_sigma: Tensor | None
    device: torch.device
    sigma_sigma: float | None

    @classmethod
    def random(
        cls,
        population_size: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
        initial_mutation_std: float,
        sigma_sigma: float | None,
        device: torch.device,
        seed: int,
    ) -> "TorchStaticMLPPopulation":
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
        scale = 0.5
        dtype = torch.float64
        input_hidden = torch.randn(
            population_size, input_size, hidden_size, generator=generator, device=device, dtype=dtype
        ) * scale
        hidden_bias = torch.randn(
            population_size, 1, hidden_size, generator=generator, device=device, dtype=dtype
        ) * scale
        hidden_output = torch.randn(
            population_size, hidden_size, output_size, generator=generator, device=device, dtype=dtype
        ) * scale
        output_bias = torch.randn(
            population_size, 1, output_size, generator=generator, device=device, dtype=dtype
        ) * scale
        if sigma_sigma is None:
            input_hidden_sigma = None
            hidden_bias_sigma = None
            hidden_output_sigma = None
            output_bias_sigma = None
        else:
            input_hidden_sigma = torch.full_like(input_hidden, initial_mutation_std)
            hidden_bias_sigma = torch.full_like(hidden_bias, initial_mutation_std)
            hidden_output_sigma = torch.full_like(hidden_output, initial_mutation_std)
            output_bias_sigma = torch.full_like(output_bias, initial_mutation_std)
        return cls(
            input_hidden=input_hidden,
            hidden_bias=hidden_bias,
            hidden_output=hidden_output,
            output_bias=output_bias,
            input_hidden_sigma=input_hidden_sigma,
            hidden_bias_sigma=hidden_bias_sigma,
            hidden_output_sigma=hidden_output_sigma,
            output_bias_sigma=output_bias_sigma,
            device=device,
            sigma_sigma=sigma_sigma,
        )

    @property
    def population_size(self) -> int:
        return int(self.input_hidden.shape[0])

    def forward_logits_batch(self, observations: np.ndarray) -> Tensor:
        x = torch.as_tensor(observations, dtype=self.input_hidden.dtype, device=self.device)
        x = x.unsqueeze(0).expand(self.population_size, -1, -1)
        hidden = torch.bmm(x, self.input_hidden) + self.hidden_bias
        hidden = torch.tanh(hidden)
        return torch.bmm(hidden, self.hidden_output) + self.output_bias

    def mean_mutation_std(self, fallback_mutation_std: float) -> float:
        sigma_values = [
            self.input_hidden_sigma,
            self.hidden_bias_sigma,
            self.hidden_output_sigma,
            self.output_bias_sigma,
        ]
        present = [float(value.mean().item()) for value in sigma_values if value is not None]
        if not present:
            return fallback_mutation_std
        return float(np.mean(present))

    def select(self, indices: Tensor) -> "TorchStaticMLPPopulation":
        return TorchStaticMLPPopulation(
            input_hidden=self.input_hidden[indices].clone(),
            hidden_bias=self.hidden_bias[indices].clone(),
            hidden_output=self.hidden_output[indices].clone(),
            output_bias=self.output_bias[indices].clone(),
            input_hidden_sigma=None if self.input_hidden_sigma is None else self.input_hidden_sigma[indices].clone(),
            hidden_bias_sigma=None if self.hidden_bias_sigma is None else self.hidden_bias_sigma[indices].clone(),
            hidden_output_sigma=None if self.hidden_output_sigma is None else self.hidden_output_sigma[indices].clone(),
            output_bias_sigma=None if self.output_bias_sigma is None else self.output_bias_sigma[indices].clone(),
            device=self.device,
            sigma_sigma=self.sigma_sigma,
        )

    def topk(self, fitness: np.ndarray, elite_count: int) -> tuple[Tensor, np.ndarray]:
        fitness_tensor = torch.as_tensor(fitness, dtype=self.input_hidden.dtype, device=self.device)
        elite_fitness, elite_indices = torch.topk(fitness_tensor, k=elite_count, largest=True, sorted=True)
        return elite_indices, elite_fitness.detach().cpu().numpy()

    def reproduce(
        self,
        elite_indices: Tensor,
        mutation_std: float,
        seed: int,
    ) -> "TorchStaticMLPPopulation":
        elites = self.select(elite_indices)
        elite_count = int(elite_indices.shape[0])
        if elite_count == self.population_size:
            return elites

        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        child_count = self.population_size - elite_count
        parent_choice = torch.randint(
            low=0,
            high=elite_count,
            size=(child_count,),
            generator=generator,
            device=self.device,
        )
        child_input_hidden = elites.input_hidden[parent_choice].clone()
        child_hidden_bias = elites.hidden_bias[parent_choice].clone()
        child_hidden_output = elites.hidden_output[parent_choice].clone()
        child_output_bias = elites.output_bias[parent_choice].clone()

        child_input_hidden_sigma = (
            None if elites.input_hidden_sigma is None else elites.input_hidden_sigma[parent_choice].clone()
        )
        child_hidden_bias_sigma = (
            None if elites.hidden_bias_sigma is None else elites.hidden_bias_sigma[parent_choice].clone()
        )
        child_hidden_output_sigma = (
            None if elites.hidden_output_sigma is None else elites.hidden_output_sigma[parent_choice].clone()
        )
        child_output_bias_sigma = (
            None if elites.output_bias_sigma is None else elites.output_bias_sigma[parent_choice].clone()
        )

        child_input_hidden, child_input_hidden_sigma = _mutate_tensor(
            child_input_hidden,
            child_input_hidden_sigma,
            mutation_std,
            self.sigma_sigma,
            generator,
        )
        child_hidden_bias, child_hidden_bias_sigma = _mutate_tensor(
            child_hidden_bias,
            child_hidden_bias_sigma,
            mutation_std,
            self.sigma_sigma,
            generator,
        )
        child_hidden_output, child_hidden_output_sigma = _mutate_tensor(
            child_hidden_output,
            child_hidden_output_sigma,
            mutation_std,
            self.sigma_sigma,
            generator,
        )
        child_output_bias, child_output_bias_sigma = _mutate_tensor(
            child_output_bias,
            child_output_bias_sigma,
            mutation_std,
            self.sigma_sigma,
            generator,
        )

        return TorchStaticMLPPopulation(
            input_hidden=torch.cat([elites.input_hidden, child_input_hidden], dim=0),
            hidden_bias=torch.cat([elites.hidden_bias, child_hidden_bias], dim=0),
            hidden_output=torch.cat([elites.hidden_output, child_hidden_output], dim=0),
            output_bias=torch.cat([elites.output_bias, child_output_bias], dim=0),
            input_hidden_sigma=(
                None
                if elites.input_hidden_sigma is None
                else torch.cat([elites.input_hidden_sigma, child_input_hidden_sigma], dim=0)
            ),
            hidden_bias_sigma=(
                None
                if elites.hidden_bias_sigma is None
                else torch.cat([elites.hidden_bias_sigma, child_hidden_bias_sigma], dim=0)
            ),
            hidden_output_sigma=(
                None
                if elites.hidden_output_sigma is None
                else torch.cat([elites.hidden_output_sigma, child_hidden_output_sigma], dim=0)
            ),
            output_bias_sigma=(
                None
                if elites.output_bias_sigma is None
                else torch.cat([elites.output_bias_sigma, child_output_bias_sigma], dim=0)
            ),
            device=self.device,
            sigma_sigma=self.sigma_sigma,
        )


def _torch_dtype_from_numpy(dtype: np.dtype) -> torch.dtype:
    if np.issubdtype(dtype, np.float32):
        return torch.float32
    return torch.float64


def _mutate_tensor(
    values: Tensor,
    sigmas: Tensor | None,
    mutation_std: float,
    sigma_sigma: float | None,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor | None]:
    if sigmas is not None and sigma_sigma is not None:
        sigma_noise = torch.randn(sigmas.shape, generator=generator, device=values.device, dtype=values.dtype)
        sigmas = torch.clamp(sigmas * (1.0 + sigma_noise * sigma_sigma), min=1e-6)
        noise_std = sigmas
    else:
        noise_std = mutation_std
    noise = torch.randn(values.shape, generator=generator, device=values.device, dtype=values.dtype) * noise_std
    return values + noise, sigmas
