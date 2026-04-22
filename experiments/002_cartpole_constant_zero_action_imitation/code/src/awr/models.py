from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


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


def _torch_dtype_from_numpy(dtype: np.dtype) -> torch.dtype:
    if np.issubdtype(dtype, np.float32):
        return torch.float32
    return torch.float64
