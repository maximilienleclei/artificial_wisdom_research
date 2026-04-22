from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from awr.models import StaticMLPGenome, TorchStaticMLPPopulation


class TargetPolicy(Protocol):
    def predict(
        self,
        observations: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, Any]:
        ...


@dataclass(slots=True)
class ActionPredictionConfig:
    env_name: str
    max_steps: int = 200
    base_seed: int = 7
    device: str | None = None


class ActionPredictionEvaluator:
    def __init__(
        self,
        config: ActionPredictionConfig,
        target_policy: TargetPolicy,
        num_workers: int,
    ) -> None:
        self.config = config
        self.target_policy = target_policy
        self.env = gym.make(config.env_name)
        self._obs_space = self.env.observation_space
        self._action_space = self.env.action_space
        self._is_discrete = isinstance(self._action_space, spaces.Discrete)
        self._last_env_rewards = np.zeros(1, dtype=np.float64)
        self.device = _resolve_torch_device(config.device)

    @classmethod
    def from_sb3(
        cls,
        config: ActionPredictionConfig,
        target_agent_path: str,
        target_agent_algo: str,
        num_workers: int,
    ) -> "ActionPredictionEvaluator":
        from stable_baselines3 import A2C, DQN, PPO, SAC, TD3

        algo_map = {"ppo": PPO, "sac": SAC, "td3": TD3, "a2c": A2C, "dqn": DQN}
        algo_cls = algo_map.get(target_agent_algo.lower())
        if algo_cls is None:
            raise ValueError(
                f"Unknown algo: {target_agent_algo}. Supported: {sorted(algo_map)}"
            )
        target_policy = algo_cls.load(target_agent_path)
        return cls(config=config, target_policy=target_policy, num_workers=num_workers)

    def retrieve_num_inputs_outputs(self) -> tuple[int, int]:
        obs_dim = int(self._obs_space.shape[0])
        if self._is_discrete:
            return obs_dim, int(self._action_space.n)
        return obs_dim, int(self._action_space.shape[0])

    def retrieve_action_space(self) -> spaces.Space:
        return self._action_space

    def retrieve_input_output_specs(self) -> tuple[spaces.Space, spaces.Space]:
        return self._obs_space, self._action_space

    def evaluate(
        self,
        genomes: list[StaticMLPGenome] | TorchStaticMLPPopulation,
        generation: int = 0,
    ) -> np.ndarray:
        if isinstance(genomes, TorchStaticMLPPopulation):
            num_genomes = genomes.population_size
            batched_population = genomes
        else:
            num_genomes = len(genomes)
            raise TypeError("The active SB3 imitation path expects a TorchStaticMLPPopulation.")
        observations, target_actions, env_reward = self._collect_target_dataset(generation)
        pred_logits = batched_population.forward_logits_batch(observations)

        if self._is_discrete:
            probs = torch.softmax(pred_logits, dim=-1)
            target_indices = torch.as_tensor(target_actions, dtype=torch.int64, device=pred_logits.device)
            target_indices = target_indices.view(1, -1, 1).expand(num_genomes, -1, 1)
            correct_probs = probs.gather(dim=-1, index=target_indices).squeeze(-1)
            fitness = correct_probs.mean(dim=1).detach().cpu().numpy()
        else:
            target_actions_arr = np.asarray(target_actions, dtype=np.float64)
            pred_actions = _map_to_action_space(
                logits=pred_logits,
                action_space=self._action_space,
            )
            mse = ((pred_actions - target_actions_arr[None, :, :]) ** 2).mean(axis=(1, 2))
            fitness = -mse

        self._last_env_rewards = np.array([env_reward], dtype=np.float64)
        return fitness

    def get_metrics(self) -> dict[str, np.ndarray]:
        return {"env_rewards": self._last_env_rewards.copy()}

    def close(self) -> None:
        self.env.close()

    def _collect_target_dataset(
        self,
        generation: int,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        observations: list[np.ndarray] = []
        target_actions: list[np.ndarray | int | float] = []
        observation, _ = self.env.reset(seed=self.config.base_seed + generation)
        env_reward = 0.0

        for _ in range(self.config.max_steps):
            obs_array = np.asarray(observation, dtype=np.float64)
            predicted_actions, _ = self.target_policy.predict(
                obs_array[None, :],
                deterministic=True,
            )
            target_action = np.asarray(predicted_actions)[0]
            observations.append(obs_array)
            target_actions.append(target_action)
            observation, reward, terminated, truncated, _ = self.env.step(target_action)
            env_reward += float(reward)
            if terminated or truncated:
                break

        return np.stack(observations), np.asarray(target_actions), env_reward


def _map_to_action_space(logits: torch.Tensor, action_space: spaces.Box) -> np.ndarray:
    low = np.asarray(action_space.low, dtype=np.float64)
    high = np.asarray(action_space.high, dtype=np.float64)
    low_tensor = torch.as_tensor(low, dtype=logits.dtype, device=logits.device)
    high_tensor = torch.as_tensor(high, dtype=logits.dtype, device=logits.device)
    squashed = torch.tanh(logits)
    mapped = (high_tensor + low_tensor) / 2.0 + squashed * (high_tensor - low_tensor) / 2.0
    return mapped.detach().cpu().numpy()


def _resolve_torch_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
