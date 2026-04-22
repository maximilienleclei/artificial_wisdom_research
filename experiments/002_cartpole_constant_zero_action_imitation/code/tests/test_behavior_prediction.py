import numpy as np
import torch

from awr.behavior_prediction import ActionPredictionConfig, ActionPredictionEvaluator
from awr.models import BatchedMLPPopulation, StaticMLPGenome


class ConstantZeroPolicy:
    def predict(
        self,
        observations: np.ndarray,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        return np.zeros(len(observations), dtype=np.int64), None


def test_discrete_action_prediction_rewards_matching_policy():
    genomes = [_genome_for_action_zero(), _genome_for_action_one()]
    evaluator = ActionPredictionEvaluator(
        config=ActionPredictionConfig(env_name="CartPole-v1", max_steps=20),
        target_policy=ConstantZeroPolicy(),
        num_workers=len(genomes),
    )

    try:
        fitness = evaluator.evaluate(genomes, generation=0)
    finally:
        evaluator.close()

    assert fitness.shape == (2,)
    assert fitness[0] > fitness[1]


def test_action_prediction_reports_env_rewards():
    genomes = [_genome_for_action_zero(), _genome_for_action_zero()]
    evaluator = ActionPredictionEvaluator(
        config=ActionPredictionConfig(env_name="CartPole-v1", max_steps=10),
        target_policy=ConstantZeroPolicy(),
        num_workers=len(genomes),
    )

    try:
        evaluator.evaluate(genomes, generation=0)
        metrics = evaluator.get_metrics()
    finally:
        evaluator.close()

    assert "env_rewards" in metrics
    assert metrics["env_rewards"].shape == (1,)
    assert metrics["env_rewards"][0] >= 0.0


def test_batched_population_matches_individual_forward():
    genomes = [_genome_for_action_zero(), _genome_for_action_one()]
    observations = np.array(
        [
            [0.1, -0.2, 0.3, -0.4],
            [-0.5, 0.4, -0.3, 0.2],
        ],
        dtype=np.float64,
    )

    batched = BatchedMLPPopulation.from_genomes(genomes, torch.device("cpu"))
    batched_logits = batched.forward_logits_batch(observations).detach().cpu().numpy()
    individual_logits = np.stack(
        [
            np.stack([genome.forward_logits(observation) for observation in observations])
            for genome in genomes
        ]
    )

    np.testing.assert_allclose(batched_logits, individual_logits)


def _genome_for_action_zero() -> StaticMLPGenome:
    return StaticMLPGenome(
        input_hidden=np.zeros((4, 2)),
        hidden_bias=np.zeros((2,)),
        hidden_output=np.array([[2.0, -2.0], [2.0, -2.0]]),
        output_bias=np.array([2.0, -2.0]),
    )


def _genome_for_action_one() -> StaticMLPGenome:
    return StaticMLPGenome(
        input_hidden=np.zeros((4, 2)),
        hidden_bias=np.zeros((2,)),
        hidden_output=np.array([[-2.0, 2.0], [-2.0, 2.0]]),
        output_bias=np.array([-2.0, 2.0]),
    )
