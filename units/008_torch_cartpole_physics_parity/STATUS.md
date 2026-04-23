# STATUS.md

## Purpose

- Test whether Gymnasium CartPole physics can be ported to Torch/GPU without changing behavior.

## Contents

- `code/torch_cartpole.py`: Torch implementation of Gymnasium CartPole step physics.
- `code/parity_check.py`: deterministic Gym-vs-Torch rollout comparisons.
- `plot/`: parity metrics and drift plot.

## Verification

- Verified on April 23, 2026 using Gymnasium `1.2.3` and `C:\Users\Max\venv\Scripts\python.exe` on the AMD Radeon RX 7800 XT machine.
- Command: `python .\parity_check.py --rollouts 32 --steps 200 --device cuda`, run with a 20-second external hard timeout.
- The test compares only valid pre-termination rollout steps; observed rollout lengths ranged from 9 to 47 steps.
- Torch CPU float64 max absolute state error vs Gymnasium internal float64 state: `8.882e-16`.
- Torch GPU float64 max absolute state error: `1.998e-15`.
- Torch GPU float32 max absolute state error: `8.405e-07`.
- Conclusion: float64 Torch physics matches Gymnasium at numerical precision; float32 GPU drift is small enough to test for speed later, but reward-training units should start from float64 if exact parity matters.
- Artifacts: `plot/metrics.csv`, `plot/summary.csv`, and `plot/parity_drift.svg`.

## Next Steps

- If used for training, build the GPU-native evaluator from `code/torch_cartpole.py` and keep a parity smoke test in the loop.
