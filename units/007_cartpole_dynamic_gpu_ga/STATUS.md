# STATUS.md

## Purpose

- Restore a GPU-batched dynamic-network CartPole GA path, closer to the old codebase than Unit 6.

## Contents

- `code/dynamic_gpu_cartpole_ga.py`: self-contained Torch dynamic population with Gymnasium vector evaluation.
- `code/plot_metrics.py`: reward/topology curve generation.
- `plot/`: metrics and curve artifacts.
- `model/`: best evolved GPU-batched dynamic genome.

## Port Notes

- Source idea: old `DynamicNets` plus `GymScoreEval`.
- TorchRL is not installed on this AMD machine, so this unit uses Gymnasium `SyncVectorEnv` with GPU network compute.
- Environment stepping remains CPU-bound; network forward and reproduction run on the AMD ROCm GPU through PyTorch's `cuda` device.

## Verification

- Verified on April 23, 2026 on the AMD Radeon RX 7800 XT machine using PyTorch `2.9.1+rocm7.2.1`; PyTorch exposes the ROCm GPU through `device=cuda`.
- Command used an internal 6-second budget plus an 8-second external hard process timeout: `python .\dynamic_gpu_cartpole_ga.py --device cuda --time-budget-s 6 --generations 100000 --population-size 16 --elite-count 4 --episodes-per-individual 1`.
- The external hard timeout stopped the process after about 8.5 seconds, and incremental flushing preserved usable artifacts.
- Saved metrics through generation 13; best reward reached `500.00`; peak mean reward was `493.875` at generation 12; final saved row was generation 13 with mean `361.50`.
- Artifacts: `plot/metrics.csv`, `plot/curve.svg`, and `model/best_genome.json`.

## Next Steps

- Treat this as a working but not yet optimized GPU-batched path.
- If continued, compare against Unit 6 using equal wall-clock budgets rather than equal generations.
