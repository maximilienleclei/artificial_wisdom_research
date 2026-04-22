# STATUS.md

## Purpose

- Test whether a static MLP population evolved by truncation-selection GA can solve CartPole by reward.

## Verified Rerun

- Rerun on April 22, 2026 using `C:\Users\Max\venv`.
- Command: `python -m awr.experiments.cartpole_ga --generations 100 --population-size 64`
- Result: best reward reached `500.00` by generation 10 and stayed at `500.00` through generation 99.
- Final generation: `best=500.00`, `mean=463.29`.

## Artifacts

- Historical plot is stored in `plot/cartpole_annealing_curve.svg`.
