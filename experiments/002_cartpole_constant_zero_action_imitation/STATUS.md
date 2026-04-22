# STATUS.md

## Purpose

- Sanity-check behavior imitation by fitting a constant-zero CartPole action policy.

## Verified Rerun

- Rerun on April 22, 2026 using `C:\Users\Max\venv`.
- Command: `python -m awr.experiments.act_pred_cartpole --generations 40 --population-size 64`
- Result matched the archived conclusion: the policy fit is effectively perfect.
- Final generation: `best=1.0000`, `mean=1.0000`.

## Artifacts

- Fresh rerun outputs are in `plot/metrics.csv` and `plot/curve.svg`.
