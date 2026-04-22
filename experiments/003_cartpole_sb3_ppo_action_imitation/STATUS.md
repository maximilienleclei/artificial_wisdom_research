# STATUS.md

## Purpose

- Test whether a static MLP GA can imitate a trained SB3 PPO CartPole policy from rollout behavior.

## Verified Rerun

- Rerun on April 22, 2026 using `C:\Users\Max\venv`.
- CUDA was visible on the NVIDIA RTX 5070 Ti Laptop GPU.
- Command: `python -m awr.experiments.act_pred_sb3 --target-agent-path .\models\ppo-CartPole-v1.zip --target-agent-algo ppo --generations 100 --population-size 256`
- Archived 100-generation result was approximately `best=0.7928`, `mean=0.7915`.
- Fresh rerun ended higher: `best=0.8845`, `mean=0.8747`.

## Artifacts

- Fresh rerun outputs are in `plot/metrics.csv` and `plot/curve.svg`.
- PPO checkpoint is stored in `models/ppo-CartPole-v1.zip`.

## Notes

- SB3 emits checkpoint deserialization warnings for saved schedule objects on Python 3.14, but inference works.
