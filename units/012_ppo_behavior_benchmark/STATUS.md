# STATUS.md

## Purpose

- Freeze a reusable closed-loop behavior benchmark for the Unit 3 PPO CartPole checkpoint on the Unit 8 Torch CartPole physics path.

## Contents

- `code/benchmark_ppo_behavior.py`: bounded PPO rollout benchmark collector.
- `code/plot_behavior_benchmark.py`: plot generation for benchmark artifacts.
- `data/`: fixed-seed rollout table for later comparisons.
- `plot/`: episode metrics, aggregate summary, and benchmark figure.

## Notes

- This unit is meant to be the common evaluation target for upcoming supervised, GA, and adversarial behavior-matching experiments.
- Primary comparison is closed-loop behavior on fresh fixed seeds, not just action fit on a collected dataset.
- The user-facing time budget must be enforced from process launch; use an OS-level watchdog wrapper around this script when collecting bounded runs.

## Verification

- Verified on April 23, 2026 using the saved Unit 3 PPO checkpoint, Unit 8 Torch CartPole physics, `C:\Users\Max\venv`, and the AMD GPU machine.
- The interrupted first collection still completed and saved a valid bounded snapshot before the process issue was diagnosed.
- Saved benchmark snapshot:
  - `13` completed episodes within an `8.03s` inner budget
  - return mean/std `500.0 / 0.0`
  - episode length mean/std `500.0 / 0.0`
  - action=`1` rate mean/std `0.5 / 0.0`
  - action-switch-rate mean/std `0.6673 / 0.0450`
- Launch-bounded smoke rerun also behaved correctly once wrapped with an explicit watchdog:
  - `4/4` episodes completed under a `3s` inner budget and `8s` outer kill window
  - elapsed `2.401s`
  - return mean/std `500.0 / 0.0`
- Artifacts:
  - `data/reference_rollouts.csv`
  - `plot/episode_metrics.csv`
  - `plot/summary.json`
  - `plot/behavior_benchmark.svg`

## Next Steps

- Use this frozen benchmark as the primary closed-loop reference for the next supervised and GA PPO-behavior-matching units.
- Compare future agents on benchmark metric deltas, not only on action-label fit.
