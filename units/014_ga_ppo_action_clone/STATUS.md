# STATUS.md

## Purpose

- Provide the smallest fair GA analogue to Unit 13: same PPO action-cloning target style, but optimized by mutation/selection instead of gradient descent.

## Contents

- `code/train_and_eval_ga_clone.py`: bounded GA fitting and closed-loop benchmark evaluation.
- `code/plot_ga_results.py`: GA fitting and benchmark plot generation.
- `model/`: best genome and metrics JSON.
- `plot/`: training curve, benchmark comparison, and clone episode metrics.

## Notes

- This unit intentionally keeps the task narrow: action fitting first, closed-loop benchmark second.
- It is meant to answer whether a small GA can beat Unit 13 on the same benchmark without changing the target definition.
- As of April 23, 2026, the script uses unit-local default output paths and absolute cross-unit benchmark/dataset defaults so detached/background runs do not fail from relative path resolution.

## Verification

- Initial bounded smoke run on April 23, 2026 used the Unit 13 PPO dataset, `C:\Users\Max\venv\Scripts\python.exe`, and the AMD GPU machine.
- Saved smoke result:
  - train/val examples `1557 / 389`
  - best train accuracy `0.9705`
  - best validation accuracy `0.9846`
  - closed-loop benchmark completed `7` Unit 12 seeds
  - clone return mean/std `500.0 / 0.0`
  - clone action=`1` mean rate `0.5`
  - clone action-switch-rate mean `0.6224`
  - absolute delta vs PPO benchmark:
    - return mean `0.0`
    - action=`1` mean rate `0.0`
    - action-switch-rate mean `0.0449`
- Fairer same-dataset rerun on April 23, 2026 kept the same Unit 13 PPO dataset and the same `6s` optimizer budget:
  - same train/val examples `1557 / 389`
  - parameter count `1282`
  - optimization forward-example evals `73,147,680`
  - estimated optimization forward FLOPs `177,895,157,760`
  - best validation accuracy `0.9820`
  - closed-loop benchmark completed `6` Unit 12 seeds
  - clone return mean/std `500.0 / 0.0`
  - action-switch-rate mean delta vs PPO `0.0484`
- Verification fix on April 23, 2026 confirmed the path-default repair:
  - a `12s` timed verification slice completed successfully instead of failing on Unit 12 path lookup
  - verification result: best validation accuracy `0.9923`, clone return mean `500.0`

## Artifacts

- `model/best_genome.json`
- `model/metrics.json`
- `plot/ga_history.csv`
- `plot/benchmark_comparison.json`
- `plot/episode_metrics.csv`
- `plot/ga_results.svg`

## Next Steps

- Rerun Unit 14 with a larger time budget so it covers the full Unit 12 benchmark seed set.
- Keep using the same frozen dataset when comparing against Unit 13 so the optimizer difference stays isolated.
