# STATUS.md

## Purpose

- Establish the plain supervised-learning baseline for PPO behavior cloning before comparing against GA-based methods.

## Contents

- `code/train_and_eval_clone.py`: bounded PPO dataset collection, supervised training, and closed-loop benchmark evaluation.
- `code/plot_clone_results.py`: training/evaluation figure generation.
- `data/`: saved PPO training dataset.
- `model/`: trained clone checkpoint and metrics JSON.
- `plot/`: benchmark comparison outputs.

## Notes

- This is the baseline that the upcoming GA behavior-matching unit should beat on the same closed-loop benchmark.
- The supervised clone should report both training and validation action accuracy before the closed-loop benchmark comparison.
- As of April 23, 2026, the timed training loop no longer stops early on high validation accuracy; timed slices now run to the allotted wall-clock budget unless killed externally.
- As of April 23, 2026, the unit writes inspectable progress snapshots during dataset, training, and evaluation stages instead of waiting until process exit.
- As of April 23, 2026, the supervised optimizer uses `AdamW` with a linear-warmup cosine-decay learning-rate schedule instead of a fixed learning rate.
- As of April 23, 2026, training now runs continuously, validation is probed every configurable `val_interval_s`, and each validation event also runs a closed-loop behavior probe on the latest checkpoint.

## Verification

- Initial bounded smoke run on April 23, 2026 used `C:\Users\Max\venv\Scripts\python.exe` on the AMD GPU machine.
- Saved smoke result:
  - `1946` dataset steps collected from `4` PPO rollout episodes
  - train/val split `1557 / 389`
  - final train accuracy `0.9395`
  - best validation accuracy `0.9769`
  - closed-loop clone benchmark completed `3` Unit 12 seeds
  - clone return mean/std `469.0 / 43.84`
  - clone action=`1` mean rate `0.4996`
  - clone action-switch-rate mean `0.6248`
  - absolute delta vs PPO benchmark:
    - return mean `31.0`
    - action=`1` mean rate `0.00041`
    - action-switch-rate mean `0.0425`
- Fairer optimizer-only rerun on April 23, 2026 reused the saved PPO dataset from this unit instead of recollecting data:
  - same train/val examples `1557 / 389`
  - parameter count `1282`
  - optimization forward-example evals `332,766`
  - estimated optimization forward FLOPs `809,286,912`
  - best validation accuracy `0.9974`
  - closed-loop benchmark completed `2` Unit 12 seeds within the same `6s` total cap
  - clone return mean/std `490.0 / 10.0`
  - action-switch-rate mean delta vs PPO `0.0454`
- Verification fix on April 23, 2026 confirmed the early-stop removal and unit-local path defaults:
  - a `20s` timed verification slice ran through the full allotted slice (`elapsed_s=20.004`)
  - verification result: train accuracy `0.9578`, validation accuracy `1.0000`, clone return mean `461.10`
- Scheduler verification on April 23, 2026 confirmed the warmup+cosine path:
  - `15s` timed verification slice with final LR about `1.5e-4`
  - train accuracy `0.9961`
  - validation accuracy `1.0000`
  - closed-loop return mean `498.29`
  - action-switch-rate mean delta vs PPO `0.0158`
- Interval-validation verification on April 23, 2026 confirmed the simpler continuous-training pattern:
  - `20s` total run with `val_interval_s=5` and `train_fraction=0.9`
  - latest checkpoint validation accuracy `1.0000`
  - closed-loop return mean `482.75`
  - action-switch-rate mean delta vs PPO `0.0154`
- Behavior-in-validation verification on April 23, 2026 confirmed that validation checkpoints now include rollout metrics:
  - `20s` total run with `val_interval_s=5`
  - latest checkpoint validation accuracy `1.0000`
  - latest closed-loop return mean `417.8`
  - `closed_loop_evaluated=true` in the final metrics snapshot
- Five-minute rerun on April 23, 2026 with `val_interval_s=30` showed the behavior-probe trajectory over time:
  - early probe return mean `9.92`
  - from the second probe onward, return mean stayed at `500.0`
  - final probe used all `13` Unit 12 benchmark seeds
  - final action-switch-rate delta vs PPO `0.0`
  - comparison plot saved at `../014_ga_ppo_action_clone/plot/u013_u014_5min_behavior_compare.svg`

## Artifacts

- `data/ppo_train_dataset.csv`
- `model/clone_policy.pt`
- `model/metrics.json`
- `plot/benchmark_comparison.json`
- `plot/episode_metrics.csv`
- `plot/clone_results.svg`

## Next Steps

- Run longer bounded slices under the new `val_interval_s` setup and judge convergence from both the validation curve and the in-flight closed-loop behavior probes.
- Compare this baseline against Unit 14 as the fair same-dataset optimizer comparison.
