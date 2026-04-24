# 013_supervised_ppo_behavior_clone

Plain supervised behavior-cloning baseline for the Unit 3 PPO CartPole policy.

This unit:

- collects PPO state-action data on training seeds
- trains a small PyTorch policy by supervised learning with a train/validation split
- evaluates that policy closed loop on the frozen Unit 12 benchmark seeds
- reports benchmark deltas against the PPO reference

Run a launch-bounded smoke test:

```powershell
$proc = Start-Process -FilePath C:\Users\Max\venv\Scripts\python.exe -ArgumentList '.\train_and_eval_clone.py','--device','cuda','--time-budget-s','6','--train-episodes','32' -PassThru -WorkingDirectory .
try {
    Wait-Process -Id $proc.Id -Timeout 9 -ErrorAction Stop
} catch {
    $proc.Refresh()
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    throw
}
```

Current smoke result:

- dataset steps: `1946` from `4` collected PPO training episodes before the budget cutover
- train/val split: `1557 / 389`
- best validation action accuracy: `0.9769`
- closed-loop benchmark result: `469.0` mean return over `3` completed Unit 12 seeds
- benchmark deltas vs PPO:
  - return mean: `31.0`
  - action=`1` mean rate: `0.00041`
  - action-switch-rate mean: `0.0425`
