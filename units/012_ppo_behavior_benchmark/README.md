# 012_ppo_behavior_benchmark

Frozen closed-loop PPO behavior benchmark on the GPU-native Torch CartPole environment.

This unit turns the Unit 3 PPO checkpoint into a reusable reference target for later comparisons.

It saves:

- fixed-seed closed-loop rollout data
- episode-level return and behavior metrics
- aggregate behavior summary statistics
- a benchmark plot

Run a bounded benchmark collection:

```powershell
$proc = Start-Process -FilePath C:\Users\Max\venv\Scripts\python.exe -ArgumentList '.\benchmark_ppo_behavior.py','--device','cuda','--time-budget-s','8','--num-episodes','64' -PassThru -WorkingDirectory .
try {
    Wait-Process -Id $proc.Id -Timeout 10 -ErrorAction Stop
} catch {
    $proc.Refresh()
    if (-not $proc.HasExited) { Stop-Process -Id $proc.Id -Force }
    throw
}
```

Current verified benchmark snapshot:

- completed `13` fresh fixed-seed episodes inside an `8s` inner budget
- PPO matched perfect CartPole returns on every completed episode
- return mean/std: `500.0 / 0.0`
- episode length mean/std: `500.0 / 0.0`
- action=`1` rate mean: `0.5`
- action-switch-rate mean/std: `0.6673 / 0.0450`

Launch-bounded smoke confirmation:

- a `3s` inner budget plus `8s` outer watchdog completed `4/4` episodes in `2.401s`
- scratch smoke outputs matched the main benchmark pattern exactly: all returns `500.0`
