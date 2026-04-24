# 014_ga_ppo_action_clone

Minimal GA counterpart to Unit 13's supervised PPO behavior clone.

This unit:

- reuses the PPO state-action dataset shape from Unit 13
- fits a small policy with genetic search instead of gradient descent
- reports train/validation action fit
- evaluates the best genome closed loop on the Unit 12 benchmark seeds

Run a bounded smoke test:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\train_and_eval_ga_clone.py --device cuda --time-budget-s 6
```

Current smoke result:

- train/val examples: `1557 / 389`
- best validation action accuracy: `0.9846`
- closed-loop benchmark result: `500.0` mean return over `7` completed Unit 12 seeds
- benchmark deltas vs PPO:
  - return mean: `0.0`
  - action=`1` mean rate: `0.0`
  - action-switch-rate mean: `0.0449`
