# 003_cartpole_sb3_ppo_action_imitation

Static MLP genetic algorithm imitating a trained SB3 PPO CartPole checkpoint with batched Torch evaluation.

Run from this folder after installing `code/` into the active venv:

```powershell
C:\Users\Max\venv\Scripts\python.exe -m awr.experiments.act_pred_sb3 --target-agent-path .\models\ppo-CartPole-v1.zip --target-agent-algo ppo --generations 100 --population-size 256
```
