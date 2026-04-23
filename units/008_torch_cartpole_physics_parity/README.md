# 008_torch_cartpole_physics_parity

Torch CartPole physics parity tests against Gymnasium.

This unit does not train anything. It checks whether a Torch implementation of the Gymnasium CartPole step equations matches Gymnasium under fixed initial states and fixed action sequences.

Run from `code/`:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\parity_check.py --rollouts 128 --steps 500 --device cuda
```

Verified short check: Torch CPU/GPU float64 matched Gymnasium to numerical precision; Torch GPU float32 max state drift was `8.405e-07` over valid pre-termination rollout steps.
