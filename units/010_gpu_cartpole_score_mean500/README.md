# 010_gpu_cartpole_score_mean500

Next active experiment: on the GPU-native CartPole environment path, try to reach perfect population mean reward `500.00` on the direct score task for both static and dynamic networks.

Planned order:

1. static network baseline on the new GPU environment
2. dynamic network baseline on the new GPU environment
3. compare wall-clock time and reward progression under matched time budgets

First static smoke run:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\static_gpu_cartpole_score.py --device cuda --time-budget-s 8 --population-size 64 --elite-count 8 --episodes-per-individual 3
```

Current static result:

- best reward `500.00` by generation 2
- peak population mean `410.79` at `7.688s`

Current dynamic result:

- best reward `500.00` by generation 2
- peak population mean `432.93` at `6.701s`
