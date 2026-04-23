# 007_cartpole_dynamic_gpu_ga

GPU-batched dynamic-topology network GA on the Unit 1 CartPole reward task.

This restores the old codebase's useful GPU idea in a minimal form: Gymnasium vector environments still step on CPU, but the whole dynamic network population forwards and mutates as Torch tensors on `cuda`.

Run from `code/`:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\dynamic_gpu_cartpole_ga.py --device cuda --time-budget-s 6 --generations 100000 --population-size 16 --elite-count 4 --episodes-per-individual 1
```

Regenerate the plot:

```powershell
C:\Users\Max\venv\Scripts\python.exe .\plot_metrics.py
```

Verified timed run: a 6-second internal budget with an 8-second external hard timeout saved metrics through generation 13. Peak mean reward was `493.88`.
