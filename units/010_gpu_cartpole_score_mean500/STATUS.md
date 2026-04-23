# STATUS.md

## Purpose

- Try to achieve perfect population mean reward `500.00` on the direct CartPole score task in the new GPU-native environment, first for static networks and then for dynamic networks.

## Planned Scope

- Static network score optimization on the GPU-native CartPole engine.
- Dynamic network score optimization on the GPU-native CartPole engine.
- Comparison under matched wall-clock budgets, not matched generations.

## Constraints

- Use explicit wall-clock budgets with internal deadlines and external hard timeouts.
- Report reward and throughput in wall-clock terms.
- The first goal is population mean `500.00`, not just best individual `500.00`.

## Verification

- Static-network GPU-native smoke run verified on April 23, 2026 using `C:\Users\Max\venv\Scripts\python.exe` on the AMD Radeon RX 7800 XT machine.
- Command: `python .\static_gpu_cartpole_score.py --device cuda --time-budget-s 8 --population-size 64 --elite-count 8 --episodes-per-individual 3`, with a 10-second external hard timeout.
- Best reward reached `500.00` by generation 2.
- Best saved population mean reached `410.79` at elapsed `7.688s` (generation 13).
- Final flushed row before timeout was generation 14 at elapsed `8.014s`, with `best=380.00`, `mean=302.19`; the prior peak mean is the more useful progress marker.
- Artifacts: `plot/static_metrics.csv`, `plot/static_curve.svg`, and `model/static_best.json`.
- Dynamic-network GPU-native smoke run verified on April 23, 2026 using `C:\Users\Max\venv\Scripts\python.exe` on the AMD Radeon RX 7800 XT machine.
- Command: `python .\dynamic_gpu_cartpole_score.py --device cuda --time-budget-s 8 --population-size 64 --elite-count 8 --episodes-per-individual 3`, with a 10-second external hard timeout.
- Dynamic best reward reached `500.00` by generation 2.
- Dynamic best saved population mean reached `432.93` at elapsed `6.701s` (generation 4), with mean hidden nodes `0.328`.
- Final flushed dynamic row before timeout was generation 6 at elapsed `8.017s`, with `best=231.67`, `mean=197.48`; again, the earlier peak mean is the useful progress marker.
- Dynamic artifacts: `plot/dynamic_metrics.csv`, `plot/dynamic_curve.svg`, and `model/dynamic_best.json`.
- Current comparison under matched 8-second budgets: dynamic beat the first static smoke run (`432.93` vs `410.79` peak mean), but the tuned static run still leads so far (`473.48` peak mean vs `432.93`).

## Next Steps

- Improve the static runner until it can hit population mean `500.00` under a reasonable wall-clock budget.
- Try one or two short dynamic tuning runs now that the baseline works.
