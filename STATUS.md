# STATUS.md

## Current State

- `main` is a fresh restart branch as of April 22, 2026.
- Prior exploratory work has been reintroduced as rerunnable research units under `units/`.
- `005_mnist_minimal_pytorch` is a minimal PyTorch port of the old `../ai_research/projects/dl_classify_mnist` project, keeping compressed MNIST data and a standalone script instead of the old Lightning/Hydra/W&B stack.
- `006_cartpole_dynamic_network_reward_ga` is a minimal dynamic-topology network port targeting the same CartPole reward task as Unit 1; the verified single-seed run reached best reward `500.00`, final mean `446.02`, and peak mean `491.93`.
- `007_cartpole_dynamic_gpu_ga` restores a GPU-batched dynamic-network path with Gymnasium vector envs on CPU and Torch population compute on the AMD ROCm GPU; a 6-second timed run saved metrics through generation 13 and reached peak mean reward `493.875`.
- `008_torch_cartpole_physics_parity` verifies a Torch/GPU CartPole physics port against Gymnasium before using it for training; float64 matched at numerical precision and GPU float32 max drift was `8.405e-07` over valid rollout steps.
- `009_cartpole_human_data_torch_engine_parity` found that the Unit 8 Torch CartPole engine reproduces Unit 4 human CartPole data closely enough to preserve final episode returns exactly across all recorded episodes, despite some long-horizon state drift.
- `010_gpu_cartpole_score_mean500` is the current active experiment: under matched 8-second GPU-native CartPole score smoke runs, the static runner reached peak mean `410.79`, the dynamic runner reached peak mean `432.93`, and a tuned static variant reached `473.48`; next step is to push population mean to `500.00`.
- `011_gpu_cartpole_adversarial_generation` is a minimal GPU-native CartPole adversarial-generation port using the Unit 3 PPO checkpoint as target; the verified static smoke run reached best mean total fitness `0.8964` and mean generator environment reward `63.56`.
- `012_ppo_behavior_benchmark` freezes the PPO reference behavior for future imitation comparisons; the saved bounded snapshot completed `13` fixed-seed episodes in about `8s`, all with return `500.0`, and recorded rollout/action statistics for later metric deltas.
- `013_supervised_ppo_behavior_clone` is the first plain behavior-cloning baseline; an initial smoke run reached best validation action accuracy `0.9769` but only `469.0` mean closed-loop return over `3` completed benchmark seeds, already showing a gap between action fit and rollout behavior.
- `014_ga_ppo_action_clone` is the GA counterpart to Unit 13; an initial smoke run reached best validation action accuracy `0.9846` and `500.0` mean closed-loop return over `7` completed benchmark seeds while still differing from PPO on action-switch rate.
- A fairer same-dataset rerun now exists for Units 13 and 14: both used the same frozen PPO dataset and the same `6s` optimization cap. Under that setup, Unit 13 reached `0.9974` validation accuracy and `490.0` mean closed-loop return over `2` completed benchmark seeds, while Unit 14 reached `0.9820` validation accuracy and `500.0` mean closed-loop return over `6` completed benchmark seeds. The GA consumed far more forward evaluations, so wall-clock fairness currently favors outcome comparison more than FLOP fairness.
- Units 13 and 14 were repaired on April 23, 2026 for proper convergence slices: Unit 13 no longer stops early on high validation accuracy, and both units now use robust unit-local / absolute default paths that survive detached background launches.
- Units 13 and 14 now also flush inspectable progress snapshots mid-run, so convergence/status checks no longer have to wait for process exit to see useful state.
- Unit 13 now uses `AdamW` plus a linear-warmup cosine-decay schedule rather than a fixed LR; a short verification slice reached validation accuracy `1.0000`, return mean `498.29`, and switch-rate delta `0.0158`.
- Units 13 and 14 now train/evolve continuously, run validation probes every configurable `val_interval_s`, and reserve the final 10% of the wall-clock budget for the final closed-loop behavior benchmark.
- Known machine environment: NVIDIA RTX 5070 Ti Laptop GPU with `C:\Users\Max\venv`, Python `3.14.3`, PyTorch `2.11.0+cu130`; CUDA reports one RTX 5070 Ti Laptop GPU.
- Known machine environment: AMD Radeon RX 7800 XT with `C:\Users\Max\venv`, PyTorch `2.9.1+rocm7.2.1`; PyTorch reports `cuda=True`, HIP `7.2.53211-158bd99533`, and device name `AMD  Radeon RX 7800 XT`.

## Working Agreement

- The user wants high-level, complete project direction while Codex handles implementation details under the hood.
- The user does not want to read repo files; chat is the only expected user interface.
- Repo docs are the durable memory for preferences, decisions, experiment results, constraints, artifacts, and next steps.
- Experiments and datasets should remain rerunnable/reusable, usually by archiving unit-specific code/data/results, but active work should not carry backwards-compatibility burden.
- Archived research units should use numbered folders like `XXX_name`.
- Units with plotted outputs should write those outputs under their own `plot/` folder; no-plot units should not create a `plot/` folder.
- Before building a new experiment, Codex should consider whether a rewrite is cleaner than extending the current code.
- Experiments should default to explicit wall-clock budgets; generations, epochs, iterations, and steps are secondary caps or reported outcomes. Enforce budgets with both internal deadlines and external hard process timeouts, and flush artifacts incrementally.
- Every shell/tool execution should use an explicit finite timeout by default; anything beyond a brief read/listing needs a hard kill path.
- Strict time caps must apply from process launch, not from after Python startup or model loading. For bounded runs, use an OS-level kill wrapper in the command itself in addition to any internal deadline.
- Background runs must remain invisible to the user; do not use execution patterns that pop open visible terminal windows or consoles.
- Time-bounded runs are safety slices, not convergence claims. Convergence should be judged across repeated bounded slices from saved optimization curves, not inferred from a single timeout-limited run.
- For time-budgeted runs, elapsed wall-clock time is the default stop contract. Do not treat target accuracy, plateau checks, solved thresholds, or fixed epoch counts as implicit early-stop rules unless the user explicitly asked for them.
- Useful state should not be hidden until exit. Runs and analyses that progress over time should flush inspectable progress artifacts during execution so status and convergence can be checked mid-run.
- When using background runs, keep them fully trackable with unique run names, run-specific output directories, and machine-readable status files. Do not overwrite canonical artifacts until the background run finishes and is verified.
- On this Windows setup, the currently validated invisible launcher pattern uses `pythonw.exe` for the detached worker and explicit run-specific output paths.
- The user may move between machines, currently including AMD Radeon RX 7800 XT and NVIDIA RTX 5070 Ti Laptop GPU systems, with more machines possible later.
- Environment setup should be documented per machine; do not assume one GPU or Python stack is permanent.
- Before future coding sessions, read this file and recent commits, then give a short orientation.

## Next Steps

- Use Unit 12 as the reference benchmark for the next PPO behavior-matching units.
- Use Unit 13 as the supervised baseline that future GA behavior-matching runs should try to beat.
- Push Units 13 and 14 toward convergence under the new continuous-train plus interval-validation setup, then compare them on full benchmark coverage.
- Separate optimization budget from evaluation budget in the next comparison pass so benchmark coverage is matched cleanly while keeping the same frozen dataset.
- For convergence studies, continue optimization in repeated bounded slices and decide convergence from the saved curves rather than from one slice length.
- If background runs are used for convergence studies, keep them one-at-a-time by default and promote outputs to canonical unit artifacts only after the run is complete and checked.
- Keep per-machine GPU/Python setup notes current as machines are used.
