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
- `013_experiment_runtime_tooling` archives the executable background-run and status-polling tooling that later convergence studies depend on.
- `014_supervised_ppo_behavior_clone` is the first plain behavior-cloning baseline; an initial smoke run reached best validation action accuracy `0.9769` but only `469.0` mean closed-loop return over `3` completed benchmark seeds, already showing a gap between action fit and rollout behavior.
- `015_ga_ppo_action_clone` is the GA counterpart to Unit 14; an initial smoke run reached best validation action accuracy `0.9846` and `500.0` mean closed-loop return over `7` completed benchmark seeds while still differing from PPO on action-switch rate.
- A fairer same-dataset rerun now exists for Units 14 and 15: both used the same frozen PPO dataset and the same `6s` optimization cap. Under that setup, Unit 14 reached `0.9974` validation accuracy and `490.0` mean closed-loop return over `2` completed benchmark seeds, while Unit 15 reached `0.9820` validation accuracy and `500.0` mean closed-loop return over `6` completed benchmark seeds. The GA consumed far more forward evaluations, so wall-clock fairness currently favors outcome comparison more than FLOP fairness.
- Units 14 and 15 were repaired on April 23, 2026 for proper convergence slices: Unit 14 no longer stops early on high validation accuracy, and both units now use robust unit-local / absolute default paths that survive detached background launches.
- Units 14 and 15 now also flush inspectable progress snapshots mid-run, so convergence/status checks no longer have to wait for process exit to see useful state.
- Unit 14 now uses `AdamW` plus a linear-warmup cosine-decay schedule rather than a fixed LR; a short verification slice reached validation accuracy `1.0000`, return mean `498.29`, and switch-rate delta `0.0158`.
- Units 14 and 15 now use the full wall-clock budget for optimization, run validation probes every configurable `val_interval_s`, and include a closed-loop behavior probe at each validation event using the latest checkpoint/genome.
- In matched 5-minute reruns with in-validation behavior probes, Unit 15 reached strong closed-loop behavior earlier, but Unit 14 caught up and finished best: both ended at return mean `500.0` over all `13` Unit 12 benchmark seeds, with final action-switch-rate delta `0.0` for Unit 14 versus about `0.0089` for Unit 15.
- Known machine environment: NVIDIA RTX 5070 Ti Laptop GPU with `C:\Users\Max\venv`, Python `3.14.3`, PyTorch `2.11.0+cu130`; CUDA reports one RTX 5070 Ti Laptop GPU.
- Known machine environment: AMD Radeon RX 7800 XT with `C:\Users\Max\venv`, PyTorch `2.9.1+rocm7.2.1`; PyTorch reports `cuda=True`, HIP `7.2.53211-158bd99533`, and device name `AMD  Radeon RX 7800 XT`.

## Working Agreement

- The user wants high-level, complete project direction while Codex handles implementation details under the hood.
- The user does not want to read repo files; chat is the only expected user interface.
- Repo docs are the durable memory for preferences, decisions, experiment results, constraints, artifacts, and next steps.
- Durable docs should stay small, deletion-first, and split by stability: keep long-lived operating rules and foundations in shared docs, current truth and next decisions in `STATUS.md`, and experiment-specific or likely-to-age details inside the relevant unit docs.
- Experiments and datasets should remain rerunnable/reusable, usually by archiving unit-specific code/data/results, but active work should not carry backwards-compatibility burden.
- Mutable executable code should not live outside `units/`; repo-root docs are fine to keep shared, but runnable code that affects experiment replay should live in a numbered unit so later edits cannot silently change archived reruns.
- Archived research units should use numbered folders like `XXX_name`.
- Units with plotted outputs should write those outputs under their own `plot/` folder; no-plot units should not create a `plot/` folder.
- Before building a new experiment, default to a rewrite plus archive rather than extending the current code. Reuse/extension is the exception and should happen only when there is substantial overlap such that a rewrite would mostly recreate the same code.
- Experiments should default to explicit wall-clock budgets, with elapsed wall-clock time as the default stop contract and optimization using the full budget unless the user asks for another split or stop rule. Generations, epochs, iterations, and steps are secondary caps or reporting details.
- Every shell/tool execution should use an explicit finite timeout by default. For bounded runs, the wall-clock cap must apply from process launch via an OS-level hard kill/watchdog plus any internal deadline, and startup/loading/teardown all count toward the budget unless the user says otherwise.
- Time-bounded runs are safety slices, not convergence claims. Judge convergence across repeated bounded slices and saved curves, not from one capped run.
- Useful progress should not be hidden until exit. Runs and analyses that evolve over time should flush inspectable intermediate state so status and convergence can be checked mid-run.
- Background runs must stay invisible, trackable, and safe: use hidden/background-safe execution, unique run names, dedicated run-specific output directories, machine-readable status files, and enough information to stop the specific run without guessing.
- Background runs must not overwrite canonical artifacts while in progress. Prefer one active background run at a time unless the user asks for parallel runs, check status files before summarizing progress, use absolute or clearly unit-local output paths for detached runs, and on this Windows setup prefer `pythonw.exe` for the detached launcher when invisibility matters.
- The user may move between machines, currently including AMD Radeon RX 7800 XT and NVIDIA RTX 5070 Ti Laptop GPU systems, with more machines possible later.
- Environment setup should be documented per machine; do not assume one GPU or Python stack is permanent.
- Before future coding sessions, read this file and recent commits, then give a short orientation.

## Next Steps

- Use Unit 12 as the reference benchmark for the next PPO behavior-matching units.
- Use Unit 14 as the supervised baseline that future GA behavior-matching runs should try to beat.
- Push Units 14 and 15 toward convergence under the new continuous-train plus interval-validation setup, then compare them on full benchmark coverage.
- Separate optimization budget from evaluation budget in the next comparison pass so benchmark coverage is matched cleanly while keeping the same frozen dataset.
- For convergence studies, continue optimization in repeated bounded slices and decide convergence from the saved curves rather than from one slice length.
- If background runs are used for convergence studies, keep them one-at-a-time by default and promote outputs to canonical unit artifacts only after the run is complete and checked.
- Keep per-machine GPU/Python setup notes current as machines are used.
