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
- The user may move between machines, currently including AMD Radeon RX 7800 XT and NVIDIA RTX 5070 Ti Laptop GPU systems, with more machines possible later.
- Environment setup should be documented per machine; do not assume one GPU or Python stack is permanent.
- Before future coding sessions, read this file and recent commits, then give a short orientation.

## Next Steps

- Use Unit 12 as the reference benchmark for the next PPO behavior-matching units.
- Keep per-machine GPU/Python setup notes current as machines are used.
