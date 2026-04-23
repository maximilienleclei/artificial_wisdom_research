# STATUS.md

## Current State

- `main` is a fresh restart branch as of April 22, 2026.
- Prior exploratory work has been reintroduced as rerunnable research units under `units/`.
- `005_mnist_minimal_pytorch` is a minimal PyTorch port of the old `../ai_research/projects/dl_classify_mnist` project, keeping compressed MNIST data and a standalone script instead of the old Lightning/Hydra/W&B stack.
- The old `to-be-deleted-archive-20260422` branch was deleted after the experiment archives were verified.
- No new active experiment has been selected yet.
- On the current NVIDIA RTX 5070 Ti Laptop GPU machine, `C:\Users\Max\venv` uses Python `3.14.3` with PyTorch `2.11.0+cu130`; CUDA is visible and reports one RTX 5070 Ti Laptop GPU.

## Working Agreement

- The user wants high-level, complete project direction while Codex handles implementation details under the hood.
- The user does not want to read repo files; chat is the only expected user interface.
- Repo docs are the durable memory for preferences, decisions, experiment results, constraints, artifacts, and next steps.
- Experiments and datasets should remain rerunnable/reusable, usually by archiving unit-specific code/data/results, but active work should not carry backwards-compatibility burden.
- Archived research units should use numbered folders like `XXX_name`.
- Units with plotted outputs should write those outputs under their own `plot/` folder; no-plot units should not create a `plot/` folder.
- Before building a new experiment, Codex should consider whether a rewrite is cleaner than extending the current code.
- The user may move between machines, currently including AMD Radeon RX 7800 XT and NVIDIA RTX 5070 Ti Laptop GPU systems, with more machines possible later.
- Environment setup should be documented per machine; do not assume one GPU or Python stack is permanent.
- Before future coding sessions, read this file and recent commits, then give a short orientation.

## Next Steps

- Choose the next research goal.
- Keep per-machine GPU/Python setup notes current as machines are used.
