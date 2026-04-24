# BACKGROUND_RUNS.md

Background runs are allowed in this repo when they improve responsiveness, but they must follow a strict tracking pattern.

The executable workflow code for this pattern now lives in:

- `units/013_experiment_runtime_tooling/code/`

## Goals

- keep chat responsive while longer optimization continues
- avoid visible terminals or desktop popups
- avoid confusion about which files came from which run
- make progress easy to inspect and easy to stop

## Required Pattern

For any background run:

1. Give it a unique run name.
2. Write all in-progress outputs to a dedicated run directory first.
3. Write a machine-readable `run_status.json` file for that run.
4. Record:
   - unit
   - run name
   - command
   - working directory
   - machine / device
   - start time
   - scheduled stop deadline
   - main output paths
   - whether the run is `running`, `completed`, `stopped`, or `failed`
5. Keep canonical unit artifacts unchanged until the run completes and is verified.
6. Prefer absolute paths or explicitly unit-local paths for output arguments. Do not rely on ambiguous `..` path resolution in detached runs.

## Preferred Layout

Within a unit, prefer a structure like:

```text
model/runs/<run_name>/
plot/runs/<run_name>/
run_status.json
```

If both model and plot artifacts are produced, record both directories in `run_status.json`.

## Progress Tracking

- Append metrics incrementally inside the run-specific directory.
- Status checks should read the run status file and the latest metrics file, not guess from timestamps alone.
- If there is an active background run, check it before starting another one unless the user explicitly wants parallel runs.
- Do not defer all useful metrics until process exit. Runs should expose enough in-flight information that we can answer "how is it going?" without waiting for the slice to finish. This same principle applies to non-background work too.

## Stop Rules

- Every background run must have a predetermined stop deadline.
- The run must be stoppable using the information recorded in `run_status.json`.
- Do not rely on memory or "the latest python process" to stop a run.
- For time-budgeted optimization runs, do not stop early because a metric looks good enough. By default, the run should continue working until the deadline or an external kill stops it, unless the user explicitly requested another stop rule.

## Promotion Rule

Only after a background run finishes and the outputs are verified:

- copy or rename the chosen artifacts into the unit's canonical `model/` and `plot/` locations
- update the unit `STATUS.md`
- then commit, if git state allows it

## Validated Local Pattern

On this Windows setup, the validated invisible background pattern is:

- launch the detached worker with `pythonw.exe`
- keep the actual experiment under `python.exe`
- write progress to run-specific files
- inspect progress by reading `run_status.json`, `stdout.log`, `stderr.log`, and run metrics files
