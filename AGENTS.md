# AGENTS.md

1. Keep responses very short by default.
2. Treat repo docs as durable memory. Anything important enough to survive a session must be written down, not just acknowledged in chat.
3. The user wants to operate at a high level: report goals, results, constraints, and next decisions; keep implementation details under the hood unless asked.
4. The user does not want to read repo files. Communicate all user-facing state, decisions, and summaries through chat; repo docs are for agent continuity.
5. After any meaningful change, update the handoff docs so a new agent can quickly understand current experiments, decisions, results, constraints, artifacts, and next steps. In this repo, the main handoff doc is `STATUS.md`; relevant unit-local docs under `units/` and `BACKGROUND_RUNS.md` are secondary handoff docs when applicable.
6. After any meaningful verified codebase or workflow change, stage and commit it by default unless the user explicitly says not to commit yet.
7. Preserve experiments so they can be rerun, but do not burden active work with backwards compatibility.
8. Before implementing a new experiment, default to rewriting the codebase for that experiment and archiving the previous codebase. Only extend an existing codebase when there is substantial overlap such that a rewrite would mostly recreate the same code.
9. Organize archived research units as numbered folders named like `XXX_name`, and treat the repo as an evolving archive of unit-specific code/data/results with the current branch optimized for the active experiment.
10. Do not keep mutable executable experiment code outside `units/`. Repo-root `.md` files may stay shared, but any code that affects reruns must live in a numbered unit so later edits do not silently change archived reruns.
11. Expect the user to work across multiple machines. Record machine-specific environment facts, but do not assume one GPU or Python stack is permanent.
12. For archived units, put generated plots and plot-adjacent metrics in that unit's `plot/` folder; units without plots should not have a `plot/` folder.
13. Keep handoff docs future-useful and deletion-first: preserve only information that still affects future decisions, reproducibility, constraints, artifact discovery, or user preferences, and prune stale process history or temporary housekeeping.
14. Keep durable memory split by stability: put long-lived operating rules and foundations in shared repo docs, current truth and next decisions in `STATUS.md`, and experiment-specific or likely-to-age details inside the relevant unit docs.
15. Run experiments against explicit wall-clock time budgets by default. Time is the primary run contract; epochs, generations, iterations, and steps are secondary caps or reporting details, and the full budget should go to optimization unless the user explicitly asks for an in-run evaluation split or another stop rule.
16. Every shell/tool execution must have an explicit finite timeout by default. For bounded runs, the wall-clock cap must apply from process launch, enforced with an OS-level hard kill/watchdog plus any internal deadline, and startup/loading/teardown all count toward the budget unless the user says otherwise.
17. A bounded run is a safety slice, not evidence of convergence. Judge convergence from repeated bounded slices and saved curves, not from a single time-capped run.
18. Do not withhold inspectable progress. Any meaningful run, analysis, or workflow that evolves over time should flush useful intermediate state so status and convergence can be checked before exit.
19. Background work must stay invisible, trackable, and safe: use hidden/background-safe execution, unique run names, dedicated run-specific output directories, machine-readable status files, and enough information to stop the specific run without guessing.
20. Background runs must not overwrite canonical unit artifacts while still in progress. Prefer one active background run at a time unless the user asks for parallel runs, check existing status files before summarizing progress, use absolute or clearly unit-local output paths for detached runs, and on this Windows setup prefer `pythonw.exe` for the detached launcher when invisibility matters.
