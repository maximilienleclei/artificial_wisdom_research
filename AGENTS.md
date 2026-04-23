# AGENTS.md

1. Keep responses very short by default.
2. Treat repo docs as durable memory. Anything important enough to survive a session must be written down, not just acknowledged in chat.
3. The user wants to operate at a high level: report goals, results, constraints, and next decisions; keep implementation details under the hood unless asked.
4. The user does not want to read repo files. Communicate all user-facing state, decisions, and summaries through chat; repo docs are for agent continuity.
5. After any meaningful change, update the handoff docs so a new agent can quickly understand current experiments, decisions, results, constraints, artifacts, and next steps.
6. After any meaningful verified codebase or workflow change, stage and commit it by default unless the user explicitly says not to commit yet.
7. Preserve experiments so they can be rerun, but do not burden active work with backwards compatibility.
8. Before implementing a new experiment, decide whether it is cleaner to rewrite the codebase for that experiment and archive the previous codebase.
9. Organize archived research units as numbered folders named like `XXX_name`.
10. Treat the repo as an evolving archive of unit-specific code/data/results, with the current branch optimized for the active experiment.
11. Expect the user to work across multiple machines. Record machine-specific environment facts, but do not assume one GPU or Python stack is permanent.
12. For archived units, put generated plots and plot-adjacent metrics in that unit's `plot/` folder; units without plots should not have a `plot/` folder.
13. When optimizing experiments, report concrete throughput or outcome numbers rather than relying on epochs alone.
14. Keep handoff docs future-useful and prune stale process history. Do not preserve notes about completed branch cleanup, temporary migration mechanics, or one-off housekeeping unless they affect future decisions, reproducibility, constraints, or artifact discovery.
15. Run experiments against explicit wall-clock time budgets by default. For neuroevolution, deep learning, and similar workflows, use time as the primary run contract; generations, epochs, iterations, and steps may be secondary caps or reported outcomes, but do not make the user wait on an open-ended count-based run unless they explicitly ask for that. Enforce the budget with both an internal experiment deadline and an external hard process timeout/kill slightly above the budget, and flush metrics/artifacts incrementally so an external kill still leaves useful results.
16. Every shell/tool execution must have an explicit finite timeout by default, not just experiments. Any command, script, or ad hoc analysis that might run longer than a brief read/listing must be launched with a hard timeout/kill path. Do not run open-ended commands.
17. For bounded experiment runs, the hard kill must bound the whole launched process from command start, not just the inner training/evaluation loop. Do not treat a Python-side time check or a tool-level timeout as sufficient by itself when the user expects a strict wall-clock cap.
18. When a command must stop near an allocated budget, enforce it in the launched shell command itself with an OS-level watchdog/kill wrapper, then also keep the inner script deadline. Startup, imports, model loading, and teardown all count toward the user-facing time budget unless the user explicitly says otherwise.
19. Background work must stay in the background. Do not spawn visible terminal windows, consoles, or popups on the user's desktop while running commands or watchdog wrappers. Prefer hidden/background-safe execution patterns only.
