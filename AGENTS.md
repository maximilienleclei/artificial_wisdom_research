# AGENTS.md

1. Keep responses very short by default.
2. Treat repo docs as durable memory. Anything important enough to survive a session must be written down, not just acknowledged in chat.
3. The user wants to operate at a high level: report goals, results, constraints, and next decisions; keep implementation details under the hood unless asked.
4. The user does not want to read repo files. Communicate all user-facing state, decisions, and summaries through chat; repo docs are for agent continuity.
5. After any meaningful change, update the handoff docs so a new agent can quickly understand current experiments, decisions, results, constraints, artifacts, and next steps.
6. After any meaningful verified codebase or workflow change, stage and commit it by default unless the user explicitly says not to commit yet.
7. Preserve experiments so they can be rerun, but do not burden active work with backwards compatibility.
8. Before implementing a new experiment, decide whether it is cleaner to rewrite the codebase for that experiment and archive the previous codebase.
9. Organize archived experiments as numbered experiment folders named like `XXX_expname`.
10. Treat the repo as an evolving archive of experiment-specific codebases, with the current branch optimized for the active experiment.
11. Expect the user to work across multiple machines. Record machine-specific environment facts, but do not assume one GPU or Python stack is permanent.
12. For experiment archives, put generated plots and plot-adjacent metrics in that experiment's `plot/` folder; experiments without plots should not have a `plot/` folder.
13. When optimizing experiments, report concrete throughput or outcome numbers rather than relying on epochs alone.
