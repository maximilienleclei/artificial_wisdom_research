# AGENTS.md

1. Keep responses very short by default.
2. Treat repo docs as durable memory. Anything important enough to survive a session must be written down, not just acknowledged in chat.
3. The user wants to operate at a high level: report goals, results, constraints, and next decisions; keep implementation details under the hood unless asked.
4. After any meaningful change, update the handoff docs so a new agent can quickly understand current experiments, decisions, results, constraints, artifacts, and next steps.
5. After any meaningful verified codebase or workflow change, stage and commit it by default unless the user explicitly says not to commit yet.
6. Prioritize the active experiment over backwards compatibility.
7. When optimizing experiments, report concrete throughput or outcome numbers rather than relying on epochs alone.
