# Repository instructions

Before starting any task in this repository, read `CLAUDE.md` in full and
follow its development, comment, evidence, and commit conventions.

Split changes into focused commits by implementation category or dependency
step. Do not combine unrelated implementations and experiments into one
commit. Each commit message must be a single line under 72 characters in
the format `<area>: <imperative summary>`, with no body or attribution trailers.

## Test queue monitoring

Delegate read-only queue monitoring to one low-cost subagent when available.
The user authorizes this delegation for TileMega test queues. Prefer
`gpt-6-luna` with `low` reasoning effort when those settings are exposed;
otherwise use an available inexpensive model and disclose the fallback.
Start with a small explicit task and queue paths instead of copying the full
development conversation. The primary agent owns fixes and gate acceptance.

Use `python -m tilemega.bench.monitor watch` to wait for state-file events
and check process health. The subagent consumes `monitor next` alerts and
reports new failures, timeouts, missing processes, a six-hour timing pause,
or queue completion. Keep its event offset to avoid repeating old failures.
Do not repeatedly tail logs or ask the primary agent to poll unchanged state.
Read only a short relevant log excerpt when an alert needs diagnosis.

The monitor must not edit frozen inputs, change guard policy, launch tests,
kill jobs, or retry them itself. Exit 75/cooldown and busy compilers are not
failures. All terminal steps do not imply all gates passed. Raw monitoring
events and cursors belong under ignored `runs/`. Continue independent work
while monitoring; advance dependent stages only after accepting their gates.

Subagent messages notify the primary agent during an active session. Do not
promise that a subagent can wake or resume an ended session without an
explicitly supported automation mechanism. Persist events for later recovery.
