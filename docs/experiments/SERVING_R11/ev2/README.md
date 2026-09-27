# EV-2 measurement admission, predeclared before the matrix

The user reduced batch endpoints to B=1 and B=16. The four cells are
Llama/Qwen3 × those two batches; TileMega and vLLM are alternate arms.
The original ten-cell G-9 remains unmeasured under this scope change.

At 2026-09-27, with `nvidia-smi --query-compute-apps` empty, five one-second
idle-power observations were 21.81, 21.81, 21.82, 21.82 and 21.78 W. Their
median, **21.81 W**, and the prompt's fixed +30 W margin set the admission
threshold at **51.81 W**. `measurement_policy.json` freezes this value before
EV-2. A round is accepted only if the sole compute PID is the benchmark
process and both power checks meet the threshold; at most three attempts are
allowed. `python -m tilemega` can use the same file for every arm by setting
`test.policy_file` in its config. The `rounds` array is intentionally empty
until actual EV-2 measurements; K-15 must remain FAIL in the meantime.
