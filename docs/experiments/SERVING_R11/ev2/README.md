# EV-2 measurement admission, predeclared before the matrix

The user reduced final acceptance to Llama at B=1 and B=16; TileMega and
vLLM are alternate arms in each of these two cells. The original ten-cell
G-9 remains unmeasured under this scope change.

At 2026-09-27, with `nvidia-smi --query-compute-apps` empty, five one-second
idle-power observations were 21.81, 21.81, 21.82, 21.82 and 21.78 W. Their
median, **21.81 W**, and the prompt's fixed +30 W margin set the admission
threshold at **51.81 W**. `measurement_policy.json` freezes this value before
EV-2. A round is accepted only if the sole compute PID is the benchmark
process and both power checks meet the threshold; at most three attempts are
allowed. `python -m tilemega` used the same file for every arm through
`test.policy_file`. The completed run recorded three accepted timing rounds
per arm, with no contamination rejections; K-15 now passes. The run stopped
at the Llama B=16 C-2 mismatch. HF teacher forcing was then run independently
on the saved B=16 outputs: TileMega failed C-1 (max gap 30.55), while vLLM
passed (max gap 0.25). `partial_report.json` contains the measured times,
`partial_status.json` records the stop, and raw guard, mode-check, and HF
files are under `raw/`. The B=16 speed is provisional.
