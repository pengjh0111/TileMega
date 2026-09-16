# R4 corrected target freeze

✅ Verified: 24 fresh configuration A processes cover six candidates and four
reference cells. Each target uses its own materialized Plan, task DAG, trace
node durations and `meta.tsv` measured time. `prepare_targets.py freeze`
requires `measured_A >= max(cp_corrected, queue_lb)` before writing any target.
No C implementation is present when these binaries are built and run.

The archived `invalid_targets_pre_resume.tsv` is the exact table committed in
`d549c5a8`. It confused configurations with candidates, used the legacy bound,
and borrowed another candidate's measured time. It is retained as invalid
history, not used to set a gate. Under the user's clarification to repair
known errors and continue independent work, the replacement `../targets.tsv`
is the first valid full freeze. `frozen.json` records its digest and all dump
input digests. This replacement must precede resumed C implementation commits;
subsequent measurements may not modify it.

The original H4 violation (`66ae000e` committed C1 before B measurements) is
not erased. That implementation was reverted by `8299ccaa`. The resumed
sequence is completed A (`23c5d618`), this corrected freeze, completed B, then
new C implementation. The final report distinguishes this repaired sequence
from the original noncompliant history.

Measured A here means the matching trace's timing, as in the R3 target
definition. Instrumented node weights and uninstrumented wall times are not
mixed. These are candidate-specific reporting ceilings; the research gate
remains the registered default-placement wait + notify <= barrier target.
