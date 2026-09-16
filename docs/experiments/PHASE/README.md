# R5 task cost diagnosis

Baseline: `a001b0ac53197551fe1e5e115528a9ea40d24783`, branch `tilemega`.
Prompt: `/root/Prompt/TileMega_R5_prompt.md`; digest in `audit/provenance.json`.

The initial audit reuses digest-checked R4 configuration-A trace executables,
starting eight new processes. No historical timing is a fresh performance claim.
`audit.py` reproduces node counts and nearest-rank duration distributions from
raw slot stamps and the matching unsimplified generated task DAG.

Verified: seq=4 averages remain about 12 us, but medians are about 4 us and
maxima about 43 us. Seq=128 averages are about 17 us. The actual measured
GEMM tile is 128x128x16, three stages, split=1. F-128's 16x64x16s2k16 is
an L1 solver example, not the configuration underlying R4's floor table.
The rotate measured/floor interval remains about 1.18–1.38. These corrected
premises preserve R5's task-cost diagnosis; no global stop is triggered.

Before phase measurements, freeze the following interpretations:

* FORK5 uses the four reference **rotate** cells, since that placement defines
  R5's research comparator and its motivating floor table. Legacy and both
  real-width sequences are separately reported; they do not change that median.
* S3-b compares to freshly measured rotate with R3 B protocol (E3-0..3 on,
  R4 C switches off, W=1). The quoted 0.7129 ms belongs to legacy placement,
  not rotate; it is never substituted for the same-session control.
* Thresholds and coverage are exactly R5 §§4.5/5.3/6.3/7. An unsupported tile
  is reported as rejected by backend legality, not silently omitted.
* Phase stamps may not add barriers, atomics or polling-loop stores. Existing
  timer quantization and inseparable SIMT arithmetic/loading are reported;
  a collapsed phase is not evidence that those memory operations are free.
* S3-c's empirical ranking universe must be published before its measurements;
  unmeasured candidates cannot be assigned a measured rank.

`SIMULATOR/` new evidence is authorized by R5 §5 despite omission from H1's
list of new directories. Existing experiment files remain untouched, except
for the explicitly authorized TRACE_V2/analyze.py.
