# Multi-page vector routing control

The SM80 page loader now routes each 16 B vector directly to the physical
page returned by CuTe's swizzled layout. The old loader rescanned the whole
logical B tile separately for every physical page. This affects multi-page
GEMM stages only; TMA retains its existing path.

The expanded `paged_gemm_test` passes 49 small-shape cases, including valid
output columns on both pages of a 256×64 B tile. Before any performance
comparison, the rebuilt Llama B=1, 16 KiB serving plan passed 50 independent
processes of the protocol check: 250 reference/candidate 64-step runs, zero
token mismatches. The tested plan SHA256 is
`a28e920583fe9fd592a9f8f6276a478d8ae74a598d36b3a6475ae32485c1312f`.
The summary is `fresh50_summary.json`; raw logs, cases and per-process token
hashes are archived in `fresh50_raw.tar.xz` (SHA256
`1ab6dc67b6b9e2cb4cca5798f3bd175c19fc0cb7bed4d2e85e6b67ebe9787312`).

The fixed-geometry old/new full-request comparison is run by
`measure_page_vector_once.py` after the protocol gate. No performance claim
is made from the earlier invalid logical-interval experiment.

`arch_compile.json` records cross-compilation and SASS checks of this plan
for sm_80/89/90/100/120. All five compiled and the two serving kernels had
zero static FP64 instructions. Only sm_89 was executed on this host.

The same-session, fixed-geometry full-request control took **4.749245 s**
with the original 16 KiB page loader and **4.485077 s** with direct physical
page routing, a 5.56% reduction in E2E time. Both arms passed their three
timed-token consistency checks; the predeclared contamination policy and all
round records are in `e2e_raw.tar.xz` (SHA256
`fb8029b76e6150643212438243823421cb9ca720a5a311395e6bb0ee43c99379`).
The result and policy are in `e2e_results.json` and
`measurement_policy.json`. This remains 1.34× the separate same-geometry
PG-off control, so the page bottleneck is only partly removed.

The isolated lm_head stage on the optimized plan measured 0.723536 ms,
compared with 1.120256 ms for the original 16 KiB page loader and
0.621008 ms for the old collective in the new run. The optimized class raw
logs and commands are in `class_raw.tar.xz` (SHA256
`7c75eeedf6f7aeb474633fffe6bc4aaa562a147da897e4d2ee4de64848421a6d`).

An additional full-request Llama B=1 correctness check on the optimized
binary passed: L1/L2 produced the same 1024 tokens on the same plan instance,
and HF teacher-forced gap was zero at all 1024 positions. The exact outputs,
commands and checks are in `correctness_raw.tar.xz` (SHA256
`571cea0a7378d35b81a4b9c7402e13dbaaa0d2663e1f80ecf966fdd5bebac377`).
