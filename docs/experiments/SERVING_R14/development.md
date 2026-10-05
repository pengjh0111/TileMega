# Independent CPU development checkpoint

The main `/root/TileMega` source/tool checkout remains frozen at `9e9f02c4b` for Phase 0.
Development uses `/root/r14_work/development`, branch `r14-development`.
These changes are not yet merged into the tested main checkout.

| Item | Code | Validation / outstanding work |
|---|---|---|
| RW-3 | `df96abb7f`; ServingMmaPipeline.h, PagedGemmTaskBody.h, ServingTiledMainloop.h; compile/CLI knob | verified: sm_80/89/90/100/120 compile; integrated paged/tiled consumers compile. Numerical GPU tests queued; model C-2, resources and timing pending. Default remains off. |
| AT-1 | `821c64cf6`; ServingAttentionLegality.h; search Ec32 and page-frontier checks | verified: 144 host legality checks; affected C++ objects compile. Ec measurement awaits SL-6; distribution correction and model C-1 remain outstanding. |

Evidence: `raw/development_checks.tar.xz`; binary paths/SHA256: `rw3_tests.json`.
The archive retains the first failed RW-3 compile (test alias conflicted with cute::E), followed by successful corrected compiles.
Host distribution enumeration is not a GPU performance measurement: for the grid=128 fixture, B1/Ec64/past1000 gives max 2 valid tasks per worker versus ideal 1; see AT1_host/result.log.
The grid=170 fixture is a host-only portability audit, not sm_120 execution.
Do not change g-major logical event numbering when correcting execution distribution.

`queue/queue_development.json` is copied into the main scheduler queue directory.
It checks test binary SHA256 before running, waits until the Phase-0 diagnostic chain ends, and uses the same GPU guard/lock. No second scheduler is started.
RW-3 test kernels cover position-coded operands and compare unchanged K-order results; they do not replace full model tests.

AT-2, AT-3a, SK-1, GV-1, RA-1, EP-1 and SL-6 are not implemented yet.
TR-4 still needs diagnostic data before its low-overhead implementation is chosen.
Conditional Phase C and default selection still require the registered evidence.
