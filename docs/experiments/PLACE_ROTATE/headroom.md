# EX-D2 offline headroom

`eft_makespan` is an earliest-finish-time list schedule over the
exact task DAG: measured task durations, measured hop p50 on every
cross-worker edge, free placement on the resident workers.  It is an
optimistic bound on what rescheduling alone could buy, and it says
nothing about whether such a schedule is legal under today's queue.

## reference

| cell | slots | measured l2 | work_lb | queue_lb | cp_lb_nosync | cp_lb_sync | EFT | EFT/measured | workers |
|---|---|---|---|---|---|---|---|---|---|
| gqa2_s4 | 200 | 0.4516 | 0.0070 | 0.3666 | 0.1434 | 0.1536 | 0.1526 | 0.34 | 118 |
| gqa2_s128 | 4416 | 0.6267 | 0.1085 | 0.5304 | 0.1966 | 0.2068 | 0.2673 | 0.43 | 256 |
| mha4_s4 | 512 | 0.9020 | 0.0155 | 0.7485 | 0.1444 | 0.1546 | 0.1546 | 0.17 | 256 |
| mha4_s128 | 11920 | 1.2780 | 0.2271 | 1.0752 | 0.1987 | 0.2089 | 0.3533 | 0.28 | 256 |

## real width l4 h4096

| cell | slots | measured l2 | work_lb | queue_lb | cp_lb_nosync | cp_lb_sync | EFT | EFT/measured | workers |
|---|---|---|---|---|---|---|---|---|---|
| real_s128 | 47936 | 8.5492 | 2.7392 | 7.9534 | 1.3158 | 1.3261 | 3.3065 | 0.39 | 256 |
| real_s4 | 2800 | 6.2312 | 0.9919 | 6.0436 | 1.2145 | 1.2247 | 1.5985 | 0.26 | 256 |
