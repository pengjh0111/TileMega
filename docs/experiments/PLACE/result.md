# B2 production placement status

The offline producer-wide fence audit is complete in
`AFFINE_PROBE/fence_producers/result.md`. It does not authorize lowering:
`overresident_proven=0` remains true for sampled 512/16 cases. Production
integration must enforce `grid <= resident_limit` as an L-sched attribute and
have lowering reject a violating concrete launch. No placement is silently
written into the generator until that constraint and the A2 projection are
wired together.

The sm_120 script is intentionally script-only and writes `NOT_RUN` until a
5090 host is available.
