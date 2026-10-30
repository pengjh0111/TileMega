# Vector LM-head argmax

`as2_argmax` calls the same `LlmVectorBody::{ArgmaxRows,ArgmaxMerge}` as the
generated persistent runtime. It is a component probe, not a model forward.
The checked-in candidate remains off until numerical/model/request gates pass.

Build only, after sourcing `scripts/as2_cpu_env.sh` and configuring the existing
`tools/ascend` project with an actual generated source:

```bash
nice -n 10 cmake --build "$AS2_BUILD" --target as2_argmax --parallel 1
```

The twelve bounded cases cover rows1/16, columns17/128/255, a151936-element
input stride, guard holes, negative values, signed zeros, BF16-collapsed ties,
1002/1187 shuffled partial IDs, and nonfinite/out-of-domain rejection. Both
AIV lanes execute twice. The independent Host oracle rounds inputs to BF16
before finding the first maximum; no expected output is supplied to the kernel.
The probe has no polling loop or resource-expansion experiment.

**Not run in the 2026-10-30 TaskBody checkpoint.** The inherited physical0
post-timeout exit is unknown; physical1 also reports Alarm. Do not run while
those safety holds remain. No device reset or other-process action is permitted.
When device work is admissible again, use the existing lock-free wrapper,
one verified physical card per process and a new controlled output directory:

```bash
python -m tilemega.bench.as2_run --help
# Use the installed wrapper's --physical-device/--out/--timeout-s interface:
# ... --physical-device <verified-id> --out runs/as2/jobs/<new-id> \
#     --timeout-s 60 -- "$AS2_BUILD/as2_argmax" \
#     /home/d50064638/junhuip/TileMega/runs/as2/probes/<new-result>
```

Retain every job receipt and failure. Five fresh successful processes are still
required before claiming the reused publication/consumption route is stable.
Then test the generated single-layer/full-model path, teacher512, C2/C3 and
full64/1024 paired TileMega control requests. Compilation and this standalone
probe cannot admit new prices, defaults or a TPOT claim.
