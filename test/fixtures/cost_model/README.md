# Host cost-model fixtures

These fixed inputs support the interface-probe tests and flow diagnostics.
They are not measurements of the current device or calibration defaults.

- `registers_{gqa2,mha4}.tsv`: original BF16 oracle register tables from
  baseline `524521488`, copied byte for byte (verified).
- `hop_ns.tsv`: original EX-S1 fitted hop curve from that baseline, copied
  byte for byte (verified).

Fresh hardware calibration uses `tilemega calibrate`, independently of these
fixtures. Historical experiment directories are not runtime dependencies.
