# User-directed R10 partial closure

`../summary.md` is the authoritative report. No further R10 tests are queued.
`status.json` distinguishes b22 selected plans from current-source seed checks
and the cancelled current plan search. `stopped_processes.json` records exact
process trees and termination. `verify.txt` is complete source-verifier output.

Large raw files are losslessly compressed. `compressed_raw.json` maps their
original paths and SHA256; uncompressed copies also remain outside Git under
`/root/r10_work/r10_closure_raw/`. To reconstruct a raw record without running tests:

```sh
gzip -dc <path.tsv.gz> > /tmp/record.tsv
sha256sum /tmp/record.tsv
```

Do not execute the archived `drivers/` automatically: the user cancelled all
remaining work. They record how completed and interrupted jobs were launched.
HF seed-check empty free-greedy lists mean skipped, not no divergence.
