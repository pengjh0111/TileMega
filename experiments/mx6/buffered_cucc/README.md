# Verbatim cucc output buffering

This MX-6 C550 build wrapper passes every argument to the unchanged MACA cucc
executable and buffers its combined diagnostic stream in 1 MiB chunks. It retains
the full MIR dump and resource report. Select it explicitly with `CUDACXX` pointing
to this directory's `cucc`; the repository's default compiler remains unchanged.

The wrapper checks the underlying cucc SHA256 before and after each invocation.
For outputs under this campaign's `/root/tilemega/runs/mx6/` directory it also
writes an exclusive `.actual_toolchain.json` sidecar with the real command,
compiler/wrapper hashes, return code, byte count and child CPU times. Native
compile receipts bind the wrapper's own bytes through their compiler input.

Verified continuation1 B11 evidence is in
`runs/mx6/continue1/buffered_cucc_preparation/`: five fresh CPU compilations have
identical device text, kernel resources and MIR instruction counts; a complete
native serving build also passes identity verification and matches its original
device text/resources. These are CPU tool checks, not synchronization or GPU
qualification. One normal-output control took 99.71 s (34.77 user + 64.69 system
CPU seconds); the five buffered runs took 35.46–38.30 s (31.90–35.57 user +
2.06–2.77 system). Different concurrent compiler activity limits a precise
throughput claim. Small diagnostic writes are the inferred source of overhead.
