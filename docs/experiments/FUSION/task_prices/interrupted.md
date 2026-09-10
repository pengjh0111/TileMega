# Interrupted CPU verification

The tool connection restarted while the full 4308-group GEMM gate was
running. Read-only process checks found no remaining runner or gate process,
and no completed gate file was written. The two model mixed-price probes
completed, but this directory does **not** provide a completed GEMM gate.
The original RUNNING marker is retained as interruption evidence.
The next run streams gate output to files rather than buffering it in Python.
