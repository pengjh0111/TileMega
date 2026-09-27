# Selected Llama PG-1 decode diagnostics

The selected B=1 and B=16 decode manifests both choose paged **L1**. Each
`report/page_chain.tsv` has 1,023 consecutive decode steps, with exact
CG-derived DRAM floors evaluated at that step's past. The uncompressed
per-CTA page trace is retained as `request/page_trace.tsv.gz`.

`TILEMEGA_PAGE_TRACE=1` and `TILEMEGA_TRACE_V2=1` materially slow these
kernels. Use the trace only for page stalls, relative floor position, and
adjacent launch gaps; use `ev2/raw/*tilemega.json` for uninstrumented E2E.

The paged L1 kernel does not stamp individual `TaskRef` Trace V2 slots or
task dependency waits. Its zero-valued slot dump cannot produce a realized
chain; `measured_chain_links`, bubble per link, and the intersection of page
full with dependency waiting are therefore **null**, not zero. The separately
instrumented fixed-geometry PG-1 controls at
`../../single_page_loader/diagnostics/summary.tsv` provide those quantities.
