# Qwen3 B16 paged attention out-of-line diagnostic

This is an isolated compile overlay, not a change to the production source.
The overlay adds `__noinline__` to `PagedAttentionTaskBody::Run` and otherwise
uses the same generated serving plan. The source diff is `overlay.diff`.

`ptxas` reports L1 spill stores/loads per thread of 168/332 bytes for the
baseline and 104/32 bytes for the overlay, but the out-of-line function
increases the L1 stack frame from 208 to 320 bytes. Both mode orders were
timed with the candidate benchmark at Qwen3 B16, past 575. The overlay was
slower in both matched orders (see `timing.tsv`), so this experiment does not
justify changing the production TaskBody. Candidate timing is synthetic and
does not replace complete-request EV-2 measurements.
