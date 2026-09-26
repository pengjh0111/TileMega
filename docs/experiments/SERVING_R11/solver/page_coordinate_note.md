# CPU-only page-coordinate smoke

On the anchored Llama decode graph at B=1, past range 64:1086, with the five-shape restricted domain in `page_smoke_domain.json`, one coordinate-descent pass priced 86 configurations. The page coordinate evaluated both 8 KiB and 16 KiB, selected 16 KiB, and ranked it at 3.113389 ms versus 3.115474 ms for the same geometry at 8 KiB. This verifies that `page_bytes` enters the search key and Level 1 score; it is a model prediction, not an execution measurement. The flow-only command does not materialize or compile a plan. See `page_coordinate_ranked.tsv` and `page_coordinate_timing.tsv`.

The single-geometry probe in `page_size_smoke.tsv` evaluated the R10 winner geometry: 8 KiB scored 3.321905 ms and 16 KiB scored 3.319340 ms.
