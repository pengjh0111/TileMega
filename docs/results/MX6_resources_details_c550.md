# MX-6 CPU rejection metadata

`MX6_resources_c550.csv` keeps every compiled plan and retains its historical bytes. From continuation1's second resource collection, **never-executed CPU resource rejections** use a compact representation; queried/measured plans keep their full CSV metadata. `resource_scope` has `MX6_resources_details_c550.jsonl.gz:<1-based line>:<SHA256>`.

The companion is append-only concatenated gzip members. Standard `gzip.open(..., 'rt')` reads all members as JSON Lines. Each line is the complete original row, including all omitted redundant fields. Hash its UTF-8 JSON serialized with `sort_keys=True, separators=(',', ':')` to match the CSV reference. Binary/source hashes, numeric resources, compile time, status and variant remain directly in CSV. An empty source-head/bridge/cost/compile-receipt field in a compact row refers to this full row. Empty body instruction counts are used only when the original attribution was pending and every count was null.

Compact `coordinates_json` maps each GEMM class to `[TN, WN, D, P_w, r_next, TileRange]`, where the final bit is1 for TileRange and0 for StreamK. Attention and other program fields remain in the hash-bound complete row. All original pin, compile receipt, MIR, resource report and source snapshot evidence remains under its recorded runs/ path or exact-byte lossless CPU archive. This storage format grants no runtime/resource qualification.

Records with checkpoint `CP4_ATTRIBUTION` and type `body_byte_attribution` explicitly add per-body code bytes. In these metadata supplements, `body_instructions_json` contains `{"instructions": {...}, "code_bytes": {...}}`; `control` and `shared` stay separate. Each byte partition sums to the complete normal device text, and each instruction partition sums to the program count. The referenced attribution artifact verifies exact text equality. These rows add no new GPU execution or measurement and leave earlier rows unchanged.

## Immutable CSV parts

Continuation1's subsequent collections can append `record_type=resource_csv_part` indexes to the main CSV. Each index names a sibling `MX6_resources_c550_cpu_partNNNN.csv.gz` containing at most20000 complete CPU rejection CSV rows, with the same header. The logical resource dataset is the main file plus its indexed parts. Every rejected candidate retains its actual MTreg/stack and full source/binary/receipt references in a CSV row. Queried/measured plans stay directly in the main file. No historical CSV or JSONL bytes are replaced.

The index's `resource_scope` JSON records compressed/uncompressed SHA256, row count and ordered-header SHA256. Parts are sealed before the index is appended and never modified afterward. This avoids exceeding the regular-Git100MiB file limit stated in [GitHub's official documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github) (checked2026-11-01), without discarding resource failures or changing qualification.

Verify every indexed part and count the logical rows:

```bash
python scripts/mx_resource_csv.py docs/results/MX6_resources_c550.csv
```

Export a derived, expanded CSV under ignored `runs/` when a single file is needed:

```bash
python scripts/mx_resource_csv.py docs/results/MX6_resources_c550.csv --out runs/mx6/expanded_resources.csv
```

The reader preserves the older compact-row/JSONL references described above. It rejects damaged payloads, changed counts/headers, nested parts and attempts to partition queried rows. Verified CPU checks: five round-trip/tampering/publication cases pass; the existing33052 main-file rows also expand with identical normalized field values. These storage checks provide no GPU, numerical or synchronization gate evidence.
