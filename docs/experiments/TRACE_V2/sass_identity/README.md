# H2 evidence: default-build SASS identity

`bash docs/experiments/TRACE_V2/sass_identity.sh` compiles both reference
models twice with identical flags and no `TILEMEGA_TRACE_V2` definition, once
against the baseline commit's headers and once against this tree's, and diffs
`cuobjdump --dump-sass` of the two.

Result: `gqa2.diff` and `mha4.diff` are both empty, and the script exits 0.
The two models' dumps differ from each other (`sha256.tsv`), so these are two
independent checks rather than one comparison run twice.

The host executables do differ between the arms, as expected: host code changed
and `-lineinfo` embeds line tables. H2 is about the device SASS, which did not.

The 6.8 MB dumps themselves are not committed; `sha256.tsv` records their
digests and the script regenerates them.
