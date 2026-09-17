# Retained intermediate multi-point certificates

These two direct symbolic-interval attempts used the final gqa2/mha4 seq4
winner geometry at grid 512. They were terminated while ISL was eliminating
multi-point parameter constraints, after partial proof output. They are not
complete certificates and are not used by the verifier.

`../bounded_certificates/` replaces them with disjoint exhaustive integer
certificate shards covering every seq in [1,128]. The template expressions and
native materializers are unchanged. Singleton certificates simplify ISL's
parameter elimination; they do not add GPU binary variants or infer legality
from the five additional native-table samples.
