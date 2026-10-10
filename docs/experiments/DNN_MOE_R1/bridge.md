# CI-1 bridge facts

verified: structured `args` is a typed list; `kwargs` is a map of typed
values. Tuple arguments use the list encoding. `scalar_args` and all other
legacy fields retain their original representation. C++ records preserve
the tags and reject malformed values instead of silently dropping them.

inferred design choices for unspecified details: symbolic literal arguments
use a `symbol` tag; nonfinite floats use `+inf`, `-inf` or `nan` strings within
the float tag. Tensor payloads up to 4096 bytes use base64. Larger integer
and boolean constants retain their equality/truth summaries. No output
above 2²⁰ elements is evaluated; empty tensors remain eligible.

Shape-only records retain the FX fragment and its exported symbols. The
bridge's `--bind-shape SYMBOL=INTEGER` resolves them for a plan and rejects
unknown or out-of-range bindings. `--pytree-module` imports original upstream
output registrations before archive loading; BERT uses
`transformers.modeling_outputs`. Forward build integration consumes this
binding interface during CI-3/DN-11.

verified: evaluation uses CPU FX Interpreter operations; lifted literal
tensors are copied before evaluation. USER_INPUT, PARAMETER and BUFFER
values are excluded. Shape queries cut only value dependence and retain
symbol dependence. A runtime key-padding mask is therefore not a constant.

Evidence: `results/CI1_dnn_bridge.json` covers all twelve before/Core DNN
archives at B=8 without evaluation errors. `results/CI1_llm_bridge.json`
covers all four legacy serving archives with exact legacy JSON equality.
Python parameter tests cover all eleven listed spellings, nested values,
payloads, binding bounds and the element limit. The native `frontend_import`
test checks parsing, malformed records and byte-identical CG output after
adding bridge fields. No DNN runtime or complete G-REG claim follows from
these checks.
