# T4.3: numerical correctness statements in related work

Scope: arXiv v1 HTML evaluation sections, inspected 2026-09-09. This is a
source review, not an independent reproduction of either system.

| Source | Evaluation describes | Elementwise reference / tolerance / depth |
|---|---|---|
| [MPK, 2512.22219v1, §6](https://arxiv.org/html/2512.22219v1#S6) | BF16 across systems, HuggingFace model architectures, five models on A100/H100/B200; serving comparisons with SGLang/vLLM | Explicit PyTorch elementwise comparison, numerical atol/rtol, and a numerical acceptance protocol indexed by layer count: **not found in the evaluation text** |
| [Event Tensor, 2604.13327v1, §4](https://arxiv.org/html/2604.13327v1#S4) | B200 experiments; single-layer MoE and full decoding pipelines for Qwen3-30B-A3B / Qwen3-32B | Explicit elementwise golden source, numerical tolerances, and layer-count-specific acceptance protocol: **not found in the evaluation text** |

✅ Text searches additionally found no `tolerance` or `rtol` in either HTML
document; MPK had no `allclose`/`numerical` match and ETC no `correctness`
match. These absences do **not** establish that their implementations lack
correctness tests, nor justify inferring a relaxed numerical criterion.
Performance comparisons against a framework are not automatically numerical
comparisons against that framework's tensor outputs.

The local Mirage path asserted in the task was not found in the current
`docs/FINDINGS.md` or `/root` top-level directories. No local Mirage source
test or tolerance is claimed inspected. The paper links the public
[Mirage repository](https://github.com/mirage-project/mirage), but a current
checkout would need to be pinned before attributing its tests to these papers.

⚠️ This completes only the literature lookup. It does not establish
TileMega's BF16 noise floor, make depth accumulation inevitable, or change
TileMega's existing tolerance. T4.1/T4.2 GPU measurements remain outstanding.
