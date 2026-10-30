# MX-5 C550 MoE recovery

Prompt SHA256: `75431f98a6286063cc73effe5ece2ed3d462304de4c4192dbd4023efe2bd6f73`.
Starting HEAD: `7fb2ddc1f`. This is an ongoing recovery report, not acceptance.

## torch.compile missing baseline cells

Verified by reading all eight original MX-4 logs and guard records: MoE-L24
B2/B4/B8/B16, rounds 1 and 2, all terminate with code 124, `guard timeout`.
Every log contains the Dynamo recompile-limit warning and specialization on
`self_attn.layer_idx`; none contains an out-of-memory exception. The logs also
record failure of fullgraph=True and retry with fullgraph=False. Recompilation
is an observed contributor; the logs do not isolate the entire timeout cost.

| B | Round 1 peak used MiB | Round 2 peak used MiB | Disposition |
|---|---:|---:|---|
| 2 | 31710.09 | 31693.64 | timeout in both rounds |
| 4 | 32043.75 | 31883.54 | timeout in both rounds |
| 8 | 32627.05 | 32843.65 | timeout in both rounds |
| 16 | 34346.00 | 34345.55 | timeout in both rounds |

These are sampled device totals on 65536 MiB cards, not allocator peaks.
Even the smallest sampled free-memory value is 31189.996 MiB. Record these
cells as timeout, not OOM or zero latency. Original fixed baseline CSVs are
unchanged; no external baseline was rerun. Raw CPU audit:
`runs/mx5/moe_recovery_20261030_v1/BASELINE_TIMEOUT_AUDIT_CPU.json`.

## TileMega B4/B16 resource block

Verified previous full-program original top-four builds reach MTreg 256 with
positive stack frames: B4 256–288 B, B16 320–368 B. Their prescribed prefetch
reductions to D still fail. This is register spilling, not a device-memory OOM.
The original thresholds remain MTreg <=255 at 512 threads and zero stack.
Previous stage-omission diagnostics implicate expert down/combine integration;
they do not establish that every spill instruction belongs to that body.

The user now prioritizes making B4/B16 executable and testing them, followed
by prompt-directed MoE optimization toward about 1.1x speedup. Supplementary
recovery explores the existing measured legal domain in native k-best order;
original failed candidates and their resource histories remain recorded.
An extended diagnostic preview does not grant execution, select defaults,
change resource/numerical thresholds, or replace the final four-candidate
end-to-end comparison. Any supplementary ranks must be identified as a
selection difference from the prompt's original predicted top-four set.
