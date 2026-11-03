# DMA-consumer visibility experiment

`verified-run`: physical1/55:00.0, G8 MIX(1:2), CANN9.1.0/BiSheng15.0.5,
pinned CATLASS2b6c6a6d5bd26559578cf7e3b2fb64af51f64e52. New target
`as2_dma_message_passing` compiles the existing message-passing source with
`TILEMEGA_DMA_CONSUMER_ONLY=1`; the conservative target remains unchanged.

Five fresh processes pass all12 phases/eight epochs/phase. Directions are
AIC→AIC, AIC→AIV, AIV→AIC and AIV→AIV, including both AIV lanes, cross-group
fanout, payload reuse,32/1056/32768 FP32 elements (128KiB largest), row tails,
guards and empty phase roles. Consumer Scalar prereads verify and cache the
old payload before arming the producer. After ready, the consumer uses DMA
without an entire-DCache acquire; it retains DSB. New payload **Scalar** reads
are excluded from this admission. DMA output is independently checked before
reverse acknowledgement. AIC uses MTE2/L1/identity MMAD/FIX; AIV uses MTE2/MTE3.

There are zero preread/DMA/guard/mapping/deadline violations. One additional
fresh process injects actor0 abort; all24 actors acknowledge/exit without a
deadline. This is a one-case abort check, not a five-process abort claim.
Pre-read cache maintenance and post-DMA result acquisition for Scalar
verification remain; this does not prove a cache-free Scalar consumer.

The new mode rejects G14 before ACL initialization. It uses the original
single-writer control/abort cells, ready/armed/ack ownership and finite2e9
counter-tick wait bounds, with Host timeout600s and one own NPU job at a time.
No locks, device reset or other-user process operation are used. The model
runtime still uses conservative cache maintenance; no policy/default changes.

Image: `tilemega-as2/build/taskbody_dma_message_20261103_v1/as2_dma_message_passing`.
SHA256 `895af0170013c74fbee6ecb0b1ffd666eb9290996d319558eceb7bb499cc6455`.
Source, protocol, six receipts, raw buffers and per-process checks:
`runs/as2/probes/dma_message_20261103_v1/` and corresponding
`runs/as2/jobs/dma_message_20261103_v1_p{0..5}/`.
CLI: `as2_dma_message_passing 8 new_output [abort-worker]`, through `as2_run`.
