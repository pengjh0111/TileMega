# DM-1 calibration and routing

verified: `fit_dm.py` passes five CPU checks for nonnegative fitting,
independent held-out cases, rank reporting and the numerical/work/identity
receipt contract (`results/CI6_fit_host.json`). No actual body fit is available.
Native calibration samples and fits accompany the corresponding DN/MO bodies.

verified: routing foundations pass five CPU checks
(`results/CI6_routing_host.json`). Independent window enumeration agrees with
the distributions, including idle experts. Invalid/duplicate expert indices
are rejected. Original per-expert safetensors distributed across two shards
load into unchanged tiny HF decoder layers with bitwise equal parameters and
outputs. That CPU test uses eager experts; production collection explicitly
uses `grouped_mm`. No CUDA or real-weight routing claim follows from it.

`python/tilemega/moe/routing_profile.py` generates a queue with CPU preparation,
one locked embedding process, one locked process per decoder layer and a CPU
finalization step. An HF layer is instantiated on meta, allocated on the
device and populated one original checkpoint tensor at a time. Gate/up and
down parameters retain HF's stacked layouts. The complete HF attention,
normalization and expert forward runs before the next layer's state is saved.
Only one layer's weights reside on the device in each process.

inferred sampling choice: retain each original R10 Qwen3 ID sequence, then
append consecutive, non-repeated Qwen3-tokenized WikiText-103 validation
tokens until it reaches 4096 tokens. Each T uses disjoint contiguous windows
within this causal context. Samples share a context and are not statistically
independent. The manifest records this choice; it does not assert equivalence
to every serving traffic distribution. Histograms include inactive experts,
with separate counts for each expert and the expected distinct expert count.

The collector saves pre-post-attention-normalization hidden states for layers
0/24/47, together with HF router logits, top-k indices and BF16 weights. These
are the inputs to the local MoE region, before RMSNorm. Saved tensors retain
sequence/token positions so all T coordinates can use identical source values.
Every layer records original tensor hashes, input/output hashes, installed HF
implementation hashes and peak allocated/reserved device memory. Finalization
checks the state chain and rejects changing implementations, configurations,
sampling manifests or saved routing/region artifacts.

Generate and submit the real-weight queue after checkpoint download completes:

```sh
env PYTHONPATH=python /root/dm1_work/venv-gpu/bin/python \
  -m tilemega.moe.routing_profile --stage queue \
  --checkpoint /root/models/qwen3_30b_a3b \
  --work runs/dm1-routing-profile/states \
  --queue runs/dm1-routing-profile/queue \
  --prompts docs/experiments/SERVING_R10/prompts/qwen3_ids.json \
  --wiki /root/datasets/wikitext103/wikitext-103-raw-v1/validation-00000-of-00001.parquet \
  --output runs/dm1-routing-profile/profile.json
/root/dm1_work/venv-gpu/bin/python docs/experiments/DNN_MOE_R1/run_queue.py \
  --queue-dir runs/dm1-routing-profile/queue \
  --out runs/dm1-routing-profile/events \
  --policy docs/experiments/DNN_MOE_R1/guard_policy.json
```

verified: the unchanged two-layer seeded HF model and the streamed CUDA
`grouped_mm` collector produce bitwise equal region inputs, router logits,
weights, indices and layer outputs. All thirteen T coordinates are present
(`results/CI6_routing_cuda.json`). This validates collection on sm_89, not the
real Qwen3 checkpoint, TileMega execution or a synchronization path.

verified: real-model input preparation retains all sixteen R10 prompt prefixes
and produces sixteen 4096-token contexts using the pinned Qwen3 tokenizer and
WikiText-103 validation data (`results/CI6_real_routing_inputs.json`). The
48-layer/E=128/K=8 configuration, tokenizer, dataset, token-file and sampling
hashes are recorded. Real HF layer forward awaits complete shard validation.

verified: `MoeRoutingProfile` validates exact model/layer/T coordinates,
integer histogram conservation, idle experts, distinct-expert means and
profile identity format. Host tests cover slot/group capacities, expected
grouped blocks, unique-expert weight bytes, slot reads and gathered-A bytes
(`results/CI7_classes_routing_host.json`). Unique weight bytes form the DRAM
lower bound for both policies; repeated slot reads remain a separate cache/work
quantity. The parser does not authenticate the advertised profile ID; the CLI
consumer must verify the canonical profile hash and source file hash.
verified: `profile_identity.py` supplies that consumer gate, passing 8/8 CPU
checks (`results/CI6_profile_identity_host.json`). It rejects altered content,
duplicate fields, model/token-domain mismatches, broken hidden-state identity
chains and non-conserving histograms. Its content identity detects changes;
checkpoint provenance remains the collector's embedded per-tensor evidence.
CLI/solver integration and real-profile validation remain pending.

Pending: real routing observations and the TaskWork/DRAM-floor/solver consumers
of the profile. No real-model routing estimate or timing value is reported.

verified: joint grouped-block histograms pass independent Python alignment
enumeration (6/6), identity checks (9/9), and six native host tests with policy
checks (`results/CI6_joint_group_host.json`). The distribution yields each
static virtual prefix task's probability of activity; its mean must equal
the sum of per-expert padded block counts. Real-profile and solver consumption
remain pending.
