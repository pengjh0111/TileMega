"""Teacher-forced near-tie and free-greedy audit for frozen serving tokens."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _quantile(values, q):
    import torch
    return float(torch.quantile(values, q).item())


def _free_greedy(model, prompt, steps):
    import torch
    generated = []
    with torch.inference_mode():
        result = model(prompt, use_cache=True)
        cache = result.past_key_values
        next_token = result.logits[:, -1, :].float().argmax(-1, keepdim=True)
        for step in range(steps):
            generated.append(int(next_token.item()))
            if step + 1 < steps:
                result = model(next_token, past_key_values=cache, use_cache=True)
                cache = result.past_key_values
                next_token = result.logits[:, -1, :].float().argmax(-1, keepdim=True)
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--prompt-ids", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True,
                        help="JSON [B][N] token array")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--vllm-metrics", type=Path,
                        help="checked vLLM output for the same model and batch")
    parser.add_argument("--skip-free-greedy", action="store_true")
    args = parser.parse_args()
    import torch
    from transformers import AutoModelForCausalLM

    ids = json.loads(args.prompt_ids.read_text())
    generated = json.loads(args.generated.read_text())
    if not generated or any(len(row) != 1024 for row in generated):
        raise ValueError("expected [B][1024] generated tokens")
    if len(generated) > len(ids) or any(len(row) != 64 for row in ids[:len(generated)]):
        raise ValueError("frozen prompt shape must be [16][64]")
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", trust_remote_code=False,
    ).cuda().eval()
    all_gaps, all_nll, buckets, first_divergence = [], [], [[] for _ in range(4)], []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for b, output in enumerate(generated):
        prompt = torch.tensor([ids[b]], device="cuda", dtype=torch.long)
        whole = torch.tensor([ids[b] + output], device="cuda", dtype=torch.long)
        with torch.inference_mode():
            logits = model(whole[:, :1087], use_cache=False).logits[0, 63:1087].float()
            chosen = logits.gather(1, whole[0, 64:1088, None]).squeeze(1)
            gaps = (logits.max(1).values - chosen).cpu()
            nll = (torch.logsumexp(logits, 1) - chosen).cpu()
        all_gaps.append(gaps)
        all_nll.append(nll)
        for bucket in range(4):
            buckets[bucket].append(gaps[bucket * 256:(bucket + 1) * 256])
        if not args.skip_free_greedy:
            free_path = args.out.parent / f"hf_free_greedy_b{b}.json"
            if free_path.exists():
                free = json.loads(free_path.read_text())
            else:
                free = _free_greedy(model, prompt, 1024)
                free_path.write_text(json.dumps(free, separators=(",", ":")) + "\n")
            first_divergence.append(next((i for i, (a, c) in enumerate(zip(free, output)) if a != c), None))
        del logits, whole
    gaps = torch.cat(all_gaps)
    nll = torch.cat(all_nll)
    ratio = float((gaps <= 0.5).float().mean())
    maximum = float(gaps.max())
    required_ratio, allowed_maximum = 0.99, 3.0
    if args.vllm_metrics:
        baseline = json.loads(args.vllm_metrics.read_text())
        if baseline["gap_le_0_5_ratio"] < required_ratio or baseline["max_gap"] > allowed_maximum:
            required_ratio = baseline["gap_le_0_5_ratio"] - 0.005
            allowed_maximum = baseline["max_gap"] * 1.5
    report = {
        "model": str(args.model), "batch": len(generated), "positions": int(gaps.numel()),
        "gap_le_0_5_ratio": ratio, "max_gap": maximum,
        "required_ratio": required_ratio, "allowed_maximum": allowed_maximum,
        "pass": ratio >= required_ratio and maximum <= allowed_maximum,
        "gap_zero_ratio": float((gaps == 0).float().mean()),
        "gap_p99": _quantile(gaps, 0.99), "gap_p999": _quantile(gaps, 0.999),
        "mean_nll": float(nll.mean()),
        "bucket_stats": [{"range": [i * 256, (i + 1) * 256 - 1],
                          "gap_le_0_5_ratio": float((torch.cat(rows) <= 0.5).float().mean()),
                          "gap_p99": _quantile(torch.cat(rows), 0.99),
                          "max_gap": float(torch.cat(rows).max())}
                         for i, rows in enumerate(buckets)],
        "first_hf_greedy_divergence": first_divergence,
    }
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
