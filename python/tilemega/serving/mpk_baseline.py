"""Independent MPK (Mirage Persistent Kernel) baseline over frozen token IDs.

Imports no TileMega code. The task graphs replicate MPK's own demos at
/root/mirage-mpk (demo/qwen3/demo_hopper.py, demo/qwen3/demo_30B_A3B_hopper.py,
demo/llama3/demo.py); MPK itself is used unchanged.

One mpk() call serves whole requests (prefill and decode in one persistent
kernel) until every row reaches max_seq_length. That length is a compile-time
constant, so T_N runs a kernel compiled for prompt_length + N positions.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import json

from baseline_common import arguments, guarded_main, measure, require_guard

MIRAGE_HOME = Path(os.environ.get("MIRAGE_HOME", "/root/mirage-mpk"))
DEMOS = {"Qwen3ForCausalLM": "demo/qwen3/demo_hopper.py",
         "Qwen3MoeForCausalLM": "demo/qwen3/demo_30B_A3B_hopper.py",
         "LlamaForCausalLM": "demo/llama3/demo.py"}


class Unsupported(RuntimeError):
    pass


def extend(parser):
    parser.add_argument("--page-size", type=int, default=4096)
    parser.add_argument("--max-num-pages", type=int, default=16)
    parser.add_argument("--split-kv-cache", action="store_true",
                        help="demo_hopper.py --split-kv-cache (Qwen3 dense only)")


def architecture(model):
    names = json.loads((model / "config.json").read_text()).get("architectures") or ["?"]
    if names[0] in DEMOS:
        return names[0]
    # MPK's registry is its own list of supported models; its error is the
    # verbatim refusal. No MPK demo exists for these architectures either.
    import mirage.mpk.models  # noqa: F401  (registers builders)
    from mirage.mpk.model_registry import get_builder
    try:
        get_builder(names[0])
    except ValueError as error:
        raise Unsupported(f"{type(error).__name__}: {error} | no MPK demo entry point "
                          f"for {names[0]} (demos: {sorted(DEMOS)})") from error
    raise Unsupported(f"{names[0]} has an MPK registry builder but no demo this wrapper replicates")


def grid_for_linear_layer(size, with_residual=False):
    if with_residual and size % 64 == 0:
        return size // 64
    if size >= 100_000:
        return size // 256
    for grid in (128, 96, 64):
        if size % grid == 0:
            return grid


def grid_for_rmsnorm_layer(size):
    for grid in (128, 96, 64):
        if size % grid == 0:
            return grid


class Engine:
    def __init__(self, args, arch):
        import torch
        self.torch, self.args, self.arch = torch, args, arch
        torch.set_default_dtype(torch.bfloat16)
        torch.cuda.set_device(0)
        import mirage
        import mirage.mpk.speculative
        # The demos call mi.speculative.spec_decode_class, which mirage 63af900
        # does not export; alias the existing submodule in this process only.
        if not hasattr(mirage, "speculative"):
            mirage.speculative = mirage.mpk.speculative
        demo = MIRAGE_HOME / DEMOS[arch]
        sys.path.insert(0, str(demo.parent))
        path = str(args.model)
        with torch.device("cuda"):
            if arch == "Qwen3ForCausalLM":
                from models.modeling_qwen3 import Qwen3ForCausalLM
                self.model = Qwen3ForCausalLM.from_pretrained(
                    path, world_size=1, max_num_pages=args.max_num_pages,
                    page_size=args.page_size).to("cuda")
            elif arch == "LlamaForCausalLM":
                from models.modeling_llama3 import Llama3ForCausalLM
                import demo as llama_demo
                self.llama_demo = llama_demo
                self.model = Llama3ForCausalLM.from_pretrained(
                    path, 1, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
            else:
                from transformers import Qwen3MoeForCausalLM
                self.model = Qwen3MoeForCausalLM.from_pretrained(path).to("cuda")
        self.model.eval()
        config = self.model.config
        positions = torch.arange(32768).unsqueeze(0).to("cuda")
        if arch == "Qwen3MoeForCausalLM":
            dummy = torch.empty(1, dtype=torch.bfloat16, device="cuda")
            self.position_embeddings = self.model.model.rotary_emb(dummy, positions)
            shape = (config.num_hidden_layers, args.max_num_pages, args.page_size,
                     config.num_key_value_heads, config.head_dim)
            self.kv_cache = (torch.empty(shape, dtype=torch.bfloat16, device="cuda"),
                             torch.empty(shape, dtype=torch.bfloat16, device="cuda"))
            # demo_30B_A3B_hopper.py stacks expert weights once and drops the
            # per-expert modules; the stacked tensors serve every kernel here.
            self.gate_up, self.down = [], []
            for layer in self.model.model.layers:
                experts = layer.mlp.experts
                self.gate_up.append(torch.stack([torch.concat(
                    [e.gate_proj.weight.detach().clone(), e.up_proj.weight.detach().clone()], dim=0)
                    for e in experts], dim=0))
                self.down.append(torch.stack([e.down_proj.weight.detach().clone() for e in experts], dim=0))
                torch.cuda.synchronize()
                del layer.mlp.experts
        elif arch == "Qwen3ForCausalLM":
            self.position_embeddings = self.model.model.rotary_emb(positions)
            self.kv_cache = self.model.model.kv_cache
        else:
            self.position_embeddings = self.model.model.rotary_emb(positions)
        if arch != "LlamaForCausalLM":
            hidden = config.hidden_size
            self.lm_head = torch.cat((self.model.lm_head.weight, torch.full(
                (153600 - config.vocab_size, hidden), 0, device="cuda")), 0)
        self.mpk = None
        self.compiles = {}

    def prepare(self, batch, count):
        torch, args = self.torch, self.args
        if self.mpk is not None:
            self.mpk.finalize()
            self.mpk = None
            torch.cuda.synchronize()
        prompt = self.prompt_length
        length = prompt + count
        meta = dict(
            step=torch.zeros(batch, dtype=torch.int32, device="cuda"),
            tokens=torch.zeros((batch, length), dtype=torch.long, device="cuda"),
            input_tokens=torch.zeros((batch, 1), dtype=torch.long, device="cuda"),
            output_tokens=torch.zeros((batch, 1), dtype=torch.long, device="cuda"),
            num_new_tokens=torch.ones(batch, dtype=torch.int32, device="cuda"),
            prompt_lengths=torch.full((batch,), prompt, dtype=torch.int32, device="cuda"))
        kernel_dir = args.out / f"B{batch}" / f"kernel_N{count}"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        cwd = os.getcwd()
        os.chdir(kernel_dir)  # the demo graph code writes task_graph_0.json and kernel_0.cu to cwd
        try:
            shape = argparse.Namespace(
                max_num_batched_requests=batch, max_num_batched_tokens=batch,
                max_num_pages=args.max_num_pages, page_size=args.page_size,
                max_seq_length=length, output_dir=str(kernel_dir), spec_decode=None,
                ngram_size=3, spec_length=3, profiling=False, trace_name="",
                use_cutlass_kernel=True)
            if self.arch == "LlamaForCausalLM":
                data = dict(meta, position_embeddings=self.position_embeddings)
                self.mpk = self.llama_demo.build_mirage_graph(self.model, shape, 1, 0, data, -1)
            else:
                self.mpk = self.build_qwen3(shape, meta)
        finally:
            os.chdir(cwd)
        self.compiles[f"B{batch}_N{count}"] = dict(
            seconds=time.perf_counter() - start, max_seq_length=length,
            num_workers=self.mpk.num_workers, num_local_schedulers=self.mpk.num_local_schedulers,
            kernel_dir=str(kernel_dir.resolve()))
        self.meta, self.length = meta, length
        # compile() binds only launch/init; the request reset lives on the same module.
        self.reset = self.mpk.launch_func.__self__.init_request_func
        self.events = []

    def generate(self, rows, count):
        torch = self.torch
        meta = self.meta
        batch, prompt = len(rows), self.prompt_length
        if meta["tokens"].shape != (batch, prompt + count):
            raise AssertionError("kernel compiled for a different shape")
        meta["tokens"][:, :prompt].copy_(torch.tensor(rows, dtype=torch.long), non_blocking=False)
        self.reset()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        self.mpk()
        end.record()
        torch.cuda.synchronize()
        steps = meta["step"].tolist()
        self.events.append(dict(N=count, kernel_ms=start.elapsed_time(end), steps=steps))
        if steps != [prompt + count - 1] * batch:
            raise AssertionError(f"MPK stopped early: steps {steps}, expected {prompt + count - 1}")
        return meta["tokens"][:, prompt:].tolist()

    def build_qwen3(self, a, meta):
        import mirage as mi
        torch, model = self.torch, self.model
        config = model.config
        moe = self.arch == "Qwen3MoeForCausalLM"
        hidden = config.hidden_size
        inter = config.moe_intermediate_size if moe else config.intermediate_size
        vocab = 153600
        q_heads, kv_heads, head_dim = (config.num_attention_heads, config.num_key_value_heads,
                                       config.head_dim)
        fused_1, fused_2 = (q_heads + 2 * kv_heads) * head_dim, 2 * inter
        workers, schedulers = mi.get_configurations_from_gpu(0)
        b = a.max_num_batched_requests
        buffers = dict(
            qo_indptr_buffer=torch.empty(b + 1, dtype=torch.int32, device="cuda"),
            paged_kv_indptr_buffer=torch.empty(b + 1, dtype=torch.int32, device="cuda"),
            paged_kv_indices_buffer=torch.empty(a.max_num_pages, dtype=torch.int32, device="cuda"),
            paged_kv_last_page_len_buffer=torch.empty(b, dtype=torch.int32, device="cuda"))
        mpk = mi.PersistentKernel(
            mode="offline", world_size=1, mpi_rank=0, num_workers=workers,
            num_local_schedulers=schedulers, num_remote_schedulers=0,
            max_seq_length=a.max_seq_length, max_num_batched_requests=b,
            max_num_batched_tokens=a.max_num_batched_tokens, max_num_pages=a.max_num_pages,
            page_size=a.page_size, eos_token_id=-1, meta_tensors=dict(meta, **buffers),
            profiler_tensor=None, trace_name="",
            spec_decode_config=mi.speculative.spec_decode_class(None, ngram_size=3, spec_length=3),
            use_cutlass_kernel=True)
        t = a.max_num_batched_tokens

        def tensor(name, dims, dtype=mi.bfloat16):
            return mpk.new_tensor(dims=dims, dtype=dtype, name=name, io_category="cuda_tensor")

        x = mpk.attach_input(torch_tensor=meta["input_tokens"], name="input_token")
        cos = mpk.attach_input(torch_tensor=self.position_embeddings[0][0, :4096, :],
                               name="cos_position_embedding")
        sin = mpk.attach_input(torch_tensor=self.position_embeddings[1][0, :4096, :],
                               name="sin_position_embedding")
        y = tensor("embed_out", (t, hidden))
        rms_qkv = tensor("rmsnorm_out_qkv", (t, hidden)) if moe else None
        rms = tensor("rmsnorm_out", (t, hidden))
        rms_moe = tensor("rmsnorm_out_moe", (t, hidden)) if moe else None
        attn_in = tensor("attn_in", (t, fused_1))
        attn_out = tensor("attn_out", (t, q_heads * head_dim))
        split = self.args.split_kv_cache
        if split:
            if moe:
                raise ValueError("demo_30B_A3B_hopper.py has no --split-kv-cache option")
            chunks = max(1, a.max_seq_length // 256)
            per_kv = chunks * q_heads // kv_heads
            lse = mpk.new_tensor(dims=(t, per_kv, kv_heads), strides=(chunks * q_heads, 1, per_kv),
                                 dtype=mi.float32, name="lse", io_category="cuda_tensor")
            attn_out_tmp = mpk.new_tensor(dims=(t, per_kv * head_dim, kv_heads),
                                          strides=(chunks * q_heads, 1, per_kv * head_dim),
                                          dtype=mi.bfloat16, name="attn_out_tmp", io_category="cuda_tensor")
        attn_proj_out = tensor("attn_proj_out", (t, hidden))
        tensor("all_reduce_buf", (1, t, hidden))
        tensor("attn_allreduce_out", (t, hidden))
        if moe:
            experts, topk = config.num_experts, config.num_experts_per_tok
            gate_out = tensor("moe_gate_out", (t, experts))
            routing = tensor("moe_routing_indices", (experts, t), mi.int32)
            mask = tensor("moe_mask", (experts + 1,), mi.int32)
            topk_weight = tensor("moe_topk_weight", (t, topk), mi.float32)
            mlp_mid = tensor("mlp_mid", (t, topk, fused_2))
            silu_out = tensor("silu_mul_out", (t, topk, inter))
            mlp_out = tensor("mlp_out", (t, topk, hidden))
            weighted = tensor("mlp_weighted_sum_out", (t, hidden))
        else:
            mlp_mid = tensor("mlp_mid", (t, fused_2))
            silu_out = tensor("silu_mul_out", (t, inter))
            mlp_out = tensor("mlp_out", (t, hidden))
        tensor("mlp_final", (t, hidden))
        argmax_in = tensor("argmax_in", (t, vocab))
        part_value = tensor("argmax_part_value", (t, mpk.num_workers))
        part_index = tensor("argmax_part_index", (t, mpk.num_workers), mi.int64)
        argmax_out = mpk.attach_input(torch_tensor=meta["output_tokens"], name="output_token")

        w = mpk.attach_input(torch_tensor=model.model.embed_tokens.weight, name="embed_tokens")
        mpk.embed_layer(input=x, weight=w, output=y, grid_dim=(1, 1, 1), block_dim=(256, 1, 1),
                        input_source=1)
        x = y
        k_all, v_all = self.kv_cache
        for i, layer in enumerate(model.model.layers):
            attach = mpk.attach_input
            w_norm = attach(torch_tensor=layer.input_layernorm.weight, name=f"layer_{i}_input_layernorm")
            w_q = attach(torch_tensor=layer.self_attn.q_proj.weight, name=f"layer_{i}_q_proj")
            w_k = attach(torch_tensor=layer.self_attn.k_proj.weight, name=f"layer_{i}_k_proj")
            w_v = attach(torch_tensor=layer.self_attn.v_proj.weight, name=f"layer_{i}_v_proj")
            w_qkv = mpk.shuffle_tensors(inputs=[w_q, w_k, w_v], shuffled_dim=0, num_groups=kv_heads,
                                        name=f"layer_{i}_qkv_proj")
            normed = rms_qkv if moe else rms
            mpk.rmsnorm_layer(input=x, weight=w_norm, output=normed, grid_dim=(t, 1, 1),
                              block_dim=(256, 1, 1))
            mpk.linear_layer(input=normed, weight=w_qkv, output=attn_in,
                             grid_dim=(grid_for_linear_layer(w_qkv.dim(0)), 1, 1), block_dim=(256, 1, 1))
            w_q_norm = attach(torch_tensor=layer.self_attn.q_norm.weight, name=f"layer_{i}_q_norm")
            w_k_norm = attach(torch_tensor=layer.self_attn.k_norm.weight, name=f"layer_{i}_k_norm")
            k_cache = attach(torch_tensor=k_all[i], name=f"layer_{i}_k_cache")
            v_cache = attach(torch_tensor=v_all[i], name=f"layer_{i}_v_cache")
            if split:
                mpk.paged_attention_split_kv_layer(
                    input=attn_in, k_cache=k_cache, v_cache=v_cache, q_norm=w_q_norm, k_norm=w_k_norm,
                    cos_pos_embed=cos, sin_pos_embed=sin, lse=lse, output=attn_out_tmp,
                    attention_params=(q_heads, chunks), grid_dim=(b, kv_heads, chunks),
                    block_dim=(256, 1, 1))
                mpk.paged_attention_split_kv_merge_layer(
                    lse=lse, output_tmp=attn_out_tmp, output=attn_out,
                    attention_params=(q_heads, head_dim), grid_dim=(b, kv_heads, 1), block_dim=(256, 1, 1))
            else:
                mpk.paged_attention_layer(input=attn_in, k_cache=k_cache, v_cache=v_cache,
                                          q_norm=w_q_norm, k_norm=w_k_norm, cos_pos_embed=cos,
                                          sin_pos_embed=sin, output=attn_out, grid_dim=(b, kv_heads, 1),
                                          block_dim=(256 if moe else 128, 1, 1))
            w = attach(torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj")
            mpk.linear_with_residual_layer(
                input=attn_out, weight=w, residual=x, output=attn_proj_out,
                grid_dim=((hidden // 64) if moe else grid_for_linear_layer(w.dim(0), with_residual=True), 1, 1),
                block_dim=(256, 1, 1))
            x = attn_proj_out
            w_norm = attach(torch_tensor=layer.post_attention_layernorm.weight,
                            name=f"layer_{i}_post_attn_layernorm")
            if moe:
                w_gate = attach(torch_tensor=layer.mlp.gate.weight, name=f"layer_{i}_moe_gate")
                w_gatedup = attach(torch_tensor=self.gate_up[i], name=f"layer_{i}_gate_proj")
                w_down = attach(torch_tensor=self.down[i], name=f"layer_{i}_down_proj")
                mpk.rmsnorm_layer(input=x, weight=w_norm, output=rms_moe, grid_dim=(t, 1, 1),
                                  block_dim=(256, 1, 1))
                mpk.linear_layer(input=rms_moe, weight=w_gate, output=gate_out,
                                 grid_dim=(experts // 16, 1, 1), block_dim=(256, 1, 1))
                mpk.moe_topk_softmax_routing_layer(input=gate_out, output=(topk_weight, routing, mask),
                                                   grid_dim=(1, 1, 1), block_dim=(256, 1, 1))
                mpk.moe_w13_linear_layer(input=rms_moe, weight=w_gatedup, moe_routing_indices=routing,
                                         moe_mask=mask, output=mlp_mid, grid_dim=(5, 24, 1),
                                         block_dim=(256, 1, 1))
                mpk.moe_silu_mul_layer(input=mlp_mid, output=silu_out, grid_dim=(t, topk, 1),
                                       block_dim=(256, 1, 1))
                mpk.moe_w2_linear_layer(input=silu_out, weight=w_down, moe_routing_indices=routing,
                                        moe_mask=mask, output=mlp_out, grid_dim=(4, 32, 1),
                                        block_dim=(256, 1, 1))
                mpk.moe_mul_sum_add_layer(input=mlp_out, weight=topk_weight, residual=x, output=weighted,
                                          grid_dim=(t, hidden // 256, 1), block_dim=(256, 1, 1))
                x = weighted
            else:
                w_gate = attach(torch_tensor=layer.mlp.gate_proj.weight, name=f"layer_{i}_gate_proj")
                w_up = attach(torch_tensor=layer.mlp.up_proj.weight, name=f"layer_{i}_up_proj")
                tasks = grid_for_rmsnorm_layer(w_gate.dim(0) + w_up.dim(0))
                w_gatedup = mpk.shuffle_tensors(inputs=[w_gate, w_up], shuffled_dim=0,
                                                num_groups=tasks // 2, name=f"layer_{i}_gatedup_proj")
                mpk.rmsnorm_layer(input=x, weight=w_norm, output=rms, grid_dim=(t, 1, 1),
                                  block_dim=(256, 1, 1))
                mpk.linear_layer(input=rms, weight=w_gatedup, output=mlp_mid,
                                 grid_dim=(grid_for_linear_layer(w_gatedup.dim(0)), 1, 1),
                                 block_dim=(256, 1, 1))
                mpk.silu_mul_layer(input=mlp_mid, output=silu_out, grid_dim=(tasks // 2, 1, 1),
                                   block_dim=(256, 1, 1))
                w = attach(torch_tensor=layer.mlp.down_proj.weight, name=f"layer_{i}_down_proj")
                mpk.linear_with_residual_layer(
                    input=silu_out, weight=w, residual=x, output=mlp_out,
                    grid_dim=(grid_for_linear_layer(w.dim(0), with_residual=True), 1, 1),
                    block_dim=(256, 1, 1))
                x = mlp_out
        w_norm = mpk.attach_input(torch_tensor=model.model.norm.weight, name="model_norm_weight")
        w_proj = mpk.attach_input(torch_tensor=self.lm_head, name="lm_head")
        mpk.rmsnorm_layer(input=x, weight=w_norm, output=rms, grid_dim=(t, 1, 1), block_dim=(256, 1, 1))
        mpk.linear_layer(input=rms, weight=w_proj, output=argmax_in,
                         grid_dim=((mpk.num_workers if moe else grid_for_linear_layer(w_proj.dim(0))), 1, 1),
                         block_dim=(256, 1, 1))
        mpk.argmax_partial_layer(input=argmax_in, output=(part_value, part_index),
                                 grid_dim=(mpk.num_workers, 1, 1), block_dim=(256, 1, 1),
                                 vocab_size=config.vocab_size)
        mpk.argmax_reduce_layer(input=(part_value, part_index), output=argmax_out, grid_dim=(1, 1, 1),
                                block_dim=(256, 1, 1))
        results = mpk.kn_graph.generate_task_graph(num_gpus=1, my_gpu_id=0)
        Path("task_graph_0.json").write_text(results["json_file"])
        Path("kernel_0.cu").write_text(results["cuda_code"])
        mpk.compile(output_dir=a.output_dir)
        return mpk


def mirage_revision():
    try:
        return subprocess.check_output(["git", "-C", str(MIRAGE_HOME), "rev-parse", "HEAD"],
                                       text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def main():
    args, ids = arguments("mpk", extend)
    require_guard()
    arch = architecture(args.model)
    engine = Engine(args, arch)
    engine.prompt_length = len(ids[0])
    config = {"dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "greedy": "argmax layers",
              "eos": "disabled (eos_token_id=-1)", "mode": "offline", "world_size": 1,
              "demo": DEMOS[arch], "mirage_home": str(MIRAGE_HOME),
              "mirage_revision": mirage_revision(), "page_size": args.page_size,
              "max_num_pages": args.max_num_pages,
              "max_num_batched_tokens": "B (equal to max_num_batched_requests, as in the demos)",
              "use_cutlass_kernel": True, "spec_decode": None,
              "split_kv_cache": args.split_kv_cache,
              "timed_scope": "prompt H2D copy + init_request_func + mpk() + synchronize + token D2H",
              "kernels": engine.compiles}

    def generate(rows, count):
        tokens = engine.generate(rows, count)
        out = args.out / f"B{len(rows)}"
        with (out / "mpk_events.jsonl").open("a") as stream:
            stream.write(json.dumps(engine.events[-1]) + "\n")
        return tokens

    measure(args, ids, "mpk", generate, config, synchronize=engine.torch.cuda.synchronize,
            prepare=engine.prepare)
    if engine.mpk is not None:
        engine.mpk.finalize()


if __name__ == "__main__":
    guarded_main(main)
