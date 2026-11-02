"""Independent MPK (Mirage Persistent Kernel) baseline over frozen token IDs.

Imports no TileMega code. The task graphs replicate MPK's own demos at
/root/mirage-mpk (demo/qwen3/demo_hopper.py, demo/qwen3/demo_30B_A3B_hopper.py,
demo/llama3/demo.py); MPK itself is used unchanged.

One mpk() call serves whole requests (prefill and decode in one persistent
kernel) until every row reaches max_seq_length. That length is a compile-time
constant, so T_N runs a kernel compiled for prompt_length + N positions.

`--tp T` replicates demo/qwen3/demo_hopper.py's multi-GPU path and nothing
else: T MPI ranks, one card each, every rank loading the model{rank}-mp{T}
shard that MPK's own demo/qwen3/models/convert.py writes, per-rank attention
and MLP widths, and mpk.allreduce_layer over NVSHMEM buffers after o_proj and
down_proj. Rank 0 times the request; the others follow its broadcasts, because
a persistent kernel is one collective and every rank must enter it. MPK itself
stays unchanged.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import json

from baseline_common import (arguments, guarded_main, measure, require_guard,
                             shared_checkpoint_identity)

MIRAGE_HOME = Path(os.environ.get("MIRAGE_HOME", "/root/mirage-mpk"))
DEMOS = {"Qwen3ForCausalLM": "demo/qwen3/demo_hopper.py",
         "Qwen3MoeForCausalLM": "demo/qwen3/demo_30B_A3B_hopper.py",
         "LlamaForCausalLM": "demo/llama3/demo.py"}
# Only the dense Qwen3 demo carries a world_size > 1 path.
TP_DEMOS = ("Qwen3ForCausalLM",)
CONVERT = "demo/qwen3/models/convert.py"
# MPK compiles its NVSHMEM path against these; it reads the four names from the
# environment and falls back to /usr paths that do not exist here.
NVSHMEM_HOME = Path(os.environ.get("NVSHMEM_HOME", "/root/.venvs/nvshmem/nvidia/nvshmem"))
MPI_HOME = Path(os.environ.get("MPI_HOME", "/usr/lib/x86_64-linux-gnu/openmpi"))


class Unsupported(RuntimeError):
    pass


def extend(parser):
    parser.add_argument("--page-size", type=int, default=4096)
    # 16 pages x 4096 is the demos' CLI default, sized for one long-context
    # request; this protocol runs B requests of prompt + max_tokens tokens. 0
    # derives the budget MPK's page allocator actually draws on (kv_pages()).
    parser.add_argument("--max-num-pages", type=int, default=0)
    parser.add_argument("--split-kv-cache", action="store_true",
                        help="demo_hopper.py --split-kv-cache (Qwen3 dense only)")
    # The demos set max_num_batched_tokens = max_num_batched_requests; 0 keeps
    # that. Overriding it only varies the token-slot count, which separates a
    # fault that tracks the slots from one that tracks the requests.
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    # --tp T reads this rank's shard from here; the default sits beside the
    # checkpoint so one conversion serves every rank of the launch.
    parser.add_argument("--shard-dir", type=Path)
    # The ranks read shards, so the shared checkpoint's own manifest is no
    # longer beside --model; mpi_launch passes the source it was built from so
    # the measurement still records which checkpoint these weights came from.
    parser.add_argument("--tp-source-model", type=Path)


def kv_pages(batch, prompt, max_tokens, page_size):
    """Pages MPK needs for `batch` concurrent requests of this length.

    persistent_kernel.cuh draws ceil((step + new_tokens)/block_size) pages per
    request from one queue of MPK_MAX_NUM_PAGES, so the pool is the sum over
    the batch. Over-sizing it costs KV bytes that are never addressed.
    """
    if page_size < 1 or batch < 1 or prompt < 1 or max_tokens < 1:
        raise ValueError("invalid KV page budget inputs")
    return batch*((prompt + max_tokens + page_size - 1)//page_size)


def shard_checkpoint(model, tp, out):
    """This launch's per-rank checkpoints, written by MPK's own convert.py.

    The protocol's checkpoint ties lm_head to the embedding and stores the one
    tensor under the embedding's name. MPK's Qwen3ForCausalLM ties the pair in
    post_init(), so safetensors expects exactly one of the two names and takes
    `lm_head.weight` (convert.py drops the embedding for the same reason when
    the source carries both). The tied tensor is therefore renamed, not
    duplicated: loading it fills the embedding through the tie, which is what
    the TP1 path gets from HF's from_pretrained. convert.py runs unmodified.
    """
    from safetensors.torch import load_file, save_file
    script = MIRAGE_HOME / CONVERT
    if not script.exists():
        raise Unsupported(f"no MPK checkpoint converter at {script}")
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, str(script), "--hf-ckpt-path", str(model),
                    "--save-path", str(out), "--model-parallel", str(tp)], check=True)
    tied = bool(json.loads((model / "config.json").read_text()).get("tie_word_embeddings"))
    shards = []
    for rank in range(tp):
        path = out / f"model{rank}-mp{tp}.safetensors"
        if not path.exists():
            raise Unsupported(f"convert.py wrote no {path.name}")
        if tied:
            state = load_file(str(path))
            if "lm_head.weight" not in state:
                state["lm_head.weight"] = state.pop("model.embed_tokens.weight")
                save_file(state, str(path), metadata={"format": "pt"})
            del state
        shards.append(path.name)
    return shards


def mpi_launch(tp, shard_dir):
    """Re-run this script as T MPI ranks over the already claimed cards.

    MPK's TP needs an MPI world (demo_hopper.py takes its rank from mpi4py)
    and NVSHMEM symmetric memory, so the guarded process becomes mpirun and
    every rank inherits the claim through the forwarded environment. Returning
    mpirun's code rather than exec-ing keeps the guard's receipts intact.
    """
    environment = dict(os.environ, NVSHMEM_INC_PATH=str(NVSHMEM_HOME / "include"),
                       NVSHMEM_LIB_PATH=str(NVSHMEM_HOME / "lib"),
                       MPI_INC_PATH=str(MPI_HOME / "include"),
                       MPI_LIB_PATH=str(MPI_HOME / "lib"),
                       LD_LIBRARY_PATH=os.pathsep.join(
                           filter(None, [str(NVSHMEM_HOME / "lib"),
                                         os.environ.get("LD_LIBRARY_PATH", "")])))
    for name in ("NVSHMEM_INC_PATH", "NVSHMEM_LIB_PATH", "MPI_INC_PATH", "MPI_LIB_PATH"):
        root = Path(environment[name])
        if not root.is_dir():
            raise Unsupported(f"{name}={root} is not a directory")
    # mpirun forwards only what -x names, and the guard's own variables are
    # what each rank checks before it touches a card.
    forward = [name for name in (
        "PATH", "LD_LIBRARY_PATH", "PYTHONPATH", "PYTHONUNBUFFERED", "CUDA_HOME", "CUDA_PATH",
        "CUDA_VISIBLE_DEVICES", "TORCH_CUDA_ARCH_LIST", "MIRAGE_HOME", "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE", "OMP_NUM_THREADS", "NVSHMEM_INC_PATH", "NVSHMEM_LIB_PATH",
        "MPI_INC_PATH", "MPI_LIB_PATH", "TILEMEGA_DEVICE_GROUP", "TILEMEGA_DEVICE_INDEX",
        "TILEMEGA_TIMING", "TILEMEGA_GPU_PID_MODE", "TILEMEGA_GPU_LOCK_HELD")
        if name in environment]
    command = ["mpirun", "--allow-run-as-root", "--bind-to", "none", "-np", str(tp)]
    for name in forward:
        command += ["-x", name]
    command += [sys.executable, "-u", str(Path(__file__).resolve())]
    arguments, skip = list(sys.argv[1:]), False
    source = None
    for index, item in enumerate(arguments):
        if skip:
            skip = False
            continue
        if item == "--model":
            source, arguments[index + 1] = arguments[index + 1], str(shard_dir)
            skip = True
        elif item.startswith("--model="):
            source = item.split("=", 1)[1]
            arguments[index] = f"--model={shard_dir}"
    if source is None:
        raise ValueError("no --model in the command to re-launch")
    command += arguments + ["--tp-source-model", source]
    print(json.dumps({"mpk_tp_launcher": command}), flush=True)
    code = subprocess.call(command, env=environment)
    if code:
        raise RuntimeError(f"mpirun -np {tp} exited {code}")
    return code


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
    def __init__(self, args, arch, world=1, rank=0):
        import torch
        self.torch, self.args, self.arch = torch, args, arch
        self.world, self.rank = world, rank
        torch.set_default_dtype(torch.bfloat16)
        # One card per rank out of the claimed group, as demo_hopper.py does.
        torch.cuda.set_device(rank)
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
                if world > 1:
                    # demo_hopper.py's --model-path branch: the class shards
                    # itself from world_size, and this rank loads its shard.
                    from transformers import AutoConfig
                    from safetensors.torch import load_model
                    self.model = Qwen3ForCausalLM(
                        AutoConfig.from_pretrained(path), world, args.max_num_pages,
                        args.page_size)
                    load_model(self.model, f"{path}/model{rank}-mp{world}.safetensors")
                else:
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
        # MPK allocates its tensors with cudaMalloc, which cannot draw on the
        # torch caching allocator's freed blocks. Stacking 128 experts per
        # layer and dropping the per-expert modules churns ~57 GiB of
        # Qwen3-30B-A3B through that cache, so without this the driver sees
        # almost nothing free when MPK's launcher starts allocating.
        torch.cuda.empty_cache()
        self.mpk = None
        self.compiles = {}
        self.build_bytes = 0

    def prepare(self, batch, count):
        torch, args = self.torch, self.args
        if self.mpk is not None:
            self.mpk.finalize()
            self.mpk = None
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        # finalize_persistent_kernel() frees the runtime's task/event/queue
        # structures, not the fused weight buffers the generated launcher
        # cudaMallocs (per layer: q/k/v into one qkv, gate/up into one
        # gatedup, originals kept). Those 0.9 GiB (Qwen3-30B-A3B) to 8.5 GiB
        # (Llama-3.1-8B) per build are outside the torch allocator and are
        # never returned, so a process can only afford a few builds. Refuse
        # the one that would not fit instead of OOMing inside MPK.
        free = torch.cuda.mem_get_info()[0]
        if self.build_bytes and free < self.build_bytes:
            raise RuntimeError(
                f"{free/2**30:.2f} GiB free is below the {self.build_bytes/2**30:.2f} GiB "
                f"the previous MPK kernel build retained; run one process per "
                f"(batch, output length) cell rather than --batch all")
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
        if self.world > 1:
            # Each rank builds its own task graph (my_gpu_id differs), so the
            # graph dumps and the launcher must not share a directory.
            kernel_dir = kernel_dir / f"rank{self.rank}"
        kernel_dir.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()
        cwd = os.getcwd()
        os.chdir(kernel_dir)  # the demo graph code writes task_graph_0.json and kernel_0.cu to cwd
        try:
            shape = argparse.Namespace(
                max_num_batched_requests=batch,
                max_num_batched_tokens=args.max_num_batched_tokens or batch,
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
            retained_bytes=max(free - self.torch.cuda.mem_get_info()[0], 0),
            kernel_dir=str(kernel_dir.resolve()))
        self.build_bytes = max(self.build_bytes,
                               self.compiles[f"B{batch}_N{count}"]["retained_bytes"])
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
        world, rank = self.world, self.rank
        hidden = config.hidden_size
        inter = config.moe_intermediate_size if moe else config.intermediate_size
        vocab = 153600
        q_heads, kv_heads, head_dim = (config.num_attention_heads, config.num_key_value_heads,
                                       config.head_dim)
        # The demo's per-rank widths: heads and the MLP split, hidden, the
        # vocabulary and lm_head replicated (convert.py shards neither).
        local_q, local_kv = q_heads // world, kv_heads // world
        fused_1, fused_2 = (q_heads + 2 * kv_heads) * head_dim, 2 * inter
        workers, schedulers = mi.get_configurations_from_gpu(rank)
        b = a.max_num_batched_requests
        buffers = dict(
            qo_indptr_buffer=torch.empty(b + 1, dtype=torch.int32, device="cuda"),
            paged_kv_indptr_buffer=torch.empty(b + 1, dtype=torch.int32, device="cuda"),
            paged_kv_indices_buffer=torch.empty(a.max_num_pages, dtype=torch.int32, device="cuda"),
            paged_kv_last_page_len_buffer=torch.empty(b, dtype=torch.int32, device="cuda"))
        mpk = mi.PersistentKernel(
            mode="offline", world_size=world, mpi_rank=rank, num_workers=workers,
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

        def shared(name, dims):
            # The all-reduce reads and writes peer memory, so its operands are
            # symmetric NVSHMEM allocations once there is more than one rank.
            return mpk.new_tensor(dims=dims, dtype=mi.bfloat16, name=name,
                                  io_category="nvshmem_tensor" if world > 1 else "cuda_tensor")

        x = mpk.attach_input(torch_tensor=meta["input_tokens"], name="input_token")
        cos = mpk.attach_input(torch_tensor=self.position_embeddings[0][0, :4096, :],
                               name="cos_position_embedding")
        sin = mpk.attach_input(torch_tensor=self.position_embeddings[1][0, :4096, :],
                               name="sin_position_embedding")
        y = tensor("embed_out", (t, hidden))
        rms_qkv = tensor("rmsnorm_out_qkv", (t, hidden)) if moe else None
        rms = tensor("rmsnorm_out", (t, hidden))
        rms_moe = tensor("rmsnorm_out_moe", (t, hidden)) if moe else None
        attn_in = tensor("attn_in", (t, fused_1 // world))
        attn_out = tensor("attn_out", (t, local_q * head_dim))
        split = self.args.split_kv_cache
        if split:
            if moe:
                raise ValueError("demo_30B_A3B_hopper.py has no --split-kv-cache option")
            chunks = max(1, a.max_seq_length // 256)
            per_kv = chunks * local_q // local_kv
            lse = mpk.new_tensor(dims=(t, per_kv, local_kv), strides=(chunks * local_q, 1, per_kv),
                                 dtype=mi.float32, name="lse", io_category="cuda_tensor")
            attn_out_tmp = mpk.new_tensor(dims=(t, per_kv * head_dim, local_kv),
                                          strides=(chunks * local_q, 1, per_kv * head_dim),
                                          dtype=mi.bfloat16, name="attn_out_tmp", io_category="cuda_tensor")
        attn_proj_out = shared("attn_proj_out", (t, hidden))
        allreduce_buf = shared("all_reduce_buf", (world, t, hidden))
        attn_allreduce_out = shared("attn_allreduce_out", (t, hidden))
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
            mlp_mid = tensor("mlp_mid", (t, fused_2 // world))
            silu_out = tensor("silu_mul_out", (t, inter // world))
            mlp_out = shared("mlp_out", (t, hidden))
        mlp_final = shared("mlp_final", (t, hidden))
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
            w_qkv = mpk.shuffle_tensors(inputs=[w_q, w_k, w_v], shuffled_dim=0, num_groups=local_kv,
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
                    attention_params=(local_q, chunks), grid_dim=(b, local_kv, chunks),
                    block_dim=(256, 1, 1))
                mpk.paged_attention_split_kv_merge_layer(
                    lse=lse, output_tmp=attn_out_tmp, output=attn_out,
                    attention_params=(local_q, head_dim), grid_dim=(b, local_kv, 1), block_dim=(256, 1, 1))
            else:
                mpk.paged_attention_layer(input=attn_in, k_cache=k_cache, v_cache=v_cache,
                                          q_norm=w_q_norm, k_norm=w_k_norm, cos_pos_embed=cos,
                                          sin_pos_embed=sin, output=attn_out, grid_dim=(b, local_kv, 1),
                                          block_dim=(256 if moe else 128, 1, 1))
            w = attach(torch_tensor=layer.self_attn.o_proj.weight, name=f"layer_{i}_o_proj")
            mpk.linear_with_residual_layer(
                input=attn_out, weight=w, residual=x, output=attn_proj_out,
                grid_dim=((hidden // 64) if moe else grid_for_linear_layer(w.dim(0), with_residual=True), 1, 1),
                block_dim=(256, 1, 1))
            x = attn_proj_out
            if world > 1:
                # o is row sharded, so its partial sums are reduced across the
                # ranks before the residual branch continues.
                mpk.allreduce_layer(input=attn_proj_out, buffer=allreduce_buf,
                                    output=attn_allreduce_out, grid_dim=(hidden // 64, 1, 1),
                                    block_dim=(256, 1, 1))
                x = attn_allreduce_out
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
                if world > 1:
                    mpk.allreduce_layer(input=mlp_out, buffer=allreduce_buf, output=mlp_final,
                                        grid_dim=(hidden // 64, 1, 1), block_dim=(256, 1, 1))
                    x = mlp_final
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
        results = mpk.kn_graph.generate_task_graph(num_gpus=world, my_gpu_id=rank)
        Path(f"task_graph_{rank}.json").write_text(results["json_file"])
        Path(f"kernel_{rank}.cu").write_text(results["cuda_code"])
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
    world = args.tp
    devices = require_guard(world if world > 1 else None)
    arch = architecture(args.model)
    comm, rank = None, 0
    if world > 1:
        if arch not in TP_DEMOS:
            raise Unsupported(f"no MPK demo runs {arch} at world_size > 1 "
                              f"(tensor parallel demos: {list(TP_DEMOS)})")
        if "OMPI_COMM_WORLD_RANK" not in os.environ:
            # The outer, guarded process converts the checkpoint once and then
            # becomes mpirun; it has touched no card yet.
            shards = args.shard_dir or args.model.parent / f"{args.model.name}-mp{world}"
            shard_checkpoint(args.model, world, shards)
            return mpi_launch(world, shards)
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        if comm.Get_size() != world:
            raise ValueError(f"--tp {world} but MPI world {comm.Get_size()}")
    batch = 16 if args.batch == "all" else int(args.batch)
    needed = kv_pages(batch, len(ids[0]), args.max_tokens, args.page_size)
    if not args.max_num_pages:
        args.max_num_pages = needed
    elif args.max_num_pages < needed:
        raise ValueError(f"--max-num-pages {args.max_num_pages} is below the "
                         f"{needed} pages MPK allocates for B{batch}")
    engine = Engine(args, arch, world, rank)
    engine.prompt_length = len(ids[0])
    config = {"dtype": "bfloat16", "kv_cache_dtype": "bfloat16", "greedy": "argmax layers",
              "eos": "disabled (eos_token_id=-1)", "mode": "offline", "world_size": world,
              "tp_devices": devices,
              "launcher": ("mpirun -np T, demo_hopper.py's MPI path" if world > 1
                           else "single process"),
              "tp_shards": (f"model<rank>-mp{world}.safetensors from "
                            f"{CONVERT} (lm_head and the embedding replicated, "
                            f"q/k/v/gate/up column sharded, o/down row sharded)"
                            if world > 1 else None),
              "tp_collective": ("mpk.allreduce_layer over NVSHMEM symmetric buffers after "
                                "o_proj and down_proj" if world > 1 else None),
              "tp_source_checkpoint": (shared_checkpoint_identity(args.tp_source_model, "auto")
                                       if world > 1 and args.tp_source_model else None),
              "demo": DEMOS[arch], "mirage_home": str(MIRAGE_HOME),
              "mirage_revision": mirage_revision(), "page_size": args.page_size,
              "max_num_pages": args.max_num_pages,
              "kv_pages_required": needed,
              # The demos default this to 8, not to B; B is this wrapper's own
              # choice, so do not describe it as the demos' setting.
              "max_num_batched_tokens": (args.max_num_batched_tokens or
                                         "B (equal to max_num_batched_requests)"),
              "use_cutlass_kernel": True, "spec_decode": None,
              "kernels_reported_for_rank": rank,
              "split_kv_cache": args.split_kv_cache,
              "timed_scope": "prompt H2D copy + init_request_func + mpk() + synchronize + token D2H",
              "kernels": engine.compiles}

    def generate(rows, count):
        tokens = engine.generate(rows, count)
        out = args.out / f"B{len(rows)}"
        with (out / "mpk_events.jsonl").open("a") as stream:
            stream.write(json.dumps(engine.events[-1]) + "\n")
        return tokens

    # One persistent kernel is one collective: every rank has to build the same
    # kernel and enter the same mpk() call, so rank 0 announces each step it is
    # about to take and the others follow. Replaying measure()'s loop on the
    # workers instead would desynchronize them the moment it reorders the token
    # counts or retries a rejected round.
    def driven(action, function):
        def call(first, second):
            comm.bcast((action, first if isinstance(first, int) else len(first), second), root=0)
            return function(first, second)
        return call

    if rank == 0:
        measure(args, ids, "mpk", driven("generate", generate) if comm else generate, config,
                synchronize=engine.torch.cuda.synchronize,
                prepare=driven("prepare", engine.prepare) if comm else engine.prepare)
        if comm:
            # Untimed, after every measurement is written, on the kernel the
            # last cell already built: each rank reduces to the full hidden
            # state and runs its own argmax over a replicated lm_head, so the
            # ranks must agree token for token. Checking it inside generate()
            # would put an MPI collective inside the timed scope.
            shape = (batch, args.max_tokens)
            comm.bcast(("verify", *shape), root=0)
            tables = comm.gather(engine.generate([list(row) for row in ids[:shape[0]]],
                                                 shape[1]), root=0)
            agree = all(table == tables[0] for table in tables)
            write = args.out / f"B{shape[0]}" / "tp_rank_agreement.json"
            write.write_text(json.dumps({"ranks": len(tables), "batch": shape[0],
                                         "tokens": shape[1], "identical": agree}) + "\n")
            comm.bcast(("stop", 0, 0), root=0)
            if not agree:
                raise AssertionError("MPK ranks generated different tokens")
    else:
        while True:
            action, batch, count = comm.bcast(None, root=0)
            if action == "stop":
                break
            if action == "prepare":
                engine.prepare(batch, count)
            elif action == "verify":
                comm.gather(engine.generate([list(row) for row in ids[:batch]], count), root=0)
            else:
                engine.generate([list(row) for row in ids[:batch]], count)
    if engine.mpk is not None:
        engine.mpk.finalize()
    if comm:
        comm.Barrier()


if __name__ == "__main__":
    guarded_main(main)
