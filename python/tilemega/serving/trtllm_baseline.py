"""Independent TensorRT-LLM 0.21 PyTorch backend baseline."""
import json
import os
from baseline_common import arguments, guarded_main, measure, require_guard
import sys
from pathlib import Path

# Standalone execution must not let serving/trace.py shadow stdlib trace.
sys.path = [entry for entry in sys.path
            if Path(entry or ".").resolve() != Path(__file__).resolve().parent]


LAUNCHER = "mpirun -np T --bind-to none trtllm-llmapi-launch"


def launch(tp):
    """Run this script again under TRT-LLM's own multi-GPU launcher.

    In-process, tp > 1 makes LLM() create an MpiPoolSession whose MPI_Comm_spawn
    never returns in this container (Open MPI 4.1.2: a bare mpi4py
    MPIPoolExecutor hangs too, singleton or under mpirun, while a static mpirun
    world works). trtllm-llmapi-launch runs rank 0 as this client and the other
    ranks as workers of the existing COMM_WORLD; no engine option changes.
    Returns the client's own exit status: the launcher's leader node can
    segfault in its CublasMMWrapper destructor after the client has finished.
    """
    import resource, subprocess, tempfile, time
    env = dict(os.environ, OMPI_ALLOW_RUN_AS_ROOT="1", OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1",
               TILEMEGA_TRTLLM_LAUNCHER=LAUNCHER,
               PATH=f"{Path(sys.executable).parent}:{os.environ.get('PATH', '')}")
    env.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
    status = Path(tempfile.mkdtemp(prefix="tilemega-trtllm-")) / "client_status"
    env["TILEMEGA_TRTLLM_CLIENT_STATUS"] = str(status)
    # core_pattern pipes to the host's apport, which never serves this
    # container, so a crashing descendant hangs instead of exiting. A core
    # limit of exactly 1 makes the kernel abort piped dumps (fs/coredump.c).
    _, maximum = resource.getrlimit(resource.RLIMIT_CORE)
    resource.setrlimit(resource.RLIMIT_CORE, (1, maximum))
    command = ["mpirun", "-np", str(tp), "--bind-to", "none", "--allow-run-as-root",
               str(Path(sys.executable).parent / "trtllm-llmapi-launch"),
               sys.executable, "-u", str(Path(__file__).resolve()), *sys.argv[1:]]
    child = subprocess.Popen(command, env=env, start_new_session=True)
    while child.poll() is None and not status.exists():
        time.sleep(1)
    try:
        child.wait(timeout=120)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, 9)
        child.wait()
    if not status.exists():
        return child.returncode or 1
    code = int(status.read_text().strip() or 1)
    if code == 0 and child.returncode:
        print(f"trtllm-llmapi-launch exited {child.returncode} after the client finished", flush=True)
    return code


def main():
    args, ids = arguments("tensorrt-llm")
    require_guard(args.tp)
    if args.tp > 1 and os.environ.get("TLLM_SPAWN_PROXY_PROCESS") != "1":
        raise SystemExit(launch(args.tp))
    if args.tp > 1:
        # The MPI worker pool needs these in the environment before the import,
        # and must not be forced into the single-process path.
        os.environ["OMPI_ALLOW_RUN_AS_ROOT"] = "1"
        os.environ["OMPI_ALLOW_RUN_AS_ROOT_CONFIRM"] = "1"
        os.environ.pop("TLLM_WORKER_USE_SINGLE_PROCESS", None)
    from tensorrt_llm import LLM, SamplingParams
    from tensorrt_llm.llmapi import KvCacheConfig
    options = dict(backend="pytorch", dtype="bfloat16", load_format=args.load_format,
                   tensor_parallel_size=args.tp,
                   skip_tokenizer_init=True, max_batch_size=16,
                   max_seq_len=max(2048, len(ids[0])+args.max_tokens),
                   use_cuda_graph=True, cuda_graph_batch_sizes=[1, 2, 4, 8, 16])
    llm = LLM(model=str(args.model), **options,
              kv_cache_config=KvCacheConfig(enable_block_reuse=False, free_gpu_memory_fraction=0.85))
    config = {**options, "kv_cache_dtype": "bfloat16 (unquantized model dtype)",
              "enable_block_reuse": False, "speculative_decoding": False,
              "worker_use_single_process": os.environ.get("TLLM_WORKER_USE_SINGLE_PROCESS", "0") == "1",
              "free_gpu_memory_fraction": 0.85,
              "launcher": os.environ.get("TILEMEGA_TRTLLM_LAUNCHER", "in-process"),
              "attention_backend": str(getattr(llm.args, "attn_backend", "TRTLLM"))}

    eos = json.loads((args.model/"config.json").read_text()).get("eos_token_id", 0)
    eos = eos[0] if isinstance(eos, list) else eos

    def generate(rows, count):
        result = llm.generate([{"prompt_token_ids": row} for row in rows],
                             SamplingParams(max_tokens=count, ignore_eos=True, top_k=1,
                                            end_id=eos or 0, pad_id=eos or 0, detokenize=False),
                             use_tqdm=False)
        if len(result) != len(rows):
            raise AssertionError("TensorRT-LLM output batch changed")
        for i, item in enumerate(result):
            if getattr(item, "prompt_token_ids", rows[i]) != rows[i]:
                raise AssertionError("TensorRT-LLM output prompt order changed")
        return [list(item.outputs[0].token_ids) for item in result]
    try:
        measure(args, ids, "tensorrt-llm", generate, config)
    finally:
        llm.shutdown()


def report(code):
    # Under the launcher the supervising parent decides the exit status from this.
    path = os.environ.get("TILEMEGA_TRTLLM_CLIENT_STATUS")
    if path and os.environ.get("TLLM_SPAWN_PROXY_PROCESS") == "1":
        Path(path).write_text(f"{code}\n")


if __name__ == "__main__":
    try:
        guarded_main(main)
    except SystemExit as exit:
        report(exit.code if isinstance(exit.code, int) else 1)
        raise
    except BaseException:
        report(1)
        raise
    report(0)
