"""Teacher-forced routing comparison, kept outside every timing path."""
import ctypes as C
import json
from pathlib import Path


def index_names(bridge, layers):
    calls = [n for n in bridge['nodes'] if n['target'] == 'tilemega.moe_experts.default']
    if len(calls) != layers:
        raise ValueError('one expert boundary per decoder layer is required')
    return [n['inputs'][1] for n in calls]


def capture(plan, names, batch, seq, top_k):
    import torch
    library = plan.library
    read = getattr(library.lib, 'tm_plan_read_buffer', None)
    if read is None:
        raise ValueError('routing diagnostics require a rebuilt plan with buffer read support')
    read.argtypes = [C.c_void_p, C.c_uint32, C.c_void_p, C.c_uint64]
    read.restype = C.c_int
    by_name = {b.name: (i, b) for i, b in enumerate(library.buffers)}
    layers = []
    for name in names:
        i, info = by_name[name]
        if info.dtype != 2 or info.elements_constant + batch * info.elements_per_batch != batch * seq * top_k:
            raise ValueError('unexpected routing index dtype or extent: ' + name)
        data = torch.empty((batch, seq, top_k), dtype=torch.int32)
        code = read(plan.handle, i, data.data_ptr(), data.numel() * data.element_size())
        if code:
            raise RuntimeError(f'routing buffer read failed ({code}): {name}')
        layers.append(data[:, -1].tolist())
    # [layer][batch][rank], one step at a time, with no GPU tensors retained.
    return layers


def agreement(actual, expected):
    import torch
    a, b = torch.as_tensor(actual), torch.as_tensor(expected)
    if a.shape != b.shape or a.ndim != 4 or not a.numel():
        raise ValueError('routing tables must agree as [step,layer,batch,rank]')
    same = a.sort(-1).values.eq(b.sort(-1).values).all(-1)
    return dict(positions=same.numel(), exact_set_ratio=float(same.double().mean()),
                exact_rank_ratio=float(a.eq(b).all(-1).double().mean()),
                per_layer_set_ratio=same.double().mean((0, 2)).tolist())


def compare_hf(checkpoint, prompts, tokens, observed, out):
    import torch
    from transformers import AutoModelForCausalLM
    from tilemega.cache import atomic_json, file_sha
    steps, layers, batch = len(observed), len(observed[0]), len(prompts)
    if steps != len(tokens[0]) or any(len(row) != steps for row in tokens):
        raise ValueError('tokens and observed routing steps differ')
    model = AutoModelForCausalLM.from_pretrained(checkpoint, torch_dtype=torch.bfloat16,
        attn_implementation='sdpa', experts_implementation='grouped_mm').cuda().eval()
    if len(model.model.layers) != layers:
        raise ValueError('routing layer count differs from checkpoint')
    expected = torch.empty((steps, layers, batch, model.config.num_experts_per_tok), dtype=torch.int64)
    for b, prompt in enumerate(prompts):
        captures = {}
        def hook(layer):
            def save(module, args, output):
                captures[layer] = output[2].detach().cpu()
            return save
        handles = [layer.mlp.gate.register_forward_hook(hook(i)) for i, layer in enumerate(model.model.layers)]
        try:
            sequence = torch.tensor([prompt + tokens[b][:-1]], device='cuda', dtype=torch.long)
            with torch.inference_mode():
                model(sequence, use_cache=False)
            for layer in range(layers):
                expected[:, layer, b] = captures[layer][len(prompt)-1:len(prompt)-1+steps]
        finally:
            for handle in handles:
                handle.remove()
    report = dict(agreement(observed, expected), evidence='verified',
        scope='teacher-forced routing on identical prompt and generated token history; diagnostic, no acceptance threshold',
        checkpoint=str(checkpoint), config_sha256=file_sha(Path(checkpoint)/'config.json'))
    atomic_json(out, report)
    return report
