"""Check the pinned Mirage registry and SM80 operators without allocating a GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from a100_baselines import BATCHES, MODELS, ROOT, inputs, sha
from tilemega.bench.scheduler import write_json

REVISION = '63af900000d8928a885f7af7889a3448621e18a9'
OUT = ROOT / 'runs/a800/mpk/preflight.json'


def main():
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('run MPK preflight on the login node')
    source = ROOT.parent / 'software/mpk-63af900'
    os.environ['MIRAGE_HOME'] = str(source)
    sys.path.insert(0, str(ROOT/'python/tilemega/serving'))
    import mpk_baseline as harness
    import mirage
    from mirage.mpk.persistent_kernel import PersistentKernel
    import inspect

    git = ['git', '-C', str(source)]
    revision = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION or Path(mirage.__file__).resolve().parents[2] != source:
        raise RuntimeError('unexpected Mirage revision or Python module')
    diff = subprocess.check_output([*git, 'diff', '--binary'])
    for name in ('moe_w13_linear_layer', 'moe_w2_linear_layer'):
        method = inspect.getsource(getattr(PersistentKernel, name))
        if ('elif self.target_cc == 90:' not in method or 'if self.target_cc == 100:' not in method
                or 'else:\n            assert False' not in method):
            raise RuntimeError(f'MPK operator dispatch changed for {name}')
    header = source / 'include/mirage/persistent_kernel/tasks/ampere/multitoken_paged_attention.cuh'
    body = header.read_text()
    if 'else if ((MAX_TOKENS * NUM_QO_HEADS) <= 64)' not in body or '} else {' in body:
        raise RuntimeError('review the Ampere attention dispatch for this source')
    target = ROOT/'configs/targets/sm_80_a800_sxm4_80gb.json'
    if 'A800' not in json.loads(target.read_text())['device_provenance']['name']:
        raise RuntimeError('incorrect hardware profile')
    cells = []
    for model in MODELS:
        config, _ = inputs(model)
        cfg = json.loads((config/'config.json').read_text())
        reason = ''
        try:
            arch = harness.architecture(config)
        except harness.Unsupported as e:
            if 'Unknown builder:' not in str(e):
                raise
            reason = str(e)
        else:
            if arch == 'Qwen3MoeForCausalLM':
                reason = 'MPK moe_w13_linear_layer and moe_w2_linear_layer register BF16 tasks only for CC90 and CC100'
        for batch in BATCHES:
            refusal = reason
            group = cfg['num_attention_heads'] // cfg['num_key_value_heads']
            if not refusal and group * batch > 64:
                refusal = f'Ampere paged attention has no implementation for {batch} token slots × {group} queries per KV head'
            cells.append(dict(model=model, batch=batch, config_sha256=sha(config/'config.json'),
                              status='N/A' if refusal else 'requires_gpu', reason=refusal))
    from a800_mpk_serving import gate_grid, linear_grid
    for cell in cells:
        if cell['status'] != 'requires_gpu':
            continue
        model = cell['model']
        cfg = json.loads((inputs(model)[0]/'config.json').read_text())
        if cfg['architectures'][0] == 'Qwen3ForCausalLM':
            qkv = (cfg['num_attention_heads'] + 2*cfg['num_key_value_heads'])*cfg['head_dim']
            gu = 2*cfg['intermediate_size']
            if qkv % linear_grid(qkv) or (qkv//linear_grid(qkv))%64 or gate_grid(gu) != linear_grid(gu):
                raise ValueError(f'invalid Ampere graph dimensions for {model}')
    extension = Path(mirage.core.__file__)
    result = dict(revision=revision, diff_sha256=hashlib.sha256(diff).hexdigest(),
                  extension_sha256=sha(extension), harness_sha256=sha(harness.__file__),
                  serving_sha256=sha(ROOT/'scripts/a800_mpk_serving.py'),
                  model_serving_sha256={'llama3_1_8b':sha(ROOT/'scripts/a800_mpk_llama8.py')},
                  attention_sha256=sha(header), target_sha256=sha(target), cells=cells)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    write_json(OUT, result)
    print(json.dumps({state: sum(c['status'] == state for c in cells)
                      for state in ('N/A', 'requires_gpu')}))


if __name__ == '__main__':
    main()
