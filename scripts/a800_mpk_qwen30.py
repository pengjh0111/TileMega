"""Pin the separately built Qwen3 MoE MPK source and SM80 runtime."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

from a100_baselines import ROOT, sha
from a800_mpk_preflight import REVISION
from tilemega.bench.scheduler import write_json

SOURCE = ROOT.parent / 'software/mpk-63af900-qwen30-sm80-kv'
PATCH = ROOT / 'scripts/mpk_patches/mpk-63af900-sm80-qwen3-moe.patch'


def fingerprint():
    git = ['git', '-C', str(SOURCE)]
    revision = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise ValueError('unexpected Mirage revision')
    subprocess.run([*git, 'apply', '--check', '--reverse', str(PATCH)], check=True)
    diff = subprocess.check_output([*git, 'diff', '--binary'])
    extension, = (SOURCE/'python/mirage').glob('core*.so')
    runtime = SOURCE/'build/libmirage_runtime.a'
    return dict(
        revision=revision, sm80_qwen3_moe=True,
        git_source=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'],
                                           text=True).strip(),
        diff_sha256=hashlib.sha256(diff).hexdigest(), patch_sha256=sha(PATCH),
        extension_sha256=sha(extension), runtime_sha256=sha(runtime),
        harness_sha256=sha(ROOT/'python/tilemega/serving/mpk_baseline.py'),
        serving_sha256=sha(ROOT/'scripts/a800_mpk_serving.py'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    if os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('bind the MPK port on the login node')
    import mirage
    from mirage.mpk.persistent_kernel import PersistentKernel

    if Path(mirage.__file__).resolve().parents[2] != SOURCE:
        raise ValueError('imported the wrong MPK runtime')
    for name in ('w13', 'w2'):
        if f'moe_{name}_linear_sm80' not in getattr(
                PersistentKernel, f'moe_{name}_linear_layer').__code__.co_consts:
            raise ValueError(f'missing SM80 dispatch for {name}')
    result = fingerprint()
    write_json(args.out, result)
    print(json.dumps(result, sort_keys=True))


if __name__ == '__main__':
    main()
