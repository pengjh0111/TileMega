#!/usr/bin/env python3
"""Collect isolated package metadata or a guarded, untimed CUDA canary."""
import argparse
import ctypes
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metadata():
    packages = {row.metadata['Name']: row.version for row in importlib.metadata.distributions()}
    result = dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                  packages=packages, sources={}, attention_backend=None, moe_backend=None,
                  backend_status='not selected until model initialization')
    for name in ('vllm', 'sglang', 'tensorrt-llm', 'torch'):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            continue
        files = sorted(str(path) for path in (dist.files or [])
                       if str(path).endswith('.py') and '..' not in Path(path).parts)
        digest = hashlib.sha256()
        for relative in files:
            path = Path(dist.locate_file(relative))
            if path.is_file():
                digest.update(relative.encode() + b'\0' + bytes.fromhex(sha(path)))
        result['sources'][name] = dict(version=dist.version, python_files=len(files),
                                      python_source_tree_sha256=digest.hexdigest())
        direct = dist.read_text('direct_url.json')
        if direct:
            value = json.loads(direct)
            result['sources'][name]['commit'] = value.get('vcs_info', {}).get('commit_id')
    return result


def canary(framework):
    sys.path.insert(0, str(ROOT / 'python'))
    from tilemega.serving.baseline_common import require_guard
    require_guard()
    result = dict(passed=False, pid=os.getpid(), framework=framework,
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  diagnostic_only=True, model_executed=False)
    driver = ctypes.CDLL('libcuda.so.1')
    result['cuInit'] = driver.cuInit(0)
    if result['cuInit']:
        error = ctypes.c_char_p()
        driver.cuGetErrorString(result['cuInit'], ctypes.byref(error))
        result.update(status='cuda_initialization_failed', error=error.value.decode())
        return result
    try:
        import torch
        torch.cuda.init()
        a = torch.ones((32, 32), device='cuda', dtype=torch.bfloat16)
        output = a @ a
        torch.cuda.synchronize()
        if not torch.equal(output.cpu(), torch.full((32, 32), 32, dtype=torch.bfloat16)):
            raise ValueError('BF16 matmul canary output differs')
        module = importlib.import_module(dict(vllm='vllm', sglang='sglang', trtllm='tensorrt_llm')[framework])
        result.update(passed=True, status='cuda_and_framework_import_passed',
                      torch_version=torch.__version__, torch_cuda=torch.version.cuda,
                      gpu_name=torch.cuda.get_device_name(),
                      capability=list(torch.cuda.get_device_capability()),
                      framework_file=module.__file__,
                      scope='Tiny BF16 kernel and framework import only; not model support or performance.')
    except Exception as error:
        result.update(status='canary_or_framework_import_failed', error=str(error),
                      traceback=traceback.format_exc())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('metadata', 'canary'), required=True)
    parser.add_argument('--framework', choices=('vllm', 'sglang', 'trtllm'), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    result = metadata() if args.mode == 'metadata' else canary(args.framework)
    result.update(evidence='verified', framework=args.framework)
    (args.out / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: value for key, value in result.items()
                      if key not in ('packages', 'sources', 'traceback')}), flush=True)
    return 0 if args.mode == 'metadata' or result['passed'] else 2


if __name__ == '__main__':
    raise SystemExit(main())
