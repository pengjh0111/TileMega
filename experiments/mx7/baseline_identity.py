"""Archive the exact Python harness and invocation before a baseline launch."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('command', nargs=argparse.REMAINDER)
args = parser.parse_args()
files = ['scripts/mx_tp_baselines.sh', 'scripts/mx_gpu_group_job.sh',
         'python/tilemega/bench/gpu_guard.py', 'python/tilemega/bench/mx_smi.py',
         'experiments/mx7/baseline_identity.py']
files += [f'python/tilemega/serving/{name}.py' for name in (
    'baseline_common', 'measurement_policy', 'vllm_baseline', 'sglang_baseline',
    'torchcompile_tp_baseline', 'workload')]
files += ['configs/bench/measurement_policy_c550.json', 'configs/bench/guard_policy_c550.json',
          'configs/prompts/qwen_isl64.json']
archive = args.out / 'source_snapshot'
archive.mkdir(parents=True, exist_ok=False)
rows = {}
for name in files:
    source = Path(name)
    target = archive / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    rows[name] = hashlib.sha256(source.read_bytes()).hexdigest()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
identity = dict(captured_at_ns=time.time_ns(), command=command, source_files=rows,
                head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
(args.out / 'source_identity.json').write_text(json.dumps(identity, indent=2) + '\n')
