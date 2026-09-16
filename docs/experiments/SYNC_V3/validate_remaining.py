#!/usr/bin/env python3
"""Serialize remaining GPU validation after the in-progress C1 scan."""
import json
from pathlib import Path
import re
import subprocess
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def c1_complete():
    for model in ('gqa2', 'mha4'):
        for seq, past in ((1, 0), (128, 512), (2048, 0)):
            folder = HERE / 'c1/seqscan' / f'{model}_s{seq}_p{past}'
            for i in range(50):
                log = folder / f'r{i}.log'
                meta = log.with_suffix('.json')
                if not log.exists() or not meta.exists():
                    return False
                if re.findall(r'^RESULT status=(\S+)', log.read_text(), re.M) != ['PASS'] or json.loads(meta.read_text())['exit_code']:
                    raise ValueError(f'C1 subset failed: {log}')
    return True


def main():
    started = time.monotonic()
    while not c1_complete():
        if time.monotonic() - started > 3600:
            raise TimeoutError('C1 scan has not completed within one hour')
        time.sleep(5)
    print('C1 subset raw logs complete: starting serialized dependent validation', flush=True)
    runner = str(REPO / 'docs/experiments/FENCE/run.py')
    operations = [
        ('c2', 'correctness'), ('c2', 'seqscan'), ('c2_dependency', 'correctness'),
        ('local2', 'correctness'), ('local4', 'correctness'),
        ('window2', 'correctness'), ('window4', 'correctness'), ('sharded_red', 'correctness')]
    manifest = []
    for configuration, phase in operations:
        cmd = ['python3', runner, phase, '--raw', str(HERE / configuration)]
        print(f'VALIDATE {configuration} {phase}', flush=True)
        result = subprocess.run(cmd, cwd=REPO)
        manifest.append(dict(configuration=configuration, phase=phase, command=cmd, exit_code=result.returncode))
        (HERE / 'validation_commands.json').write_text(json.dumps(manifest, indent=2)+'\n')
        result.check_returncode()
    subprocess.run(['python3', str(HERE / 'trace_dependency.py'), 'run'], cwd=REPO, check=True)
    print('VALIDATE complete', flush=True)


if __name__ == '__main__':
    main()
