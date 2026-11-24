"""Refresh the native constants used by the two SQ2 dense workloads."""

import subprocess

from calibration_queue import CALIB, REPRO, execute
from guarded import ROOT, read, sha, write


def main():
    legacy = CALIB / 'legacy'
    out = legacy / 'dense-measurement'
    out.mkdir(parents=True, exist_ok=True)
    original = legacy / 'constants'
    dense = legacy / 'constants-sq2'
    modes = [
        ('base', dense, '--sq2-base'),
        ('epilogue', dense, '--sq2-epilogue'),
        ('attention', original, '--attention-only'),
        ('gates', original, '--gates-only'),
        ('dsmem-merge', original, '--dsmem-merge-only'),
    ]
    jobs = [
        {'id': f'constants-{mode}-f{repeat}', 'command': [str(binary), flag],
         'binary_sha256': sha(binary)}
        for repeat in range(5) for mode, binary, flag in modes
    ]
    existing = legacy / 'measurement/items/constants-attention-f0/attempt0000'
    reused = out / 'items/constants-attention-f0/attempt0000'
    if existing.exists() and not reused.exists():
        guard = read(existing / 'guard/guard_result.json')
        command = read(existing / 'job/physical-device-command.json')
        assert guard['code'] == 0 and command['command'] == [str(original), '--attention-only']
        original_job = next(j for j in read(legacy / 'measurement/jobs.json') if j['id'] == 'constants-attention-f0')
        assert original_job['binary_sha256'] == sha(original)
        reused.parent.mkdir(parents=True, exist_ok=True)
        reused.symlink_to(existing, target_is_directory=True)
    predeclared = out / 'jobs.json'
    if predeclared.exists():
        assert read(predeclared) == jobs
    else:
        write(predeclared, jobs)
    bound = read(CALIB / 'body_binding.json')['header_sha256']
    for path, digest in bound.items():
        assert sha(ROOT / path) == digest, path
    assert read(REPRO / 'summary.json')['pass']
    cards = read(REPRO / 'reproduction_inputs.json')['authorized_cards']
    try:
        state = execute(jobs, out / 'items', cards, REPRO / 'guard_policy.json')
    finally:
        for path, digest in bound.items():
            assert sha(ROOT / path) == digest, path
    for name, result in state.items():
        link = out / f'{name}.log'
        if not link.exists():
            link.symlink_to(result['log'])
    write(out / 'state.json', state)
    subprocess.run([
        'python3', str(ROOT / 'scripts/collect_sm90r3_constants.py'), '--state', str(out / 'state.json'),
        '--constants', *(f'constants-{mode}' for mode, _, _ in modes), '--processes', '5',
        '--csv', str(out / 'constants.csv'), '--target', str(ROOT / 'configs/targets/sm_90_h800.json'),
        '--section-only', str(out / 'sm90_native_constants.json'),
    ], cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
