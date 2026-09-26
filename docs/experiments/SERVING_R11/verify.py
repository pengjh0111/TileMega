#!/usr/bin/env python3
"""R11 code-contract audit.  Missing runtime evidence is a failure, not a skip."""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / 'docs/experiments/SERVING_R11'
BASELINE = '4bf26fb85ada63fccea101b323f3d7a09351eca2'


def lines(path: str):
    file = ROOT / path
    if not file.exists():
        return []
    return [(i, line.strip()) for i, line in enumerate(file.read_text(errors='replace').splitlines(), 1)]


def hit(path: str, pattern: str):
    return [(path, n, line) for n, line in lines(path) if re.search(pattern, line)]


def absent(path: str, pattern: str):
    found = hit(path, pattern)
    return (not found, found[:1] or [(path, 0, f'absent: {pattern}')])


def require(path: str, *patterns: str):
    evidence = []
    for pattern in patterns:
        match = hit(path, pattern)
        if not match:
            return False, evidence + [(path, 0, f'missing: {pattern}')]
        evidence.append(match[0])
    return True, evidence


def combine(*checks):
    return all(check[0] for check in checks), [item for _, items in checks for item in items]


def evidence_json(path: str, predicate, description: str):
    file = EVIDENCE / path
    if not file.is_file():
        return False, [(str(file.relative_to(ROOT)), 0, 'missing: ' + description)]
    try:
        data = json.loads(file.read_text())
        good = bool(predicate(data))
    except (ValueError, KeyError, TypeError, OSError):
        good = False
    return good, [(str(file.relative_to(ROOT)), 1,
                   description + (' satisfied' if good else ' failed'))]


def ev2_contamination_policy():
    """K-15 checks the decisions, not just the existence of a policy file."""
    path = EVIDENCE / 'ev2/measurement_policy.json'
    label = str(path.relative_to(ROOT))
    if not path.is_file():
        return False, [(label, 0, 'missing predeclared EV-2 policy and round decisions')]
    try:
        data = json.loads(path.read_text())
        idle = float(data['idle_power_w'])
        margin = float(data['power_margin_w'])
        rounds = data['rounds']
        if idle <= 0 or margin <= 0 or not isinstance(rounds, list):
            raise ValueError('invalid idle power, margin, or rounds')
        expected = {(model, batch, arm) for model in ('llama', 'qwen3')
                    for batch in (1, 16) for arm in ('tilemega', 'vllm')}
        seen = set()
        for row in rounds:
            cell = (row['model'], int(row['batch']), row['arm'])
            if cell not in expected or cell in seen:
                raise ValueError(f'duplicate or unexpected EV-2 arm {cell}')
            seen.add(cell)
            observations = row['observations']
            if len(observations) < 3:
                raise ValueError(f'no per-round contamination decisions for {cell}')
            for observation in observations:
                power = float(observation['power_w'])
                threshold = float(observation['threshold_w'])
                if abs(threshold - (idle + margin)) > 1e-6:
                    raise ValueError(f'changed power threshold for {cell}')
                if not isinstance(observation['exclusive'], bool) or \
                        not isinstance(observation['accepted'], bool):
                    raise ValueError(f'missing exclusivity decision for {cell}')
                if observation['accepted'] and not (observation['exclusive'] and power <= threshold):
                    raise ValueError(f'accepted a contaminated round for {cell}')
            if sum(bool(x['accepted']) for x in observations) < 3:
                raise ValueError(f'fewer than three accepted timed runs for {cell}')
        if seen != expected:
            raise ValueError(f'missing EV-2 arms: {expected - seen}')
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, [(label, 1, f'EV-2 contamination evidence invalid: {error}')]
    return True, [(label, 1,
                   f'idle={idle} W margin={margin} W; {len(rounds)} arms with logged round decisions')]


def changed_sources():
    result = subprocess.run(['git', 'diff', '--name-only', BASELINE, '--', 'include', 'lib', 'tools', 'python'],
                            cwd=ROOT, text=True, check=True, capture_output=True)
    return [s for s in result.stdout.splitlines() if (ROOT / s).is_file()]


def fingerprints_equal():
    binary = Path(os.environ.get('TILEMEGA_BIN', '/root/r11_work/build/tools/tilemega'))
    if not binary.is_file():
        return False, [(str(binary), 0, 'missing built tilemega for fingerprint comparison')]
    from sys import path as import_path
    import_path.insert(0, str(ROOT / 'python'))
    from tilemega.fingerprint import source_fingerprint
    try:
        version = json.loads(subprocess.check_output([str(binary), 'version', '--json'], text=True))
        actual = source_fingerprint()
        expected = version['source_sha256']
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        return False, [(str(binary), 0, f'fingerprint check failed: {error}')]
    return expected == actual, [(str(binary), 1, f'binary={expected} Python={actual}')]


def no_arch_comparisons():
    offenders = []
    for path in changed_sources():
        if path.endswith('/ArchDispatch.h'):
            continue
        offenders += hit(path, r'__CUDA_ARCH__\s*(?:[<>!=]=?|\))|(?:[<>!=]=?)\s*__CUDA_ARCH__')
    return not offenders, offenders[:3] or [('include/tilemega/Target/ArchDispatch.h', 0,
                                              'no other modified source compares __CUDA_ARCH__')]


def k17():
    paths = list((ROOT / 'include/tilemega/Codegen/executor').glob('*.cuh'))
    paths += list((ROOT / 'include/tilemega/Codegen/tasks').glob('Serving*'))
    paths += [ROOT / 'include/tilemega/Backend/ServingEpilogue.h']
    bad = []
    for file in paths:
        if file.is_file():
            bad += hit(str(file.relative_to(ROOT)), r'\bblockDim\.x\b|__syncthreads\s*\(')
    return not bad, bad[:3] or [('include/tilemega/Codegen/executor/ComputeGroup.cuh', 0,
                                 'no blockDim.x or __syncthreads in decode serving sources')]


def checks():
    ex = 'include/tilemega/Codegen/executor/'
    ck = {}
    ck['K-1'] = combine(require('lib/Codegen/ServingPages.cpp', 'page_bytes', 'pages'),
                        require(ex + 'ServingPages.cuh', 'PageRing'),
                        absent(ex + 'ServingPages.cuh', r'\bTaskSmem\b'))
    loader = ex + 'ServingPages.cuh'
    ck['K-2'] = combine(require(loader, 'Loader'),
                        absent(ex + 'PageRing.cuh', r'WaitTaskDependencies|GradedWait|EventPoll'))
    ck['K-3'] = combine(require(ex + 'PageRing.cuh', r'SlotIndex\(sequence\)'),
                        require('tools/commands/compile.cpp', 'page_bytes', 'tmexec.pages'))
    ck['K-4'] = combine(require('tools/commands/compile.cpp', 'TILEMEGA_WAIT_POLICY',
                               'wait_spin_iters'), absent('lib/Codegen/Codegen.cpp', r'__nanosleep\(64\)'))
    ck['K-5'] = combine(require('include/tilemega/Dialect/CouplingGraph/ExecOps.td', 'HandoffOp'),
                        require('lib/Dialect/CouplingGraph/HandoffPass.cpp', 'VerifyHandoffAccess', 'ApplyHandoffs'),
                        require('include/tilemega/Codegen/tasks/ServingRMSNormTaskBody.h', 'RowInvRms', 'Transform'),
                        require('include/tilemega/Codegen/tasks/PagedGemmTaskBody.h', 'RowInvRms', 'Transform'),
                        require(ex + 'ServingPages.cuh', 'norm_input', 'norm_weight'))
    ck['K-6'] = combine(require(ex + 'LastArriver.cuh', 'ticket', 'fetch_add'),
                        require('include/tilemega/Codegen/tasks/LastArriverTaskBody.h',
                                'ServingGemmCombineTaskBody', 'AttentionMergeTaskBody'),
                        require(ex + 'ServingPages.cuh', 'LastArriverGemmTaskBody',
                                'LastArriverAttentionTaskBody'))
    ck['K-7'] = combine(require('include/tilemega/Codegen/tasks/FusedAttentionTaskBody.h', 'warp'),
                        absent('include/tilemega/Codegen/tasks/FusedAttentionTaskBody.h', r'__syncthreads\s*\('))
    ck['K-8'] = combine(no_arch_comparisons(),
                        require(ex + 'Async.cuh', 'kMbarrierTryWait', 'kMbarrierTx'),
                        require(ex + 'TensorMap.h', 'cudaGetDriverEntryPoint'))
    ck['K-9'] = combine(require(ex + 'Async.cuh', 'griddepcontrol.wait', 'griddepcontrol.launch_dependents'),
                        require(ex + 'ServingLaunch.cuh', 'cudaLaunchAttributeProgrammaticStreamSerialization'),
                        require(ex + 'ServingPages.cuh', 'GridDependency<PageArch>::Wait'))
    ck['K-10'] = combine(require(ex + 'ServingPrefetch.cuh', 'PrefetchRanges'),
                         require('lib/Codegen/ServingPages.cpp', 'written', 'Subtract'))
    ck['K-11'] = combine(require('lib/Solver/StageFlowModel.cpp', 'prefetch', 'page_bytes'),
                         require('lib/Solver/SkeletonSearch.cpp', 'handoff', 'page_bytes', 'TIE_GROUP'))
    ck['K-12'] = combine(require('CMakeLists.txt', 'TILEMEGA_BUILD_EXPERIMENTAL', 'tilemega-unit', '-UNDEBUG'),
                         absent('python/tilemega/compile.py', 'NotImplementedError'),
                         absent('python/tilemega/serve.py', 'NotImplementedError'))
    ck['K-13'] = combine(require('tools/tilemega.cpp', 'source_sha256'),
                         require('python/tilemega/fingerprint.py', 'source_fingerprint'),
                         fingerprints_equal())
    ck['K-14'] = combine(require('python/tilemega/cli.py', 'calibration_sections', 'source_sha256',
                                'features', 'plan_key', 'tilemega.serving'))
    ck['K-15'] = ev2_contamination_policy()
    ck['K-16'] = combine(absent('tools/commands/compile.cpp', r'TILEMEGA_MIDPOINT_REFINE=1'),
                         evidence_json('ev2/sass_audit.json', lambda d: d.get('all_fp64_zero') is True,
                                       'all serving SASS FP64 counts zero'))
    ck['K-17'] = k17()
    ck['K-18'] = combine(absent('CMakeLists.txt', r'CUDA_ARCHITECTURES\s+89\b'),
                         require('CMakeLists.txt', 'sm_100'),
                         require('python/tilemega/cli.py', "arch_tag", 'configs/targets'))
    return ck


def main():
    results = checks()
    for number, (good, sources) in results.items():
        rendered = '; '.join(f'{path}:{line}: {snippet[:160]}' for path, line, snippet in sources)
        print(f'{number} {"PASS" if good else "FAIL"} {rendered}')
    passed = sum(ok for ok, _ in results.values())
    print(f'R11 code contract: {passed}/{len(results)} PASS')
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == '__main__':
    main()
