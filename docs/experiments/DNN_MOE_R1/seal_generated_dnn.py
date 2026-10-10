#!/usr/bin/env python3
"""Seal CG-generated shared-library correctness and resource evidence."""
import argparse
import hashlib
import json
from pathlib import Path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--failures', nargs='*', default=[])
    parser.add_argument('--events', type=Path)
    parser.add_argument('--checks-events', type=Path,
                        help='completed fresh-process replay, retaining original failures')
    args = parser.parse_args()
    root=args.root.resolve();events=args.events.resolve() if args.events else root/'queue-results'
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    completion=json.loads((events/'completion.json').read_text())
    state=json.loads((events/'state.json').read_text())
    checks_events=args.checks_events.resolve() if args.checks_events else events
    checks_completion=json.loads((checks_events/'completion.json').read_text());assert checks_completion['passed']
    checks_state=json.loads((checks_events/'state.json').read_text())
    checks_preparation=None
    if args.checks_events:
        assert all(value['status']=='done' for name,value in state.items() if name.startswith('build_'))
        assert len([name for name in checks_state if name.startswith('check_')])==50
        assert set(checks_state)-{f'check_{i:02}' for i in range(50)}<= {'memcheck','racecheck'}
        path=checks_events.parent/'preparation.json'
        if path.exists():
            checks_preparation=json.loads(path.read_text())
            for path,digest in checks_preparation['inputs'].items():assert sha(path)==digest,path
    else:
        assert completion['passed']
    preparation=json.loads((root/'preparation.json').read_text())
    for path,digest in preparation['inputs'].items():assert sha(path)==digest,path
    cases=[]
    for process in range(50):
        name=f'check_{process:02}';assert checks_state[name]['status']=='done'
        path=checks_events/(name+'.log')
        result=[json.loads(line) for line in path.read_text().splitlines() if line.startswith('{')][-1]
        assert result['passed'] and len(result['cases'])==6
        cases.append(dict(process=process,log_sha256=sha(path),result=result))
    artifacts=[]
    for arch in [80,89,90,100,120]:
        path=root/f'generated-sm_{arch}.so.identity.json';identity=json.loads(path.read_text())
        if arch==89 and checks_preparation:
            assert checks_preparation['artifact_id']==identity['artifact_id']
        assert sha(root/f'generated-sm_{arch}.so')==identity['binary_sha256']
        assert sha(root/'generated.cu')==identity['cu_sha256']
        assert sha(identity['complete_macros']['path'])==identity['complete_macros']['sha256']
        artifacts.append(dict(identity=str(path),identity_sha256=sha(path),artifact_id=identity['artifact_id'],
            architecture=identity['target_arch'],resources=identity['resources']))
    sanitizers={}
    for tool,marker in [('memcheck','ERROR SUMMARY: 0 errors'),
        ('racecheck','RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')]:
        tool_events=checks_events if tool in checks_state else events
        path=tool_events/(tool+'.log')
        assert (checks_state if tool in checks_state else state)[tool]['status']=='done' and marker in path.read_text()
        sanitizers[tool]=dict(passed=True,log=str(path),log_sha256=sha(path))
    result=dict(evidence='verified',passed=True,scope=cases[0]['result']['scope'],
        preparation=preparation,completion=completion,fresh_processes=dict(passed=50,attempts=50),
        checks_completion=checks_completion,checks_events=str(checks_events),
        checks_preparation=checks_preparation,
        original_failed_steps={name:value for name,value in state.items() if value['status']!='done'},
        cases=cases,artifacts=artifacts,sanitizers=sanitizers,failures_retained=args.failures,
        artifact_spill_count=sum(any(r['spill'] for r in a['resources'].values()) for a in artifacts),
        artifact_stack_count=sum(any(r['stack_bytes'] for r in a['resources'].values()) for a in artifacts))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,fresh_processes=50,artifacts=5,
        spills=result['artifact_spill_count'],stack=result['artifact_stack_count'])),flush=True)
