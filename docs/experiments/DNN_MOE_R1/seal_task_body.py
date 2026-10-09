#!/usr/bin/env python3
"""Seal a completed immutable TaskBody queue, including every resource row."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--events', default='events')
    args = parser.parse_args()
    root = args.root.resolve()
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    events=root/args.events
    completion = json.loads((events/'completion.json').read_text())
    assert completion['passed']
    state = json.loads((events/'state.json').read_text())
    prepared = json.loads((root/'preparation.json').read_text())
    for path, value in prepared['inputs'].items():
        assert sha(path)==value, path
    checks = [name for name in state if name.startswith('check_')]
    assert len(checks)==50 and all(state[name]['status']=='done' for name in checks)
    artifacts = []
    for arch in [80, 89, 90, 100, 120]:
        path = root/f'task-sm_{arch}.identity.json'
        identity = json.loads(path.read_text())
        artifacts.append(dict(identity=str(path), identity_sha256=sha(path),
            artifact_id=identity['artifact_id'], target_arch=identity['target_arch'],
            resources=identity['resources']))
    tools = {}
    for tool, marker in [('memcheck', 'ERROR SUMMARY: 0 errors'),
                         ('racecheck', 'RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)')]:
        path = events/(tool+'.log')
        assert marker in path.read_text()
        tools[tool] = dict(passed=True, log=str(path), sha256=sha(path))
    result = dict(evidence='verified', passed=True, scope=prepared['scope'],
        preparation=prepared, completion=completion, fresh_processes=dict(passed=50, attempts=50),
        checks={name: sha(events/(name+'.log')) for name in checks}, artifacts=artifacts,
        sanitizers=tools, artifact_spill_count=sum(any(r['spill'] for r in a['resources'].values()) for a in artifacts),
        artifact_stack_count=sum(any(r['stack_bytes'] for r in a['resources'].values()) for a in artifacts),
        sample_check=(events/'check_00.log').read_text(),
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    args.out.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(dict(passed=True, fresh_processes=50,
                         artifacts=len(artifacts), spills=result['artifact_spill_count'])), flush=True)


if __name__ == '__main__':
    main()
