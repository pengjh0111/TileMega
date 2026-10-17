"""Check a proposed queue against existing definitions and scheduler receipts."""
import argparse
import json
from pathlib import Path
import re


UNSUCCESSFUL = {'failed', 'skipped', 'not_run', 'cancelled'}


def check(existing, proposed, state):
    """Reject invalid graphs and required dependencies that cannot succeed.

    Completed receipts remain authoritative: a successful recovery is a new
    step, never a rewrite of an earlier failure. after_any only needs closure.
    """
    steps, errors = {}, []
    selected = {step['name'] for step in proposed}
    for step in [*existing, *proposed]:
        name = step['name']
        if name in steps:
            errors.append(dict(step=name, reason='duplicate definition'))
        steps[name] = step
    for name in selected:
        if name in state:
            errors.append(dict(step=name, reason='receipt already exists; use a new recovery name'))
        step = steps[name]
        argv = [str(value) for value in step.get('command', [])]
        scripts = {Path(value).name for value in argv}
        if scripts & {'validate_r16_numeric_gate.py', 'validate_r16_dnn_numeric_gate.py',
                      'validate_shared_numeric.py'}:
            try:
                target = argv[argv.index('--job')+1]
            except (ValueError, IndexError):
                target = None
            # A recovery must validate the GPU receipt it actually depends on,
            # rather than accidentally reading the failed pre-recovery name.
            if target not in step.get('after', []):
                errors.append(dict(step=name, reason='numeric validator target is not a required dependency', target=target))
        if 'run_r16_with_device_identity.py' in scripts:
            try:
                pci = argv[argv.index('--expected-pci')+1]
            except (ValueError, IndexError):
                pci = ''
            if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', pci, re.I):
                errors.append(dict(step=name, reason='CUDA device wrapper requires normalized four-digit PCI domain', pci=pci))
    visiting, visited = set(), set()

    def graph(name, trail):
        if name in visiting:
            errors.append(dict(step=trail[0], reason='dependency cycle', path=[*trail, name]))
            return
        if name in visited or name not in steps:
            return
        visiting.add(name)
        step = steps[name]
        for dep in [*step.get('after', []), *step.get('after_any', [])]:
            if dep not in steps and dep not in state:
                errors.append(dict(step=name, reason='unknown dependency', path=[name, dep]))
            graph(dep, [*trail, name])
        visiting.remove(name)
        visited.add(name)

    for name in sorted(selected):
        graph(name, [])

    def blocker(name, trail):
        if name in trail:
            return None  # Cycles are reported by the graph check.
        status = state.get(name, {}).get('status')
        if status in UNSUCCESSFUL:
            return [*trail, name]
        if status == 'done':
            return None
        for dep in steps.get(name, {}).get('after', []):
            path = blocker(dep, [*trail, name])
            if path:
                return path
        return None

    for name in sorted(selected):
        path = blocker(name, [])
        if path:
            errors.append(dict(step=name, reason='unsuccessful required dependency', path=path,
                               status=state[path[-1]]['status']))
    return dict(pass_check=not errors, proposed_steps=len(proposed), errors=errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--queue-dir', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--proposed', required=True, type=Path)
    parser.add_argument('--out', type=Path)
    args = parser.parse_args(argv)
    existing = []
    for path in sorted(args.queue_dir.glob('queue_*.json')):
        existing.extend(json.loads(path.read_text()))
    state = json.loads(args.state.read_text()) if args.state.exists() else {}
    result = check(existing, json.loads(args.proposed.read_text()), state)
    payload = json.dumps(result, indent=2) + '\n'
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload)
    print(payload, end='')
    return 0 if result['pass_check'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
