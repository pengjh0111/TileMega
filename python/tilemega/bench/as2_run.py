"""AS-2 entry point for the shared, no-lock single-device job runner."""
import json
import hashlib
from pathlib import Path
import struct
import sys
from tilemega.bench.as1_run import main, process_identity


def model_topology_preflight(command, physical, repo=None):
    """Check our model's admission evidence; this does not reserve a device."""
    if not command or Path(command[0]).name != 'as2_llm':
        return dict(status='not_applicable')
    repo = Path(repo or '/home/d50064638/junhuip/TileMega')
    runs = repo/'runs/as2'
    def controlled(raw, root):
        path = Path(raw).absolute()
        if path.resolve() != path or not path.is_relative_to(root) or path == root:
            raise ValueError('Model admission path escapes the controlled project')
        return path
    def digest(path):
        with path.open('rb') as stream:
            return hashlib.file_digest(stream, 'sha256').hexdigest()
    try:
        if len(command) < 3:
            raise ValueError('Missing model plan argument')
        declared_groups = None
        if command[1] in ('request', 'bench', 'teacher'):
            binding = controlled(command[2], runs)
            receipt = json.loads((binding/'request.json').read_text())
            declared_groups = receipt['groups']
            plans = [(binding/phase, controlled(receipt['phases'][phase]['source_plan'], runs))
                     for phase in ('prefill', 'decode')]
        else:
            plan = controlled(command[1], runs)
            plans = [(plan, plan)]
        groups = set()
        targets = []
        for bound, source in plans:
            metadata = (bound/'metadata.bin').read_bytes()
            if len(metadata) != 72:
                raise ValueError('Unknown model metadata extent')
            words = struct.unpack('<12I3Q', metadata)
            group = words[1]
            if words[0] not in (2, 3, 4) or group not in (2, 4, 8, 14) or words[2] != 3*group:
                raise ValueError('Unadmitted model topology/ABI')
            groups.add(group)
            if group != 14:
                continue
            origin = json.loads((source.parent/'import.json').read_text())
            target_path = controlled(origin['target_config'], repo/'configs/targets')
            if digest(target_path) != origin['target_config_sha256']:
                raise ValueError('Model target changed after compilation')
            target = json.loads(target_path.read_text())
            if (target.get('arch_tag') != 'ascend_3510' or target.get('product_sku') != '950PR' or
                    target.get('max_verified_mix_groups', 0) < group):
                raise ValueError('Model target does not admit this architecture/group count')
            if physical not in target.get('verified_physical_device_ids', []):
                raise ValueError(f'G14 target has no physical-device{physical} admission')
            admission = target['topology_experiment']
            raw = Path(admission['source_summary'])
            proof_path = controlled(raw if raw.is_absolute() else repo/raw, runs)
            if digest(proof_path) != admission['source_summary_sha256']:
                raise ValueError('Changed G14 protocol evidence')
            proof = json.loads(proof_path.read_text())
            if (proof.get('status') != 'pass' or proof.get('physical_device_id') != physical or
                    proof.get('processes_passed_per_group', 0) < 5 or 14 not in proof.get('groups', [])):
                raise ValueError('G14 requires matching five-process physical-card evidence')
            targets.append(str(target_path))
        if len(groups) != 1:
            raise ValueError('Request phases use different model topologies')
        if declared_groups is not None and declared_groups != next(iter(groups)):
            raise ValueError('Request identity differs from its binary topology')
        return dict(status='checked', groups=next(iter(groups)), physical_device_id=physical,
                    target_configs=sorted(set(targets)))
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        return dict(status='blocked', reason=str(exc), physical_device_id=physical)


def unfinished_owned_jobs():
    """Observe our receipts only; this is not a device/PID lock or mutex.

    The caller must still await each submission. This read-only preflight
    catches an accidentally unawaited previous tool session without inspecting
    or restricting unrelated processes, including jobs on the same account.
    """
    root=Path('/home/d50064638/junhuip/TileMega/runs/as2/jobs')
    terminal={'passed','failed','timeout','interrupted','launch_error','blocked-resource','blocked-health','blocked-capability','telemetry_unavailable'}
    active=[]
    for path in root.glob('*/job.json'):
        receipt=json.loads(path.read_text())
        if receipt.get('project')!='as2' or receipt.get('status') in terminal or not receipt.get('pid'):
            continue
        try:actual=process_identity(int(receipt['pid']))
        except (FileNotFoundError,ProcessLookupError):continue
        if all(actual[k]==receipt.get(k) for k in ('pid','start_ticks','process_group','session')):
            active.append(dict(job=str(path.parent),**actual))
    return active

if __name__ == '__main__':
    active=unfinished_owned_jobs()
    if active:
        print(json.dumps(dict(status='own_submission_still_running',jobs=active,
            device_locked=False,reason='Await our previous submission; external device users are not checked')),file=sys.stderr)
        raise SystemExit(75)
    raise SystemExit(main(project='as2', command_preflight=model_topology_preflight))
