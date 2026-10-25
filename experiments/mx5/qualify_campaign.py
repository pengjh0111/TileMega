#!/usr/bin/env python3
"""Audit exact inventory coverage and qualify completed MX-5 campaign batches.

The complete domain is scoped to the explicitly requested thread count. It
does not approve a global DT=576 resource policy or change target defaults.
--check-only audits immutable manifests without requiring timing completion.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
KINDS = ('gemm', 'head', 'attention')


def sibling(name):
    spec = importlib.util.spec_from_file_location('mx5_qualifier_' + name,
                                                  Path(__file__).with_name(name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


table = sibling('stage_table')
scheduler = sibling('schedule_campaign')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


class Inputs:
    def __init__(self):
        self.records = {}

    def read(self, path):
        path = Path(path).resolve()
        raw = path.read_bytes()
        record = dict(path=str(path), sha256=sha(raw))
        old = self.records.setdefault(str(path), record)
        if old != record:
            raise ValueError('input changed while qualifying: ' + str(path))
        return raw

    def document(self, path):
        return json.loads(self.read(path))

    def rows(self, path):
        return [json.loads(line) for line in self.read(path).decode().splitlines() if line.strip()]


def canonical(kind, key, inventory=False):
    key = dict(key)
    field = 'prefetch_kind' if kind == 'attention' else 'prefetch'
    if inventory and field not in key and 'pw' not in key:
        key[field] = 0
    return table.canonical(dict(key=key), kind)


def base_id(kind, key):
    field = 'prefetch_kind' if kind == 'attention' else 'prefetch'
    return kind, tuple((name, key[name]) for name in table.FIELDS[kind] if name != field)


def uses(row):
    values = row.get('uses')
    if not isinstance(values, list) or not values:
        raise ValueError('inventory/checked row lacks model-stage uses')
    if any(not isinstance(item, dict) or set(item) != {'model', 'stage'}
           or any(not isinstance(value, str) or not value for value in item.values()) for item in values):
        raise ValueError('invalid model-stage uses')
    result = [json.dumps(item, sort_keys=True) for item in values]
    return tuple(sorted(set(result)))


def checked_identity(row):
    kind = row['kind']
    key = canonical(kind, row['key'])
    provenance = table.identity(row['provenance'])
    if table.kind_of(provenance) != kind:
        raise ValueError('checked kind/provenance mismatch')
    if (row.get('fits') is not True or row.get('query_only') is not True
            or row.get('query_status') != 0 or type(row.get('selected_resource_step')) is not int
            or row['selected_resource_step'] < 0):
        raise ValueError('checked row lacks a passing selected resource query')
    table.resources(row['resources'], key['threads'])
    ref_kind, ref_key, ref_provenance, _ = table.validate_reference(row['r_next_reference'])
    expected = dict(key)
    expected['prefetch_kind' if kind == 'attention' else 'prefetch'] = 0
    if ref_kind != kind or ref_key != expected:
        raise ValueError('checked R_next reference coordinates mismatch')
    if any(ref_provenance[field] != provenance[field] for field in
           ('executor', 'implementation', 'source_sha256')):
        raise ValueError('checked R_next source identity mismatch')
    return kind, key


def audit(args, inputs):
    inventory = {}
    for row in inputs.rows(args.inventory):
        kind = row['kind']
        if kind not in KINDS:
            raise ValueError('unknown inventory kind')
        key = canonical(kind, row['key'], inventory=True)
        if key['threads'] != args.threads:
            continue
        index = base_id(kind, key)
        if index in inventory:
            raise ValueError('duplicate inventory key')
        inventory[index] = uses(row)
    if {index[0] for index in inventory} != set(KINDS):
        raise ValueError('requested inventory must contain gemm, head, and attention')
    jobs, seen, manifests = [], {}, set()
    for path in args.manifest:
        path = path.resolve()
        if path in manifests:
            raise ValueError('duplicate campaign manifest')
        manifests.add(path)
        document = inputs.document(path)
        if document.get('processes_per_key') != 3:
            raise ValueError('campaign must specify three processes per key')
        source_path = Path(document['checked'])
        if sha(inputs.read(source_path)) != document['checked_sha256']:
            raise ValueError('campaign checked source changed')
        source = {}
        for row in inputs.rows(source_path):
            kind, key = row['kind'], canonical(row['kind'], row['key'])
            index = table.key_id(kind, key)
            if index in source:
                raise ValueError('duplicate checked source key')
            source[index] = row
        count = 0
        for number, job in enumerate(document['jobs']):
            kind = job['kind']
            if kind not in KINDS:
                raise ValueError('unknown campaign kind')
            keys_path = Path(job['keys'])
            if sha(inputs.read(keys_path)) != job['keys_sha256']:
                raise ValueError('campaign keys changed')
            keys = inputs.document(keys_path)
            directory = Path(job['directory'])
            checked = inputs.rows(directory / 'checked.jsonl')
            references = inputs.document(directory / 'references.json')['references']
            if len(keys) != job['rows'] or len(checked) != len(keys) or not keys:
                raise ValueError('batch keys/checked count mismatch')
            if references != [row['r_next_reference'] for row in checked]:
                raise ValueError('batch references differ from checked source rows')
            for key_row, row in zip(keys, checked):
                checked_kind, checked_key = checked_identity(row)
                reference = row['r_next_reference']
                if reference.get('compiler_probe_only'):
                    for name in ('binary','build_metadata','source_archive'):
                        descriptor = reference['probe'][name]
                        if sha(inputs.read(descriptor['path'])) != descriptor['sha256']:
                            raise ValueError('compiler probe proof changed during qualification: ' + descriptor['path'])
                key = canonical(kind, key_row)
                if (checked_kind != kind or checked_key != key
                        or key_row.get('library') != row.get('library')):
                    raise ValueError('job key differs from checked source row')
                if source.get(table.key_id(kind, key)) != row:
                    raise ValueError('batch checked row differs from immutable campaign checked source')
                index = base_id(kind, key)
                if index in seen:
                    raise ValueError('duplicate campaign base key')
                seen[index] = dict(manifest=str(path), job=number)
                if index in inventory and uses(row) != inventory[index]:
                    raise ValueError('checked model-stage uses differ from inventory')
            count += len(keys)
            jobs.append(dict(manifest=str(path), index=number, job=job, checked=checked))
        if count != document['rows']:
            raise ValueError('campaign row count mismatch')
    missing = set(inventory) - set(seen)
    extra = set(seen) - set(inventory)
    coverage = dict(expected_keys=len(inventory), provided_keys=len(seen),
                    missing_keys=len(missing), extra_keys=len(extra),
                    expected_by_kind={kind: sum(index[0] == kind for index in inventory) for kind in KINDS},
                    provided_by_kind={kind: sum(index[0] == kind for index in seen) for kind in KINDS},
                    missing_examples=[repr(index) for index in sorted(missing)[:3]],
                    extra_examples=[repr(index) for index in sorted(extra)[:3]],
                    exact=not missing and not extra)
    return jobs, coverage


def card_for(job, inputs, job_log):
    directory = Path(job['directory'])
    marker = directory / 'scheduler_job.json'
    if marker.exists():
        record = inputs.document(marker)
        card = record.get('card')
        command = record.get('command', [])
        if (type(card) is not int or card not in (4, 5, 6, 7) or len(command) < 4
                or command[1] != str(card) or Path(command[3]).resolve() != (directory / 'guard').resolve()
                or record.get('keys_sha256') != job['keys_sha256']):
            raise ValueError('invalid scheduler card provenance')
        return str(card), dict(source=str(marker.resolve()), sha256=sha(inputs.read(marker)))
    path = job_log.resolve()
    raw = path.read_bytes()
    matches = []
    guard = (directory / 'guard').resolve()
    for number, line in enumerate(raw.decode().splitlines(), 1):
        match = re.fullmatch(r'\S+ gpu=(\d+) (timeout=\d+ out=\S+ cmd=.*|rc=-?\d+ out=\S+)', line)
        if not match:
            continue
        fields = match[2].split(' cmd=', 1)[0].split()
        phase = fields[0]
        out = Path(fields[1].removeprefix('out='))
        if not out.is_absolute():
            out = ROOT / out
        if out.resolve() == guard:
            matches.append(dict(line=number, text=line, card=match[1], phase=phase))
    if (len(matches) != 2 or not matches[0]['phase'].startswith('timeout=')
            or matches[1]['phase'] != 'rc=0' or matches[0]['card'] != matches[1]['card']
            or matches[0]['card'] not in ('4', '5', '6', '7')):
        raise ValueError('missing or ambiguous successful manual guard card provenance: ' + str(guard))
    return matches[0]['card'], dict(source=str(path), snapshot_sha256=sha(raw), matched_lines=matches)


def qualify(args):
    inputs = Inputs()
    snapshots = []
    for path in (Path(__file__), Path(table.__file__), Path(table.compiler_probe.__file__), Path(scheduler.__file__)):
        raw = path.read_bytes()
        snapshots.append(dict(path=str(path.resolve()), sha256=sha(raw), source=raw.decode()))
    jobs, coverage = audit(args, inputs)
    scope = dict(coverage_threads=[args.threads], full_table=False,
                 global_dt576_policy_approved=False, target_defaults_changed=False,
                 actual_plan_resource_exceptions_approved=False)
    if args.check_only:
        return dict(schema_version=2, table_kind='mx5_campaign_coverage_audit', qualified=False,
                    qualification_scope='manifest_coverage_only', measured_domain_complete=False,
                    coverage=coverage, **scope,
                    provenance=dict(domain=dict(path=str(args.inventory.resolve()),
                                                sha256=sha(inputs.read(args.inventory))),
                                    manifests=[dict(path=str(path.resolve()), sha256=sha(inputs.read(path)))
                                               for path in args.manifest], tool_sources=snapshots,
                                    artifacts=list(inputs.records.values()))), int(not coverage['exact'])
    if not coverage['exact']:
        raise ValueError('campaigns do not exactly cover requested inventory: '
                         f"missing={coverage['missing_keys']} extra={coverage['extra_keys']}")
    rows, batches = [], []
    for batch in jobs:
        job = batch['job']
        if scheduler.batch_state(job) != 'completed':
            raise ValueError('campaign batch is incomplete: ' + job['directory'])
        directory = Path(job['directory'])
        card, card_evidence = card_for(job, inputs, args.job_log)
        runs = [directory / f'round{number}.jsonl' for number in (1, 2, 3)]
        result = table.aggregate(runs, [directory / 'references.json'], card='C550 GPU ' + card)
        for record in result['provenance']['inputs'] + result['provenance']['resource_inputs']:
            if sha(inputs.read(record['path'])) != record['sha256']:
                raise ValueError('aggregation input changed while qualifying: ' + record['path'])
        expected = {table.key_id(job['kind'], canonical(job['kind'], row['key'])): row
                    for row in batch['checked']}
        if len(result['rows']) != len(expected):
            raise ValueError('aggregated batch count differs from checked source')
        for row in result['rows']:
            checked = expected[table.key_id(row['kind'], row['key'])]
            if row['provenance'] != checked['provenance'] or row['resources'] != checked['resources']:
                raise ValueError('measurement provenance/resources differ from checked source')
            row['uses'] = checked['uses']
            # The solver must retain evidence for prescribed prefetch reductions.
            row['resource_selection'] = {field: checked[field] for field in
                ('selected_resource_step', 'resource_attempts') if field in checked}
            row['card'] = card
            row['campaign_batch'] = dict(manifest=batch['manifest'], job=batch['index'])
        rows.extend(result['rows'])
        files = [*runs, *(path.with_suffix('.log') for path in runs),
                 *(directory / f'process{number}.json' for number in (1, 2, 3)),
                 directory / 'guard/guard_result.json', directory / 'guard/guard.jsonl']
        for path in files:
            inputs.read(path)
        batches.append(dict(manifest=batch['manifest'], job=batch['index'], card=card,
                            card_evidence=card_evidence, aggregation=result['provenance']))
    # Re-read immutable evidence before publishing; the shared append-only job
    # log is instead retained as matched lines and a hash of its read snapshot.
    for record in list(inputs.records.values()):
        inputs.read(record['path'])
    manifests = [dict(path=str(path.resolve()), sha256=sha(inputs.read(path))) for path in args.manifest]
    return dict(schema_version=2, table_kind='mx5_decode_stage_costs', qualified=True,
                qualification_scope='complete_inventory_for_requested_threads',
                measured_domain_complete=True, coverage=coverage, diagnostic_reasons=[], **scope,
                provenance=dict(domain=dict(path=str(args.inventory.resolve()), sha256=sha(inputs.read(args.inventory))),
                                manifests=manifests, artifacts=list(inputs.records.values()),
                                batches=batches, tool_sources=snapshots), rows=rows), 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--threads', type=int, choices=(512, 576), required=True)
    parser.add_argument('--job-log', type=Path, default=ROOT / 'runs/mx5/gpu_jobs.log')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('output already exists; preserve historical qualification')
    try:
        document, code = qualify(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open('x') as stream:
            json.dump(document, stream, indent=2, allow_nan=False)
            stream.write('\n')
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    print(json.dumps(dict(qualified=document['qualified'], full_table=document['full_table'],
                          coverage_threads=document['coverage_threads'], coverage=document['coverage'])))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
