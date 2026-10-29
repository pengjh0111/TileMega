"""Verify native MoE cost evidence without granting geometry or model acceptance."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess


def need(condition, message):
    if not condition:
        raise ValueError('MoE cost evidence: ' + message)


class Files:
    def __init__(self):
        self.verified = {}

    def check(self, ref):
        need(isinstance(ref, dict) and isinstance(ref.get('path'), str)
             and isinstance(ref.get('sha256'), str), 'file reference is missing')
        path = Path(ref['path'])
        state = path.stat()
        stamp = (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns)
        old = self.verified.get(str(path))
        if old is None or old[0] != stamp:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            old = stamp, digest.hexdigest()
            self.verified[str(path)] = old
        need(old[1] == ref['sha256'], 'file hash changed: ' + str(path))
        return path

    def walk(self, value):
        if isinstance(value, dict):
            if 'path' in value and 'sha256' in value:
                self.check(value)
            for child in value.values():
                self.walk(child)
        elif isinstance(value, list):
            for child in value:
                self.walk(child)

    def unchanged(self):
        for path, (stamp, _) in self.verified.items():
            state = Path(path).stat()
            need(stamp == (state.st_dev, state.st_ino, state.st_size, state.st_mtime_ns),
                 'file changed during validation: ' + path)


def resources(sample):
    if 'resources' not in sample:
        return {k: sample[k] for k in ('mtreg', 'local_bytes', 'static_smem_bytes', 'resident_ctas')}
    value = sample['resources']
    return dict(mtreg=value['MTreg'], local_bytes=value['stack'],
        static_smem_bytes=value['static_smem'], resident_ctas=value['resident'])


def fitting(value, threads):
    return (0 < value['mtreg'] <= (227 if threads == 576 else 255)
        and value['local_bytes'] == value['static_smem_bytes'] == 0
        and value['resident_ctas'] == 1)


def validate_row(row, files=None):
    own_files = files is None
    files = files or Files()
    files.walk(row)
    key, provenance = row['key'], row['provenance']
    need(key['threads'] in (512, 576) and key['b'] in (1, 2, 4, 8, 16), 'unsupported request')
    need(fitting(row['resources'], key['threads']), 'measured row fails the resource gate')
    need(row['processes'] == 3 and [r['round'] for r in row['rounds']] == [1, 2, 3],
         'three ordered fresh-process rounds required')
    need(provenance['executor'] == 'mx_static_stage_bench', 'unexpected executor')
    medians, p90s, inputs, processes, pids = [], [], set(), set(), set()
    legacy_commands = 0
    for entry in row['rounds']:
        ref = entry['input']
        path = files.check(ref)
        need(str(path) not in inputs, 'a raw round file was reused')
        inputs.add(str(path))
        need(type(ref.get('line')) is int and ref['line'] > 0, 'raw row line is invalid')
        lines = path.read_text().splitlines()
        need(ref['line'] <= len(lines), 'raw row line is missing')
        sample = json.loads(lines[ref['line'] - 1])
        need(sample == entry['measurement'] and sample['round'] == entry['round'],
             'embedded sample differs from the original raw row')
        need(sample['executor'] == provenance['executor']
             and sample['implementation'] == provenance['implementation'], 'sample implementation differs')
        source = sample.get('source_sha256', sample.get('source_digest'))
        need(source == provenance['source_sha256'], 'sample source epoch differs')
        need(resources(sample) == row['resources'], 'actual sample resources differ')
        actual = sample['key']
        aliases = dict(depth='d', prefetch='pw')
        for name, value in key.items():
            present = actual.get(name, actual.get(aliases.get(name)))
            need(present == value, 'sample coordinate differs: ' + name)
        if row['kind'] == 'moe_router':
            dispatch = row['dispatch_context']
            need((actual['dispatch_gate_tn'], actual['dispatch_down_tn']) ==
                 (dispatch['gate_up_tn'], dispatch['down_tn']), 'router dispatch TNs differ')
        binary = actual.get('binary_sha256', sample.get('binary_sha256', provenance['binary_sha256']))
        need(binary == provenance['binary_sha256'], 'sample binary differs')
        files.check(dict(path=actual['library'], sha256=binary))
        working_set = sample.get('addressed_rotating_weight_bytes', sample.get('working_set_bytes', 0))
        need(working_set >= 256 * 1024**2, 'rotating working set is smaller than 256MiB')
        need(sample['iterations'] > 0, 'zero-launch query cannot be used as a cost')
        if row['kind'] in ('moe_gate_up', 'moe_down_combine', 'moe_router'):
            flags = ['source_bound', 'cost_protocol', 'repeat_bitwise']
            flags += (['FP32_reference_passed', 'legacy_topk_weights_bitwise', 'dispatch_verified']
                      if row['kind'] == 'moe_router' else ['reference_passed'])
            need(all(sample.get(flag) is True for flag in flags), 'body validation or cost protocol failed')
        for name in ('median_ns', 'p90_ns'):
            need(type(sample[name]) in (int, float) and math.isfinite(sample[name])
                 and sample[name] > 0, 'invalid sample time')
        if 'process' in entry:
            process_path = files.check(entry['process'])
            guard_path = files.check(entry['guard'])
            need(type(entry.get('PID')) is int and entry['PID'] > 0
                 and entry['PID'] not in pids, 'round PID is missing or reused')
            pids.add(entry['PID'])
            need(int((guard_path.parent / 'pgid').read_text()) == entry['PID'], 'guard PID differs')
        else:
            # The original head campaign recorded commands, not child PIDs.
            need(row['kind'] == 'head', 'non-head row lacks a bound process record')
            process_path = path.parent / ('process%d.json' % entry['round'])
            guard_path = path.parent / 'guard/guard_result.json'
            process = json.loads(process_path.read_text())
            need(process['round'] == entry['round'] and str(path) in process['command'],
                 'original head process command differs')
            legacy_commands += 1
        need(str(process_path) not in processes, 'round process record was reused')
        processes.add(str(process_path))
        need(json.loads(process_path.read_text())['returncode'] == 0
             and json.loads(guard_path.read_text())['code'] == 0, 'process or guard failed')
        medians.append(sample['median_ns'])
        p90s.append(sample['p90_ns'])
    median = statistics.median(medians)
    need(row['median_ns'] == median and row['p90_ns'] == statistics.median(p90s)
         and row['spread'] == round((max(medians) - min(medians)) / median, 4),
         'aggregate quantiles differ from the original rounds')
    if own_files:
        files.unchanged()
    return dict(rounds=3, actual_guard_pids=len(pids), legacy_head_command_records=legacy_commands)


def validate_outcomes(path):
    path = Path(path)
    data = path.read_bytes()
    doc = json.loads(data)
    kinds = {'mx5_moe_geometry_outcomes', 'mx5_moe_regular_geometry_outcomes',
        'mx5_moe_grouped_geometry_outcomes', 'mx5_moe_head_geometry_outcomes',
        'mx5_moe_router_context_geometry_outcomes'}
    need(doc.get('schema_version') == 1 and doc.get('table_kind') in kinds,
         'wrong geometry-outcome schema')
    need(not doc.get('synthetic_fixture'), 'synthetic costs are not measurement evidence')
    files = Files()
    files.walk(doc)
    legacy = 0
    for row in doc['rows']:
        legacy += validate_row(row, files)['legacy_head_command_records']
    for row in doc['rejections']:
        need(not any(k in row for k in ('median_ns', 'p90_ns', 'spread', 'rounds', 'processes')),
             'resource rejection contains a measured cost')
        need(not fitting(row['resources'], row['key']['threads']), 'rejection has fitting resources')
    need(path.read_bytes() == data, 'outcome table changed during validation')
    files.unchanged()
    return dict(schema_version=1, kind='mx5_moe_cost_file_evidence',
        table=dict(path=str(path.resolve()), sha256=hashlib.sha256(data).hexdigest()),
        measured=len(doc['rows']), rejected=len(doc['rejections']),
        verified_files=len(files.verified), legacy_head_command_records=legacy,
        geometry_completeness_implied=False, model_acceptance_implied=False,
        solver_selected=False, final=False, default_enabled=False)


def validate_complete_scope(path, domain_tool):
    path, domain_tool = Path(path), Path(domain_tool)
    result = validate_outcomes(path)
    doc = json.loads(path.read_text())
    need(doc.get('table_kind') == 'mx5_moe_geometry_outcomes'
         and doc.get('complete_scope') is True, 'complete per-request scope is required')
    need(all(doc.get(name, False) is False for name in
             ('execution_authorized', 'solver_selected', 'final', 'default_enabled')),
         'cost evidence cannot grant execution or model/default acceptance')
    need(doc.get('batch') in (1, 2, 4, 8, 16) and doc.get('threads') in (512, 576),
         'scope request is missing')
    need(all(row['key']['b'] == doc['batch'] and row['key']['threads'] == doc['threads']
             for row in doc['rows'] + doc['rejections']), 'scope contains another batch/thread request')
    files = Files()
    files.check(result['table'])
    closure_ref = doc['body_header_closure']
    closure = json.loads(files.check(closure_ref).read_text())
    families = closure['families']
    need(len(families) == 5 and {f['family'] for f in families} ==
         {'gemm', 'head', 'attention', 'grouped_v2', 'router_contexts'}, 'body dependency closure differs')
    files.walk(closure)
    for family in families:
        need(bool(family['headers']), 'body header dependency list is empty')
        for header in family['headers']:
            need(files.check(header['frozen']).read_bytes() == files.check(header['adopted']).read_bytes(),
                 'adopted body differs from the measured dependency closure')
    for row in doc['rows']:
        need(row['provenance']['source_sha256'] == doc['source_epochs'][row['kind']],
             'row source differs from the declared family epoch')
    digest = hashlib.sha256(domain_tool.read_bytes()).hexdigest()
    tool_ref = dict(path=str(domain_tool.resolve()), sha256=digest)
    files.check(tool_ref)
    command = [str(domain_tool.resolve()), '--audit-rows', str(path.resolve()),
        str(doc['batch']), str(doc['threads'])]
    process = subprocess.run(command, capture_output=True, text=True, check=False)
    need(process.returncode == 0, 'actual native geometry audit failed: ' + process.stderr)
    coverage = json.loads(process.stdout)
    need(coverage.get('complete_geometry') is True and coverage.get('missing') == []
         and coverage['measured'] == result['measured'] and coverage['rejected'] == result['rejected']
         and coverage['expected'] == result['measured'] + result['rejected'], 'geometry coverage differs')
    files.unchanged()
    result.update(kind='mx5_moe_complete_scope_file_evidence', batch=doc['batch'], threads=doc['threads'],
        body_header_closure=closure_ref, domain_tool=tool_ref, coverage=coverage,
        geometry_completeness_implied=True, model_acceptance_implied=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--domain-tool', type=Path,
        help='also require complete request geometry and the adopted body dependency closure')
    args = parser.parse_args()
    need(not args.out.exists(), 'existing evidence output is preserved')
    result = (validate_complete_scope(args.table, args.domain_tool)
              if args.domain_tool else validate_outcomes(args.table))
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write('\n')


if __name__ == '__main__':
    main()
