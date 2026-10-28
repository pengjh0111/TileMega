"""Check normal WD0/WD1 resource pairs before the MX5 §4.5 retry branch.

This tool never compiles, launches, authorizes 512 threads or accepts a result.
Invoke with PYTHONPATH pointing to the immutable producer's Python tree.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path

from tilemega import mx5_plan_receipt as receipts
from tilemega import mx5_plan_resources as policy
from tilemega import mx5_resolved_top4 as resolved
from tilemega.mx5_select import check_file, record


def require(value, reason):
    if not value:
        raise ValueError(reason)


def proposed_width(pin, used):
    """The existing deterministic branch, shared by the two instrumentations."""
    require(type(used) is int and 0 <= used <= 3, 'invalid shared reduction count')
    c = pin['mx_decode_v2']
    require(c['threads'] == 576, 'watchdog pairing cannot authorize 512')
    require(all(type(c[n]['prefetch']) is int and c[n]['prefetch'] >= c[n]['d']
                for n in policy.CLASSES), 'invalid width/floor')
    eligible = [n for n in policy.CLASSES if c[n]['prefetch'] > c[n]['d']]
    if not eligible or used == 3:
        return None
    name = min(eligible, key=lambda n: (-c[n]['prefetch'], policy.CLASSES.index(n)))
    following = copy.deepcopy(pin)
    previous = c[name]['prefetch']
    width = max(c[name]['d'], previous // 2)
    following['mx_decode_v2'][name]['prefetch'] = width
    return dict(shared_class=name, from_pw=previous, to_pw=width,
                reduction_step=used + 1, next_pin=following)


def _normal(receipt_path, resources_path, origin, candidate, pin, watchdog, sources):
    native_ref, native = receipts.load_native(receipt_path)
    require(native['returncode'] == 0, 'failed device compilation is not a resource observation')
    files = dict(native['files'], compile_receipt=native_ref)
    require(files['producer'] == origin.data['preview']['producer'], 'producer changed')
    require(json.loads(check_file(files['pin']).read_text()) == pin, 'requested pin changed')
    actual = json.loads(check_file(files['resolved_pin']).read_text())['mx_decode_v2']
    for n in policy.CLASSES:
        for field in ('tn', 'wn', 'd', 'part', 'prefetch', 'r_next'):
            require(type(actual[n][field]) is type(pin['mx_decode_v2'][n][field])
                    and actual[n][field] == pin['mx_decode_v2'][n][field], 'resolved coordinate changed')
    for field in ('threads', 'b_min', 'attn_prefetch'):
        require(actual[field] == pin['mx_decode_v2'][field], 'resolved thread/attention changed')
    snapshot = json.loads(check_file(files['source_snapshot']).read_text())
    policy._snapshot_inputs(snapshot)
    resolved.exception._normal_options(native, files, snapshot, watchdog)
    policy._generated_config(check_file(files['generated_source']).read_text(), origin, candidate, pin)
    checked_sources = set()
    for rows in origin.candidate(candidate)['evidence'].values():
        for row in rows:
            source = row['provenance']['source_sha256']
            if source not in checked_sources:
                policy._verify_archive(row, sources, snapshot['source_files'])
                checked_sources.add(source)
    resource_ref = record(resources_path)
    resource = policy.inspect(check_file(files['binary']), check_file(files['log']), 576)
    require(resource == json.loads(check_file(resource_ref).read_text()), 'resource report changed')
    k = resource['kernel']
    require(k['static_smem_bytes'] == 0 and not k.get('callees'), 'static smem/callee failure')
    mir = resolved.exception._mir(files['log'], k['name'])
    require(mir['rings'] > 0 and not any(mir[n] for n in
            ('addrspace0', 'generic', 'calls', 'ring4160', 'ring_generic')), 'MIR hot-loop gate failed')
    failed = k['registers'] > 227 or k['stack_bytes'] > 0
    require(failed or mir['private'] == 0, 'fitting program has private accesses')
    return dict(native=native_ref, files=files, resources=resource_ref,
                kernel=k, MIR=mir, failed_resources=failed, snapshot=snapshot)


def inspect_pair(preview, candidate, pin, wd0, wd1, sources, used_reductions=0):
    origin = resolved.Qualified576Origin.capture(record(preview))
    require(type(candidate) is int and 0 <= candidate < 4, 'invalid original candidate ID')
    require(pin['mx_decode_v2']['threads'] == 576, 'only original576 resource pairs are supported')
    policy._same_geometry(origin.candidate(candidate)['pin'], pin)
    a = _normal(wd0['native'], wd0['resources'], origin, candidate, pin, 0, sources)
    b = _normal(wd1['native'], wd1['resources'], origin, candidate, pin, 1, sources)
    require({k:v for k,v in a['snapshot'].items() if k != 'patch'} ==
            {k:v for k,v in b['snapshot'].items() if k != 'patch'}
            and a['files']['source_patch']['sha256'] == b['files']['source_patch']['sha256'],
            'paired source snapshots differ')
    require(a['files']['generated_source']['sha256'] == b['files']['generated_source']['sha256'],
            'paired generated code differs')
    failed = a['failed_resources'] or b['failed_resources']
    proposal = proposed_width(pin, used_reductions) if failed else None
    return dict(kind='mx5_watchdog_pair_resource_observation', schema_version=1,
                origin_preview=record(preview), candidate_id=candidate, pin=pin,
                WD0=a, WD1=b, failed_resources=failed, used_reductions=used_reductions,
                proposal=proposal,
                action=('needs_actual_pair_history_and_exact_width_costs' if proposal else
                        'stop_resource_branch' if failed else 'needs_both_actual_queries'),
                execution_authorized=False, final=False, default_enabled=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    require(not args.out.exists(), 'preserve previous output')
    document = json.loads(args.inputs.read_text())
    result = inspect_pair(**document)
    with args.out.open('x') as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write('\n')


if __name__ == '__main__':
    main()
