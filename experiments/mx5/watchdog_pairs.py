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


def replay_history(history_path, domain_tool, stage_sources, queries=None, catalog=None):
    """Replay every actual pair and reduction; missing costs/queries never grant execution."""
    saved = json.loads(Path(history_path).read_text())
    preview = check_file(saved['original_preview'])
    candidate = saved['candidate']
    origin = resolved.Qualified576Origin.capture(record(preview))
    table = json.loads(check_file(origin.data['preview']['stage_table']).read_text())
    resolved._complete_domain(origin.data['preview'], table, Path(domain_tool))
    source_inventory = record(stage_sources)
    sources = json.loads(check_file(source_inventory).read_text())
    chain = saved['chain']
    require(0 < len(chain) <= 4, 'pair history must contain one to four actual attempts')
    require([r['step'] for r in chain] == list(range(len(chain))), 'noncontiguous pair history')
    require(saved['used_reductions'] == len(chain) - 1, 'shared reduction count changed')
    expected = origin.candidate(candidate)['pin']
    initial_epoch = None
    authorizations = {}
    actual_requests = []
    for ordinal, entry in enumerate(chain):
        path = check_file(entry['observation'])
        observed = json.loads(path.read_text())
        require(observed['origin_preview'] == record(preview) and observed['candidate_id'] == candidate,
                'observation belongs to another origin/candidate')
        require(observed['pin'] == expected and observed['used_reductions'] == ordinal,
                'pair changed the prescribed pin or reset its budget')
        for tag in ('WD0', 'WD1'):
            for name in ('native', 'resources'):
                check_file(observed[tag][name])
        fresh = inspect_pair(preview, candidate, expected,
                             {k:observed['WD0'][k]['path'] for k in ('native', 'resources')},
                             {k:observed['WD1'][k]['path'] for k in ('native', 'resources')},
                             sources, ordinal)
        require(fresh == observed, 'saved pair differs from actual immutable native observations')
        epoch = policy.program_epoch(fresh['WD0']['files'])
        require(initial_epoch is None or initial_epoch == epoch, 'pair source/compiler/options changed')
        initial_epoch = epoch
        if ordinal + 1 < len(chain):
            proposal = fresh['proposal']
            require(fresh['failed_resources'] and proposal is not None,
                    'a fitting pair or exhausted branch cannot authorize another reduction')
            actual_requests.append(dict(parent_pair=entry['observation'], proposal=proposal))
            name = proposal['shared_class']
            authorizations[name] = dict(kind='plan_resource_retry', schema_version=1,
                origin_sha256=origin.sha256, candidate_id=candidate,
                parent_attempt_sha256=policy.digest(fresh), parent_pair_observation=entry['observation'],
                program_source_epoch=epoch, reduction_step=ordinal + 1, shared_class=name,
                from_pw=proposal['from_pw'], to_pw=proposal['to_pw'],
                previous_pin_sha256=policy.digest(expected),
                next_pin_sha256=policy.digest(proposal['next_pin']),
                affected_edges=policy.affected_edges(name, origin.data['layers']),
                trigger='actual_WD0_WD1_resource_pair', policy=policy.POLICY)
            expected = proposal['next_pin']
    require(saved['changed_width_requests'] == actual_requests, 'saved width requests differ from actual failures')
    require(saved['pair_resources_fit'] is True and not fresh['failed_resources'], 'terminal pair still fails resources')
    missing_queries = []
    for tag in ('WD0', 'WD1'):
        if not queries or tag not in queries:
            missing_queries.append(tag)
        else:
            check_file(queries[tag])
            a = fresh[tag]
            resource = json.loads(check_file(a['resources']).read_text())
            query = resolved.exception._query(queries[tag], a['files'], a['resources'], 576, resource)
            # _query has already verified every raw API return code and guard binding.
            require(query['resident_ctas'] == 1,
                    'terminal actual occupancy is not one CTA/AP')
    stages = copy.deepcopy(origin.candidate(candidate)['predicted_stage_ns'])
    missing_costs = []
    bindings = {}
    variants = policy.VariantCatalog.load(catalog) if catalog is not None else None
    for name, auth in authorizations.items():
        request = policy._key(origin, candidate, name, expected)
        binding = variants.lookup(request, auth, fresh['WD0']['snapshot']['source_files']) if variants else None
        if binding is None:
            missing_costs.append(dict(stage=name, request=request, authorization=auth))
        else:
            require(binding.data['row']['r_next'] == expected['mx_decode_v2'][name]['r_next'],
                    'changed-width cost R_next differs from executed plan')
            bindings[name] = binding.data
            stages[name] = binding.data['row']['median_ns']
    ready = not missing_queries and not missing_costs
    prediction = stages['head'] + origin.data['layers'] * sum(stages[n] for n in policy.STAGES if n != 'head')
    return dict(kind='mx5_watchdog_pair_history_review', schema_version=1,
                history=record(history_path), origin_preview=record(preview), candidate_id=candidate,
                stage_sources=source_inventory, domain_tool=record(domain_tool),
                terminal_pair=chain[-1]['observation'], pin=expected, used_reductions=len(chain)-1,
                authorizations=authorizations, bindings=bindings, missing_queries=missing_queries,
                missing_costs=missing_costs, effective_stage_ns=stages,
                effective_predicted_tpot_ns=prediction if not missing_costs else None,
                resource_cost_query_ready=ready, execution_authorized=False,
                action='needs_all_four_paired_protocol' if ready else 'needs_actual_pair_queries_or_costs',
                final=False, default_enabled=False)


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
