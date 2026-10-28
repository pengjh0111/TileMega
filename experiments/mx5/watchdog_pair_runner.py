"""Execute source-bound native top4 only after complete WD0/WD1 pair proofs.

The kernel/serving code and low-level correctness, timing and guard checks come
from the immutable producer tree selected through PYTHONPATH. This adds the
specified resource branch for a watchdog failure; it does not accept legacy code.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import os
from pathlib import Path
import sys

from tilemega import mx5_resolved_runner as base
from tilemega.mx5_select import check_file, record

spec = importlib.util.spec_from_file_location('mx5_actual_watchdog_pairs', Path(__file__).with_name('watchdog_pairs.py'))
pairs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pairs)
ROOT = base.ROOT
load, write, need = base.load, base.write, base.need


def ready(protocol_path, domain_tool):
    protocol = load(protocol_path)
    need(protocol['kind'] == 'mx5_watchdog_pair_top4_protocol' and protocol['schema_version'] == 1,
         'wrong paired protocol')
    need(protocol['domain_tool'] == record(domain_tool), 'paired domain changed')
    need(protocol['observer_tool'] == record(Path(pairs.__file__)) and
         protocol['runner_tool'] == record(Path(__file__)), 'paired helper bytes changed')
    need(len(protocol['plans']) == 4 and [p['candidate_id'] for p in protocol['plans']] == list(range(4)),
         'paired protocol must retain exactly four original IDs')
    preview = load(check_file(protocol['origin_preview']))
    plans = []
    for index, item in enumerate(protocol['plans']):
        h = check_file(item['history'])
        result = pairs.replay_history(h, domain_tool, check_file(protocol['stage_sources']),
                                      item['queries'], check_file(protocol['catalog']))
        need(result['resource_cost_query_ready'] and result['candidate_id'] == index and
             result['origin_preview'] == protocol['origin_preview'], 'pair resources, queries or costs are incomplete')
        observation = load(check_file(result['terminal_pair']))
        a = observation['WD0']
        identity = base._identity(check_file(a['files']['binary']), result['pin'], preview['request'],
                                  preview['producer'], 0, a['snapshot'])
        artifact = dict(binary=a['files']['binary'], identity_id=identity['artifact_id'],
                        pin=result['pin'], source_snapshot=a['files']['source_snapshot'])
        wd, _ = base._watchdog(check_file(observation['WD1']['files']['binary']), result['pin'],
                               preview, a['snapshot'], item['queries']['WD1'])
        plans.append(dict(original=preview['candidates'][index],
                          final=dict(pin=result['pin'], effective_prediction=dict(
                              effective_predicted_tpot_ns=result['effective_predicted_tpot_ns'],
                              effective_stage_ns=result['effective_stage_ns'])),
                          artifact=artifact, watchdog=wd, review=result))
    return dict(protocol=record(protocol_path), origin_preview=protocol['origin_preview'], plans=plans)


def _bound(manifest):
    for field in ('protocol', 'domain_tool', 'prefill', 'prefill_watchdog', 'model_config', 'prompts',
                  'policy', 'runner', 'observer', 'guard_wrapper', 'check_wrapper', 'guard_policy'):
        check_file(manifest[field])
    need(manifest['runner'] == record(Path(__file__)) and manifest['observer'] == record(Path(pairs.__file__)),
         'paired execution helpers changed')
    for field, path in [('guard_wrapper', ROOT/'scripts/mx_gpu_job.sh'),
                        ('check_wrapper', ROOT/'scripts/mx_check_modes.sh'),
                        ('guard_policy', ROOT/'configs/bench/guard_policy_c550.json')]:
        need(manifest[field] == record(path), 'frozen serving helper changed')


def verify_results(path, protocol, domain_tool):
    result = load(path)
    need(result['kind'] == 'mx5_watchdog_pair_top4_results' and result['schema_version'] == 1,
         'wrong paired result schema')
    m = result['manifest']
    need(m['protocol'] == record(protocol) and m['domain_tool'] == record(domain_tool), 'paired result binding changed')
    _bound(m)
    proof = ready(protocol, domain_tool)
    preview = load(check_file(proof['origin_preview']))
    need(m['batch'] == preview['request']['batch'] and m['gpu'] in (4,5,6,7), 'paired result workload changed')
    base._check_process(m['native_revalidation'], m['native_revalidation_file'])
    base._fresh(preview, m['fresh_preview'], m['native_revalidation'])
    need(base._prefill(check_file(m['prefill']), 0)['artifact_id'] == m['prefill_identity'] and
         base._prefill(check_file(m['prefill_watchdog']), 1)['artifact_id'] == m['prefill_watchdog_identity'],
         'paired prefill identity changed')
    need(len(m['candidates']) == 4 and [e['index'] for e in m['candidates']] == list(range(4)), 'lost original candidates')
    for i, entry in enumerate(m['candidates']):
        need(entry['artifact'] == proof['plans'][i]['artifact'] and entry['watchdog'] == proof['plans'][i]['watchdog'],
             'checked/measured native pair substituted')
        process = entry['check_process']
        base._check_process(process, entry['check_process_file'])
        base._guard(process, m['gpu'], 1700)
        need(base._guard_files(process) == entry['check_guard'], 'paired correctness guard changed')
        expected = ['bash', str(ROOT/'scripts/mx_check_modes.sh'), m['model'],
                    str(check_file(m['prefill_watchdog'])), str(check_file(entry['watchdog']['binary'])),
                    str(m['batch']), '512', entry['check_out'], str(check_file(m['prompts']))]
        need(process['argv'][4:] == expected, 'paired C1/C2 did not execute the bound watchdog')
        need(base._correctness(Path(entry['check_out']), m['batch'], Path(m['model'])) == entry['correctness'],
             'paired C1/C2 evidence changed')
    order = [(r,(j+r)%4) for r in range(2) for j in range(4)]
    need([(r['round'],r['candidate']) for r in m['measurements']] == order, 'paired timing order changed')
    values = [[] for _ in range(4)]
    pids = set()
    args = argparse.Namespace(root=ROOT, gpu=m['gpu'], model=Path(m['model']),
        prefill_so=check_file(m['prefill']), prompt_ids=check_file(m['prompts']), batch=m['batch'],
        prefill_mode=m['prefill_mode'], policy=check_file(m['policy']))
    for item in m['measurements']:
        i = item['candidate']
        process = item['process']
        base._check_process(process,item['process_file'])
        _, pid = base._guard(process, m['gpu'], 900)
        need(pid == item['measurement_pid'] and pid not in pids, 'paired timing process is not fresh')
        pids.add(pid)
        need(base._guard_files(process) == item['guard'], 'paired timing guard changed')
        expected = base.common.measurement_command(args, check_file(proof['plans'][i]['artifact']['binary']),
                                                    Path(item['output']), Path(process['argv'][3]))
        need(process['argv'] == expected, 'paired measured binary/protocol changed')
        report = load(check_file(item['raw']))
        need(report.get('measurement_policy',{}).get('guard') is True and not report.get('synthetic_fixture'),
             'unguarded/synthetic paired timing')
        tpot = base.common.validate_measurement(report, preview, proof['plans'][i]['artifact']['identity_id'], m['prefill_identity'])
        need(tpot == item['tpot_ns'], 'paired actual TPOT changed')
        values[i].append(tpot)
    selected = base._selection(proof,values,[p['artifact'] for p in proof['plans']],[p['watchdog'] for p in proof['plans']])
    need(result['selection'] == selected, 'paired one-percent selection changed')
    for name, value in [('selection.json',selected),('selected.pin.json',selected['selected_pin'])]:
        p = Path(path).parent/name
        if p.exists():need(load(p) == value, 'paired published selection changed')
    return selected


def run(args):
    need(args.root == ROOT and args.gpu in (4,5,6,7), 'unsupported paired producer root/card')
    os.chdir(ROOT)
    proof = ready(args.protocol,args.domain_tool)
    preview = load(check_file(proof['origin_preview']))
    q = preview['request']
    need(q['batch'] == args.batch and (q['capacity'],q['past_lo'],q['past_hi']) == (1088,64,1086),
         'standard paired decode workload required')
    from tilemega.serving.workload import validate_prompt_ids
    validate_prompt_ids(load(args.prompt_ids),args.batch,64)
    need(load(args.policy).get('guard') is True, 'guard policy disabled')
    p0,p1 = base._prefill(args.prefill_so,0),base._prefill(args.prefill_watchdog_so,1)
    args.out.mkdir(parents=True,exist_ok=False)
    m = dict(protocol=record(args.protocol),domain_tool=record(args.domain_tool),
        model=str(args.model),model_config=record(args.model/'config.json'),prompts=record(args.prompt_ids),
        policy=record(args.policy),prefill=record(args.prefill_so),prefill_watchdog=record(args.prefill_watchdog_so),
        prefill_identity=p0['artifact_id'],prefill_watchdog_identity=p1['artifact_id'],
        runner=record(Path(__file__)),observer=record(Path(pairs.__file__)),
        guard_wrapper=record(ROOT/'scripts/mx_gpu_job.sh'),check_wrapper=record(ROOT/'scripts/mx_check_modes.sh'),
        guard_policy=record(ROOT/'configs/bench/guard_policy_c550.json'),gpu=args.gpu,batch=args.batch,
        prefill_mode=args.prefill_mode,candidates=[],measurements=[],invocation='explicit_native_WD0_WD1_resource_pair',
        audit_execution_authorized=False)
    env = dict(os.environ,PYTHONPATH=str(ROOT/'python'),NEEDS_MIB=str(args.needs_mib),
               TILEMEGA_GPU_JOB_LOG=str(os.environ.get('TILEMEGA_GPU_JOB_LOG',ROOT/'runs/mx5/gpu_jobs.log')))
    for k in ('TILEMEGA_MX_DIAG','TILEMEGA_TRACE'):env.pop(k,None)
    cpu = dict(env,CUDA_VISIBLE_DEVICES='',MACA_VISIBLE_DEVICES='')
    def snapshot(status,**extra):
        write(args.out/f"state{len(list(args.out.glob('state*.json'))):03d}.json",dict(status=status,manifest=m,**extra))
    snapshot('paired_preparing')
    try:
        prefix = args.out/'fresh_native/preview'
        prefix.parent.mkdir()
        process = base.run_command(base.common.native_solve_command(preview,prefix),prefix.parent/'process',env=cpu)
        m.update(native_revalidation=process,native_revalidation_file=base._process_file(prefix.parent/'process',process),
                 fresh_preview=record(str(prefix)+'.mx5_candidates.json'))
        base._fresh(preview,m['fresh_preview'],process)
        snapshot('paired_origin_revalidated')
        for i in range(4):
            _bound(m)
            proof = ready(args.protocol,args.domain_tool)
            p = proof['plans'][i]
            directory = args.out/f'candidate{i}'
            directory.mkdir()
            check_out,guard = directory/'correctness',directory/'check_guard'
            command = [str(ROOT/'scripts/mx_gpu_job.sh'),str(args.gpu),'1700',str(guard),
                       'bash',str(ROOT/'scripts/mx_check_modes.sh'),str(args.model),str(args.prefill_watchdog_so),
                       str(check_file(p['watchdog']['binary'])),str(args.batch),'512',str(check_out),str(args.prompt_ids)]
            process = base.run_command(command,directory/'check_process',timeout=1730,env=env)
            base._guard(process,args.gpu,1700)
            m['candidates'].append(dict(index=i,artifact=p['artifact'],watchdog=p['watchdog'],check_process=process,
                check_process_file=base._process_file(directory/'check_process',process),check_guard=base._guard_files(process),
                check_out=str(check_out),correctness=base._correctness(check_out,args.batch,args.model)))
            snapshot('paired_C1C2_passed')
        for r in range(2):
            for j in range(4):
                i = (j+r)%4
                _bound(m)
                proof = ready(args.protocol,args.domain_tool)
                a = proof['plans'][i]['artifact']
                directory = args.out/f'candidate{i}/round{r}'
                output,guard = directory/'measure',directory/'guard'
                command = base.common.measurement_command(args,check_file(a['binary']),output,guard)
                process = base.run_command(command,directory,timeout=930,env=env)
                _,pid = base._guard(process,args.gpu,900)
                raw = record(output/'measurements.json')
                report = load(check_file(raw))
                need(report.get('measurement_policy',{}).get('guard') is True and not report.get('synthetic_fixture'),
                     'unguarded/synthetic paired measurement')
                value = base.common.validate_measurement(report,preview,a['identity_id'],p0['artifact_id'])
                m['measurements'].append(dict(candidate=i,round=r,process=process,
                    process_file=base._process_file(directory,process),guard=base._guard_files(process),
                    measurement_pid=pid,output=str(output),raw=raw,tpot_ns=value))
                snapshot('paired_measured')
        values = [[p['tpot_ns'] for p in m['measurements'] if p['candidate'] == i] for i in range(4)]
        selected = base._selection(proof,values,[p['artifact'] for p in proof['plans']],[p['watchdog'] for p in proof['plans']])
        result = dict(kind='mx5_watchdog_pair_top4_results',schema_version=1,manifest=m,selection=selected)
        write(args.out/'results.json',result)
        verified = verify_results(args.out/'results.json',args.protocol,args.domain_tool)
        write(args.out/'selection.json',verified)
        write(args.out/'selected.pin.json',verified['selected_pin'])
        snapshot('paired_measured_selection_complete')
    except BaseException as e:
        snapshot('paired_blocked',reason=str(e))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--domain-tool',type=Path,required=True)
    parser.add_argument('--verify-results',type=Path)
    if '--verify-results' in sys.argv:
        args = parser.parse_args()
        verify_results(args.verify_results.resolve(),args.protocol.resolve(),args.domain_tool.resolve())
        return
    for name in ('root','model','prefill-so','prefill-watchdog-so','prompt-ids','policy','out'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--batch',type=int,choices=(1,2,4,8,16),required=True)
    parser.add_argument('--gpu',type=int,choices=(7,6,5,4),required=True)
    parser.add_argument('--prefill-mode',choices=('L1','L2'),default='L2')
    parser.add_argument('--needs-mib',type=int,default=20480)
    args = parser.parse_args()
    for name,value in vars(args).items():
        if isinstance(value,Path):setattr(args,name,value.resolve())
    run(args)


if __name__ == '__main__':
    main()
