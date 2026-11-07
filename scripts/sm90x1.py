#!/usr/bin/env python3
"""SM90X1 frozen inputs and Phase 0 validation; raw evidence stays in runs/sm90x1."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import shutil
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r4_inputs import extract, model_dir
from sm90r4_fixed import native_command
from sm90r4_dispatch import claim, POLICY, watch, STALL_S
from sm90x1_hardware import run_unit, audit_unit, audit_samples, audit_trace_unit
from sm90r3_llm import environment, MODELS, PCI

RUN = ROOT / 'runs/sm90x1'
CELLS = ROOT / 'configs/serving/sm90x1/cells.json'
BASE = '3ff07bc2e'


def head():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()


def coordinate_bytes(c):
    return (json.dumps(c, sort_keys=True, indent=2) + '\n').encode()


def cells(selected=None):
    rows = read(CELLS)['cells']
    if selected == 'pilot':
        rows = [c for c in rows if c['pilot']]
    elif selected and selected != 'all':
        rows = [c for c in rows if c['cell'] == selected]
    if not rows:
        raise ValueError(f'unknown cell: {selected}')
    return rows


def directory(cell):
    return RUN / 'cells' / cell['cell']


def prepare(a):
    # A tracked list at this HEAD is required before any timed process.
    subprocess.check_call(['git', 'diff', '--exit-code', 'HEAD', '--', str(CELLS)], cwd=ROOT)
    subprocess.check_call(['git', 'ls-files', '--error-unmatch', str(CELLS)], cwd=ROOT,
                          stdout=subprocess.DEVNULL)
    for c in cells(a.cell):
        d = directory(c)
        d.mkdir(parents=True, exist_ok=True)
        coords = d / 'W.json'
        expected = coordinate_bytes(c['coordinates'])
        if coords.exists() and coords.read_bytes() != expected:
            raise ValueError(f'frozen W changed: {coords}')
        coords.write_bytes(expected)
        if sha(coords) != c['w_coordinates_sha256'] or sha(c['prefill']) != c['prefill_sha256']:
            raise ValueError('frozen input hash mismatch')
        graph = d / 'decode.export.json'
        extract(c['model'], 'decode', graph)
        write(d / 'inputs.json', dict(cell=c, list_sha256=sha(CELLS), export_sha256=sha(graph)))
    RUN.mkdir(parents=True, exist_ok=True)
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id,name,memory.used,'
                                         'utilization.gpu,power.draw,clocks.sm', '--format=csv'], text=True)
    apps = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,process_name,used_memory',
                                    '--format=csv'], text=True)
    write(RUN / 'environment.json', dict(head=head(), baseline=BASE, inventory=inventory, apps=apps,
          hostname=os.uname().nodename, frozen_cells_sha256=sha(CELLS), at=time.time()))


def tree_rss(pid):
    processes = {}
    for entry in Path('/proc').glob('[0-9]*/stat'):
        try:
            fields = entry.read_text().rsplit(')', 1)[1].split()
            processes[int(entry.parent.name)] = (int(fields[1]), int(fields[21]))
        except (OSError, ValueError, IndexError):
            continue
    descendants = {pid}
    while True:
        expanded = descendants | {p for p, (parent, _) in processes.items() if parent in descendants}
        if expanded == descendants:
            break
        descendants = expanded
    return sum(rss for p, (_, rss) in processes.items() if p in descendants)*os.sysconf('SC_PAGE_SIZE')


def freeze_compiler(compiler):
    compiler=Path(compiler).resolve()
    digest=sha(compiler)
    frozen=RUN/'compiler-snapshots'/digest/'tilemega'
    frozen.parent.mkdir(parents=True,exist_ok=True)
    if not frozen.exists():
        shutil.copy2(compiler,frozen)
    if sha(frozen)!=digest:
        raise ValueError('immutable compiler snapshot changed')
    return frozen


def build(a):
    compiler = freeze_compiler(a.compiler)
    for c in cells(a.cell):
        d = directory(c)
        out = d / a.tag / 'W'
        if out.exists():
            raise FileExistsError(f'choose a new tag, preserving failed builds: {out}')
        out.mkdir(parents=True)
        binary = out / 'decode.so'
        command = native_command(compiler, d / 'decode.export.json', binary, d / 'W.json', c['batch'])
        start = time.monotonic()
        with (out / 'build.log').open('w') as log:
            process = subprocess.Popen(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                       stdout=log, stderr=subprocess.STDOUT)
            peak_rss = 0
            while process.poll() is None:
                peak_rss = max(peak_rss, tree_rss(process.pid))
                time.sleep(0.5)
            status = process.returncode
        record = dict(cell=c['cell'], head=head(), command=command, status=status,
                      wall_s=time.monotonic()-start, peak_tree_rss_bytes=peak_rss,
                      memory_method='sum process-tree RSS sampled every 0.5s', compiler_sha256=sha(compiler))
        write(out / 'build.json', record)
        if status:
            raise RuntimeError(f'build failed: {out}')
        abi = [PYTHON, str(ROOT / 'scripts/check_sm90r2_native_abi.py'), '--prefill', c['prefill'],
               '--decode', str(binary), '--model-config', str(model_dir(c['model']) / 'config.json')]
        with (out / 'host_abi.log').open('w') as log:
            status = subprocess.call(abi, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES='',
                                     PYTHONPATH=str(ROOT / 'python')), stdout=log, stderr=subprocess.STDOUT)
        identity = read(str(binary) + '.identity.json')
        record.update(host_abi_status=status, decode=str(binary), decode_sha256=sha(binary),
                      coordinates_sha256=sha(d / 'W.json'), prefill=c['prefill'],
                      prefill_sha256=c['prefill_sha256'], source_head=identity['source']['head'],
                      trace=identity['trace'], performance_eligible=not identity['trace'])
        write(out / 'binding.json', record)
        if status or identity['trace'] or identity['source']['head'] != head():
            raise RuntimeError(f'ABI/identity failed: {out}')
        print(json.dumps(record), flush=True)


def measurement(a):
    for c in cells(a.cell):
        d = directory(c)
        binding = read(d / a.tag / 'W/binding.json')
        if binding['source_head'] != binding['head'] or binding['decode_sha256'] != sha(binding['decode']):
            raise ValueError('measurement requires the recorded build HEAD and binary')
        validate_runtime(binding)
        if sha(c['prefill']) != c['prefill_sha256']:
            raise ValueError('frozen prefill changed')
        unit = d / a.tag / a.action
        spec = dict(name=f"sm90x1-{a.action}-{a.tag}-{c['cell']}", model=c['model'], batch=c['batch'],
                    kind='corr' if a.action == 'corr' else 'final',
                    arms={'W': dict(decode=binding['decode'], prefill=c['prefill'])},
                    order=['W'] * (5 if a.action == 'corr' else 3),
                    correctness=dict(c1=256, c2=True, reference_processes=1),
                    **{'class': c['class']})
        if a.action == 'time':
            spec.pop('correctness')
            corr = read(d / a.tag / 'corr/result.json')
            if not corr['pass'] or not corr['c1_pass'] or not corr['c2_pass']:
                raise ValueError('P-CORR must pass before timing')
            if not audit_unit(d / a.tag / 'corr')['passed']:
                raise ValueError('P-CORR hardware audit must pass before timing')
        spec_path = d / a.tag / f'{a.action}-spec.json'
        write(spec_path, spec)
        result = run_unit(spec_path, unit, c['class'])
        print(json.dumps(dict(cell=c['cell'], action=a.action,
                              result={k: v for k, v in result.items() if k not in ('rows', 'spec')})), flush=True)


def validate_runtime(binding):
    # Tooling commits may follow Phase 0 builds; executable kernel/runtime inputs may not.
    changed = subprocess.check_output(['git', 'diff', binding['head'], '--', 'include', 'lib',
                                      'python', 'configs/targets/sm_90_h800.json'], cwd=ROOT)
    if changed:
        raise ValueError('runtime/core differs from the recorded build; rebuild at one HEAD')


def trace(a):
    for c in cells(a.cell):
        trace_cell(a,c,'W',reference=True)


def trace_cell(a,c,build_id,reference=False):
    d = directory(c) / a.tag
    artifact = d / build_id
    binding = read(artifact / 'binding.json')
    validate_runtime(binding)
    binary = artifact / 'trace.so'
    if not binary.exists():
        command = [PYTHON, str(ROOT / 'scripts/run_sm90r1_build.py'), '--receipt',
                   str(artifact / 'trace-limits.json'), '--max-gib', '48', '--timeout', '5400',
                   '--lock', '/tmp/tilemega_sm90r2_native_build.lock', '--', PYTHON,
                   str(ROOT / 'scripts/build_sm90r2_trace.py'), '--input', binding['decode'],
                   '--out', str(binary), '--stage-level', '2']
        with (artifact / 'trace-build.log').open('w') as log:
            subprocess.check_call(command, cwd=ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''),
                                  stdout=log, stderr=subprocess.STDOUT)
    identity = read(str(binary) + '.identity.json')
    if not identity['trace'] or identity['performance_eligible']:
        raise ValueError('trace identity is performance-ineligible')
    unit = d / 'trace' if a.phase == 0 else artifact / 'trace'
    if (unit / 'unit.json').exists():
        if not audit_trace_unit(unit)['passed']:
            raise ValueError(f'cached trace failed the GPU-owner audit: {unit}')
        return
    unit.mkdir(parents=True, exist_ok=True)
    attempt = unit / f'try{time.time_ns()}'
    attempt.mkdir()
    gpu, handle = claim(c['class'], str(unit))
    try:
        policy = read(POLICY)
        uuid = policy['idle_power_mapping_provenance']['uuid_by_gpu'][str(gpu)]
        model_inputs, prompt, gib = MODELS[c['model']]
        jobs, previous = [], []
        for fresh in range(2):
            name = f"sm90x1-trace-{c['cell']}-{build_id}-{fresh}"
            out = '{attempt_dir}/job'
            command = ['numactl', f'--cpunodebind={0 if gpu < 4 else 1}', PYTHON,
                       str(ROOT / 'scripts/run_r16_with_device_identity.py'), '--guard-policy', str(POLICY),
                       '--out', out, '--expected-uuid', uuid, '--expected-pci', f'0000:{PCI[gpu]}:00.0',
                       '--numa-node', str(0 if gpu < 4 else 1), '--allow-no-measurements', '--',
                       PYTHON, '-m', 'tilemega.serving.random_checkpoint', '--config', str(ROOT / model_inputs),
                       '--out', out+'/checkpoint', '--seed', '0', '--max-gib', str(gib), '--',
                       PYTHON, str(ROOT / 'scripts/sm90r4_ledger.py'), 'run', '--prefill', c['prefill'],
                       '--decode', str(binary), *(['--reference', binding['decode']] if reference else []), '--prompt-ids',
                       str(ROOT / prompt), '--batch', str(c['batch']), '--tokens', '520', '--out', out+'/trace']
            jobs.append(dict(name=name, gpu=True, timing=True, priority=10, timeout_s=3600,
                             needs_free_mib=32768, cwd=str(ROOT), after=previous,
                             env=environment(gpu, 'trial' if c['class']=='trial' else 'serial'), command=command))
            previous = [name]
        write(attempt / 'queue/queue_trace.json', jobs)
        env = dict(os.environ, **environment(gpu, 'trial' if c['class']=='trial' else 'serial'))
        with (attempt / 'scheduler.log').open('w') as log:
            scheduler = subprocess.Popen([PYTHON, '-m', 'tilemega.bench.scheduler', '--queue-dir',
                                   str(attempt / 'queue'), '--out', str(attempt / 'scheduler'), '--policy',
                                   str(POLICY), '--deadline-hours', '24'], cwd=ROOT, env=env,
                                  stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            stalled = watch(scheduler, attempt / 'scheduler/state.json', c['class'], gpu, STALL_S)
        if stalled:
            write(attempt / 'abandoned.json', dict(reason='foreign occupancy while another allowed card was free'))
            raise RuntimeError(f'trace unit stalled; claim released and evidence retained: {attempt}')
        state = read(attempt / 'scheduler/state.json')
        if scheduler.returncode or len(state)!=2 or any(r['status']!='done' for r in state.values()):
            raise RuntimeError(f'trace unit incomplete: {unit}')
        audits = [audit_samples([json.loads(line) for line in
                  (Path(r['out'])/'guard.jsonl').read_text().splitlines()]) for r in state.values()]
        write(attempt/'hardware-audit.json', dict(passed=all(r['passed'] for r in audits), checks=audits))
        if not all(r['passed'] for r in audits):
            raise RuntimeError(f'trace GPU-owner audit failed: {attempt}')
        runs = [str(Path(r['out']) / 'job/trace') for r in state.values()]
        write(unit / 'unit.json', dict(gpu=gpu, runs=runs, processes=2, **{'class':c['class']},
                                       performance_eligible=False, head=binding['head']))
    finally:
        handle.close()
    print(json.dumps(dict(cell=c['cell'], overhead_pct=[read(Path(p)/'run.json').get('overhead_pct')
                                                      for p in runs])), flush=True)


def tables(a):
    rows = []
    stopped = False
    for c in cells('all'):
        d = directory(c) / a.tag
        row = dict(cell=c['cell'], batch=c['batch'], pilot=c['pilot'], status='not_run',
                   accepted_tpot_ms=c['accepted_tpot_ms'], tpot_ms='', change_pct='',
                   range_pct='', processes='', gpu='', **{'class': c['class']},
                   c1_pass='', c2_pass='', corr_status='not_run', hardware_rejected_units=0,
                   guard_rejected_attempts=0)
        rejected=list((d/'corr').glob('rejected-*/hardware-audit.json'))
        row['hardware_rejected_units']=sum(not read(p)['passed'] for p in rejected)+\
            len(list((d/'corr').glob('try*/hardware-audit-rejection.json')))
        row['guard_rejected_attempts']=sum(bool(read(p)['code']) for p in
            d.glob('*/try*/raw/*/attempt*/guard_result.json'))
        if (d/'corr/result.json').exists():
            corr=read(d/'corr/result.json')
            valid=audit_unit(d/'corr')['passed']
            row.update(c1_pass=corr['c1_pass'],c2_pass=corr['c2_pass'],
                       corr_status='passed' if corr['pass'] and valid else 'rejected')
        elif row['hardware_rejected_units']:
            row['corr_status']='hardware_rejected; awaiting_clean_unit'
        elif (d/'corr').exists():row['corr_status']='pending'
        if (d / 'time/result.json').exists():
            result = read(d / 'time/result.json')
            corr = read(d / 'corr/result.json')
            binding = read(d / 'W/binding.json')
            if binding['trace'] or not corr['pass']:
                raise ValueError('ineligible result')
            if not all(audit_unit(d / kind)['passed'] for kind in ('corr', 'time')):
                row.update(status='hardware_rejected')
                rows.append(row)
                continue
            timing = result['arms']['W']
            change = 100*(timing['tpot_ms']/c['accepted_tpot_ms']-1)
            row.update(status='gate_fail' if abs(change)>2 else 'gate_pass',
                       tpot_ms=timing['tpot_ms'], change_pct=change, range_pct=timing['range_pct'],
                       processes=timing['processes'], gpu=read(d / 'time/unit.json')['gpu'])
            stopped |= abs(change)>2
        rows.append(row)
    complete = sum(r['status'] in ('gate_pass', 'gate_fail') for r in rows) == 2
    write(RUN / 'phase0.json', dict(rows=rows, stop=stopped, complete=complete,
                                   head=head(), gate_pct=2, evidence='verified'))
    results = ROOT / 'docs/results'
    results.mkdir(exist_ok=True)
    with (results / 'SM90X1_phase0.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator='\n');writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(stop=stopped, complete=complete, measured=[r for r in rows if r['status']!='not_run']),indent=2))
    return 2 if stopped else 0 if complete else 3


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'build', 'corr', 'time', 'trace', 'analyze', 'arms', 'tables'))
    p.add_argument('--cell', help='pilot, all, or exact model-Bbatch')
    p.add_argument('--tag')
    p.add_argument('--phase', type=int, choices=(0,1,2,3), default=0)
    p.add_argument('--k', type=int, choices=(3,4))
    p.add_argument('--arms-only', action='store_true')
    p.add_argument('--compiler', type=Path, default=ROOT / 'runs/sm90x1/compiler/tools/tilemega')
    a = p.parse_args()
    a.tag = a.tag or f'phase{a.phase}'
    a.cell = a.cell or ('all' if a.phase>=2 or a.action=='tables' else 'pilot')
    if a.phase:
        import sm90x1_experiment as experiment
        if a.action=='tables':return experiment.tables(a)
        experiment.gate(a.phase)
        failures=[]
        for c in experiment.rows(a):
            local=argparse.Namespace(**vars(a));local.cell=c['cell']
            try:
                (experiment.measurement if a.action in ('corr','time') else getattr(experiment,a.action))(local)
            except Exception as error:
                record=dict(cell=c['cell'],action=a.action,error=str(error),at=time.time())
                path=experiment.cell_dir(c,a)/'failures'/f'{a.action}-{time.time_ns()}.json'
                write(path,record);failures.append(record)
                print(json.dumps(record),flush=True)
        return 1 if failures else 0
    if a.action in ('corr', 'time'):
        return measurement(a)
    return globals()[a.action](a)


if __name__ == '__main__':
    raise SystemExit(main() or 0)
