#!/usr/bin/env python3
"""SM90X1 frozen inputs and Phase 0 validation; raw evidence stays in runs/sm90x1."""
import argparse
import csv
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r4_inputs import extract, model_dir
from sm90r4_dispatch import run_unit

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


def build(a):
    compiler = a.compiler.resolve()
    for c in cells(a.cell):
        d = directory(c)
        out = d / a.tag / 'W'
        if out.exists():
            raise FileExistsError(f'choose a new tag, preserving failed builds: {out}')
        out.mkdir(parents=True)
        binary = out / 'decode.so'
        command = [str(compiler), 'compile', str(d / 'decode.export.json'), str(binary),
                   '--tmexec-config', str(d / 'W.json'), '--serving', 'decode', '--emit', 'serving',
                   '--batch', str(c['batch']), '--runtime-target', str(ROOT / 'configs/targets/sm_90_h800.json'),
                   '--nvcc-split-compile', '2']
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
        if binding['head'] != head() or binding['decode_sha256'] != sha(binding['decode']):
            raise ValueError('measurement requires the recorded build HEAD and binary')
        unit = d / a.tag / a.action
        spec = dict(name=f"sm90x1-{a.action}-{a.tag}-{c['cell']}", model=c['model'], batch=c['batch'],
                    kind='corr' if a.action == 'corr' else 'final',
                    arms={'W': dict(decode=binding['decode'], prefill=c['prefill'])},
                    order=['W'] * (5 if a.action == 'corr' else 3),
                    correctness=dict(c1=256, c2=True, reference_processes=1),
                    **{'class': c['class']})
        if a.action == 'time':
            corr = read(d / a.tag / 'corr/result.json')
            if not corr['pass'] or not corr['c1_pass'] or not corr['c2_pass']:
                raise ValueError('P-CORR must pass before timing')
        spec_path = d / a.tag / f'{a.action}-spec.json'
        write(spec_path, spec)
        result = run_unit(spec_path, unit, c['class'])
        print(json.dumps(dict(cell=c['cell'], action=a.action,
                              result={k: v for k, v in result.items() if k not in ('rows', 'spec')})), flush=True)


def tables(a):
    rows = []
    stopped = False
    for c in cells('all'):
        d = directory(c) / a.tag
        row = dict(cell=c['cell'], batch=c['batch'], pilot=c['pilot'], status='not_run',
                   accepted_tpot_ms=c['accepted_tpot_ms'], tpot_ms='', change_pct='',
                   range_pct='', processes='', gpu='', **{'class': c['class']})
        if (d / 'time/result.json').exists():
            result = read(d / 'time/result.json')
            corr = read(d / 'corr/result.json')
            binding = read(d / 'W/binding.json')
            if binding['trace'] or not corr['pass']:
                raise ValueError('ineligible result')
            timing = result['arms']['W']
            change = 100*(timing['tpot_ms']/c['accepted_tpot_ms']-1)
            row.update(status='gate_fail' if abs(change)>2 else 'gate_pass',
                       tpot_ms=timing['tpot_ms'], change_pct=change, range_pct=timing['range_pct'],
                       processes=timing['processes'], gpu=read(d / 'time/unit.json')['gpu'])
            stopped |= abs(change)>2
        rows.append(row)
    complete = sum(r['status'] != 'not_run' for r in rows) == 2
    write(RUN / 'phase0.json', dict(rows=rows, stop=stopped, complete=complete,
                                   head=head(), gate_pct=2, evidence='verified'))
    results = ROOT / 'docs/results'
    results.mkdir(exist_ok=True)
    with (results / 'SM90X1_phase0.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(dict(stop=stopped, complete=complete, measured=[r for r in rows if r['status']!='not_run']),indent=2))
    return 2 if stopped else 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('prepare', 'build', 'corr', 'time', 'tables'))
    p.add_argument('--cell', default='pilot', help='pilot, all, or exact model-Bbatch')
    p.add_argument('--tag', default='phase0')
    p.add_argument('--compiler', type=Path, default=ROOT / 'runs/sm90x1/compiler/tools/tilemega')
    a = p.parse_args()
    if a.action in ('corr', 'time'):
        return measurement(a)
    return globals()[a.action](a)


if __name__ == '__main__':
    raise SystemExit(main() or 0)
