#!/usr/bin/env python3
"""Advance AB4 B4 dependencies using independently queued GPU processes."""
import argparse
import concurrent.futures
import json
from pathlib import Path
import statistics as st
import subprocess
import time
import traceback

from sm90d1_common import ROOT, PYTHON, read, sha, write
from sm90r3_llm import span
from sm90r6_ab4_supplement import validate
from sm90r6_audit_final_measurements import audit_directory
from sm90r6_merge_ab4_supplement import audit_correctness, comparison_table
from sm90r6_merge_ablation import compact
from sm90r6_merge_analysis import write as csv_write

PROTOCOL = 'user-authorized dynamic independent-process supplement'


def summarize(rows):
    result = {}
    for label in dict.fromkeys(v['arm'] for v in rows):
        values = [v for v in rows if v['arm'] == label]
        result[label] = dict(**{k: st.median(v[k] for v in values) for k in
            ('tpot_ms', 'ttft_ms', 'e2e_ms', 'tokens_per_s')},
            range_pct=span([v['tpot_ms'] for v in values]), processes=len(values))
    return result


class Collection:
    def __init__(self, root):
        self.root = root
        self.previous = Path(read(root / 'source.json')['previous'])
        self.cells = read(root / 'cells.json')
        self.compiler = ROOT / 'runs/sm90r6/ablation/finalists-v1/compiler'
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self.builds = {}
        self.process_audits = {}
        self.failures = {}
        self.items = {}
        paths = [ROOT / 'scripts/sm90r3_llm.py', ROOT / 'scripts/sm90r6_realize_check.py',
                 ROOT / 'configs/targets/sm_90_h800.json', self.compiler]
        for cell in self.cells:
            d = self.previous / cell['id']
            paths += [d / 'receipt.json']
            for value in read(d / 'build/candidates.json'):
                paths += [Path(value['binary']), Path(value['binary'] + '.identity.json')]
        self.inputs = {str(p): sha(p) for p in paths}
        if (root / 'inputs.json').exists() and read(root / 'inputs.json') != self.inputs:
            raise ValueError('collection inputs changed')
        write(root / 'inputs.json', self.inputs)
        write(root / 'controller-source.json', dict(path=__file__, sha256=sha(__file__), protocol=PROTOCOL))

    def emit(self, job):
        path = self.root / 'requests' / (job['id'] + '.json')
        value = dict(job=job)
        if path.exists():
            if read(path)['job'] != job:
                raise ValueError('item definition changed: ' + job['id'])
        else:
            write(path, value)

    def completed(self, names):
        for name in names:
            if self.items.get(name, {}).get('status') == 'failed':
                raise ValueError('GPU item failed: ' + name)
        return all(self.items.get(name, {}).get('status') == 'done' for name in names)

    def audit_process(self, name, cell):
        if name in self.process_audits:
            return self.process_audits[name]
        item = self.items[name]
        row = item['result']
        folder = self.root / 'process-audits' / name
        write(folder / 'inputs.json', self.inputs)
        write(folder / 'unit/result.json', dict(rows=[row], arms=summarize([row])))
        result = audit_directory(folder, [row['arm']], cell['model'], 4)
        write(folder / 'audit.json', result)
        self.process_audits[name] = result
        return result

    def build_sequence(self, cell, selected):
        directory = self.root / cell['id'] / 'AB4c'
        directory.mkdir(exist_ok=True)
        binary = directory / 'decode.so'
        coordinates = self.previous / cell['id'] / 'build' / f'candidate-{selected["winner_rank"]}' / 'coordinates.json'
        if not Path(str(binary) + '.compile-result.json').exists():
            command = list(map(str, [self.compiler, 'compile', cell['export'], binary,
                '--tmexec-config', coordinates, '--runtime-target', ROOT / 'configs/targets/sm_90_h800.json',
                '--batch', 4, '--serving', 'decode', '--emit', 'kernel-sequence', '--nvcc-split-compile', 4]))
            write(directory / 'compile-command.json', command)
            with (directory / 'compile.log').open('w') as log:
                subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        outcome = read(str(binary) + '.compile-result.json')
        if outcome['status'] != 'compiled' or not outcome['inputs_unchanged']:
            raise ValueError('sequence build is not stable')
        return str(binary)

    def correctness(self, cell, label, binary, reference):
        directory = self.root / cell['id'] / label
        directory.mkdir(parents=True, exist_ok=True)
        prompt = 'olmo' if cell['model'].startswith('olmo') else 'llama3' if cell['model'].startswith('llama') else 'qwen'
        names = []
        for slot in range(5):
            name = f'{cell["id"]}-{label}-c2-{slot}'
            names.append(name)
            self.emit(dict(id=name, kind='c2', model=cell['model'], batch=4, slot=slot, priority=20,
                reference=reference, candidate=binary, prompts=str(ROOT / f'configs/prompts/{prompt}_isl64.json')))
        c1name = f'{cell["id"]}-{label}-c1'
        spec = dict(name=c1name, kind='corr', protocol=PROTOCOL, model=cell['model'], batch=4,
                    arms={'winner': dict(decode=binary, prefill='prompt')},
                    correctness=dict(c1=256, reference_processes=1))
        self.emit(dict(id=c1name, kind='llm', model=cell['model'], batch=4, slot=0,
                       priority=20, arm='winner', spec=spec))
        if self.completed(names) and not (directory / 'c2/result.json').exists():
            processes = []
            for slot, name in enumerate(names):
                source = Path(self.items[name]['result']['directory'])
                adapter = directory / 'c2' / f'p{slot}'
                adapter.mkdir(parents=True, exist_ok=True)
                for link, target in [('guard', source), ('checks', source / 'checks'),
                                     ('checkpoint', source / 'checkpoint'), ('command.json', source / 'command.json')]:
                    if not (adapter / link).exists():
                        (adapter / link).symlink_to(target, target_is_directory=link != 'command.json')
                write(adapter / 'result.json', dict(returncode=0, source_item=name, source_attempt=str(source)))
                value = dict(self.items[name]['result'])
                value['directory'] = str(adapter)
                processes.append(value)
            if len({v['checkpoint'] for v in processes}) != 1 or len({v['tokens_sha256'] for v in processes}) != 1:
                raise ValueError('C2 fresh-process/GPU results differ')
            write(directory / 'c2/result.json', dict(passed=True, processes=processes, model=cell['model'],
                batch=4, steps=1024, artifacts={k: dict(path=v, sha256=sha(v)) for k, v in
                    [('reference', reference), ('candidate', binary)]}, protocol=PROTOCOL,
                ownership='provisional_container_pid_namespace'))
        if self.completed([c1name]) and not (directory / 'c1.json').exists():
            item = self.items[c1name]
            row = item['result']
            attempt = Path(row['raw'])
            write(directory / 'c1.json', dict(spec=read(attempt / 'spec.json'), rows=[row],
                c1_pass=row['c1']['pass'], **{'pass': row['c1']['pass']}, protocol=PROTOCOL))
        if (directory / 'c2/result.json').exists() and (directory / 'c1.json').exists():
            arm = dict(binary=binary, c1=str(directory / 'c1.json'), c2=str(directory / 'c2/result.json'))
            if not (directory / 'correctness-audit.json').exists():
                write(directory / 'correctness-audit.json', audit_correctness(arm, cell))
            return arm
        return None

    def advance(self, cell):
        directory = self.root / cell['id']
        previous = self.previous / cell['id']
        ocname = cell['id'] + '-occupancy'
        if not self.completed([ocname]):
            return 'occupancy'
        occupancy = self.items[ocname]['result']
        if len(occupancy['candidates']) != 4 or any(v['effective_residency'] != v['requested_residency'] for v in occupancy['candidates']):
            raise ValueError('occupancy rescore is required')
        write(directory / 'occupancy.json', occupancy)
        candidates = read(previous / 'build/candidates.json')
        labels = ['c' + str(v['rank']) for v in candidates]
        order = labels + labels[::-1]
        names = [f'{cell["id"]}-psel-{slot:02d}-{label}' for slot, label in enumerate(order)]
        if not self.completed(names):
            return 'P-SEL'
        selected_path = directory / 'psel/result.json'
        if not selected_path.exists():
            rows = [self.items[name]['result'] for name in names]
            for name in names:
                self.audit_process(name, cell)
            arms = summarize(rows)
            ranked = sorted(arms, key=lambda k: arms[k]['tpot_ms'])
            best, second = ranked[:2]
            threshold = max(.5, 2 * max(arms[best]['range_pct'], arms[second]['range_pct']))
            gap = (arms[second]['tpot_ms'] / arms[best]['tpot_ms'] - 1) * 100
            prediction = {f'c{v["rank"]}': v['predicted_ns'] for v in candidates}
            winner = min((best, second), key=lambda k: prediction[k]) if gap <= threshold else best
            rank = int(winner[1:])
            write(selected_path, dict(winner_rank=rank, winner_binary_sha256=sha(candidates[rank]['binary']),
                measured_order=ranked, arms=arms, rows=rows, predicted_ns=prediction, top_gap_pct=gap,
                threshold_pct=threshold, tie_broken_by_prediction=winner != best, protocol=PROTOCOL,
                gpu_assignment=[v['gpu'] for v in rows], logical_order=order, items=names))
            print(cell['id'] + ' P-SEL complete', flush=True)
        selected = read(selected_path)
        binary = candidates[selected['winner_rank']]['binary']
        if cell['id'] not in self.builds:
            self.builds[cell['id']] = self.pool.submit(self.build_sequence, cell, selected)
        arm_b = self.correctness(cell, 'AB4b', binary, binary)
        future = self.builds[cell['id']]
        if not future.done():
            return 'AB4c build / AB4b correctness'
        seqbinary = future.result()
        arm_c = self.correctness(cell, 'AB4c', seqbinary, binary)
        if not arm_b or not arm_c:
            return 'C1/C2'
        arms = dict(AB4b=arm_b, AB4c=arm_c)
        validate(arms, cell['model'], 4)
        write(directory / 'arms.json', arms)
        order = ['AB4b', 'AB4c', 'AB4c', 'AB4b', 'AB4b', 'AB4c']
        names = []
        for slot, label in enumerate(order):
            name = f'{cell["id"]}-final-{slot}-{label}'
            names.append(name)
            spec = dict(name=name, kind='sel', protocol=PROTOCOL, model=cell['model'], batch=4,
                arms={k: dict(decode=v['binary'], prefill='prompt') for k, v in arms.items()})
            self.emit(dict(id=name, kind='llm', model=cell['model'], batch=4, priority=30,
                           slot=slot, arm=label, spec=spec))
        if not self.completed(names):
            return 'final timing'
        if not (directory / 'measurement/result.json').exists():
            rows = [self.items[name]['result'] for name in names]
            audits = [self.audit_process(name, cell) for name in names]
            tokens = [read(v['tokens']) for v in rows]
            if any(v != tokens[0] for v in tokens[1:]) or len({v['checkpoint'] for v in audits}) != 1:
                raise ValueError('final timing tokens or checkpoint differ')
            values = summarize(rows)
            comparisons = {}
            for label, value in values.items():
                change = (value['tpot_ms'] / values['AB4b']['tpot_ms'] - 1) * 100
                threshold = max(.5, 2 * max(value['range_pct'], values['AB4b']['range_pct']))
                comparisons[label] = dict(change_pct=change, threshold_pct=threshold,
                    decision='indistinguishable' if abs(change) <= threshold else 'faster' if change < 0 else 'slower')
            write(directory / 'measurement/result.json', dict(protocol=PROTOCOL, arms=values, rows=rows,
                comparisons=comparisons, logical_order=order, items=names,
                gpus=sorted({v['gpu'] for v in rows}), checkpoint=audits[0]['checkpoint'],
                cross_arm_tokens_equal=True, same_card_unit_required=False))
            write(directory / 'complete.json', dict(status='complete', cell=cell, protocol=PROTOCOL))
            print(cell['id'] + ' final timing audited', flush=True)
        return 'complete'

    def run(self):
        while True:
            if any(sha(p) != digest for p, digest in self.inputs.items()):
                raise ValueError('collection inputs changed')
            self.items = read(self.root / 'queue/items.json')
            status = []
            for cell in self.cells:
                try:
                    phase = self.advance(cell) if cell['id'] not in self.failures else 'failed'
                except Exception as error:
                    self.failures[cell['id']] = str(error)
                    write(self.root / cell['id'] / 'failure.json', dict(error=str(error), traceback=traceback.format_exc()))
                    phase = 'failed'
                relevant = [v for k, v in self.items.items() if k.startswith(cell['id'] + '-')]
                status.append(dict(model=cell['model'], cell=cell['id'], phase=phase,
                    psel_accepted=sum(v['status'] == 'done' and '-psel-' in v['job']['id'] for v in relevant),
                    c2_accepted=sum(v['status'] == 'done' and v['job']['kind'] == 'c2' for v in relevant),
                    c1_accepted=sum(v['status'] == 'done' and v['job']['kind'] == 'llm'
                                    and v['job']['spec']['kind'] == 'corr' for v in relevant),
                    final_accepted=sum(v['status'] == 'done' and '-final-' in v['job']['id'] for v in relevant)))
            write(self.root / 'status.json', dict(at=time.time(), protocol=PROTOCOL, cells=status,
                running=[dict(item=k, gpu=v['gpu']) for k, v in self.items.items() if v['status'] == 'running'],
                failed=self.failures))
            if all(v['phase'] == 'complete' for v in status):
                (self.root / 'requests/CLOSED').touch()
                return
            time.sleep(2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    Collection(args.root.resolve()).run()
