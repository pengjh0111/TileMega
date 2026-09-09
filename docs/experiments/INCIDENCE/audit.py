#!/usr/bin/env python3
"""CPU-only A1/A11 evidence, without replacing historical experiment data."""
import csv
import json
import os
from pathlib import Path
import subprocess


def main():
    here = Path(__file__).resolve().parent
    repo = here.parents[2]
    build = repo/'build-portable'
    out = here/'history_audit'
    out.mkdir(exist_ok=True)
    commands = []
    def run(command, name):
        command = list(map(str, command))
        result = subprocess.run(command, capture_output=True, text=True,
                                env=dict(os.environ, TILEMEGA_ISL_AUDIT='1'))
        (out/name).write_text(result.stdout)
        (out/(name+'.log')).write_text(result.stderr)
        commands.append({'command': command, 'returncode': result.returncode})
        (out/'commands.json').write_text(json.dumps(commands, indent=2)+'\n')
        result.check_returncode()
    for name in ('llama','decode','llama4','mlp','mha','gather'):
        run([build/'tools/tilemega-derive', name], f'derived-{name}.md')
    for fixture, path in (('gqa2','E2E_GEN/raw/export_bridge.json'),
                          ('mha4','P3_GENERALIZATION/raw/export_bridge.json')):
        for granularity in ('reference','launch'):
            run([build/'tools/tilemega-wiring', repo/'docs/experiments'/path, granularity],
                f'wiring-{fixture}-{granularity}.md')
    run([build/'wiring_coupling_test'], 'wiring-test.txt')
    run([build/'table27_test'], 'table27-test.txt')
    run([build/'semantic_lifting_test'], 'semantic-lifting-test.txt')
    for model in ('e2e','mha'):
        stages = []
        for form in ('plain','core'):
            generated = out/f'{model}_{form}.cu'
            run([build/'tools/tilemega-compile',
                 repo/f'docs/experiments/SEMANTIC/raw/{model}_{form}.json', generated],
                f'{model}_{form}-codegen.txt')
            source = generated.read_text()
            start = source.index('kStages[]')
            stages.append(source[start:source.index('};',start)+2])
        if stages[0] != stages[1]: raise RuntimeError('normalization changed stage plan')
    before = list(csv.DictReader((here/'production_before.tsv').open(), delimiter='\t'))
    after = list(csv.DictReader((here/'production.tsv').open(), delimiter='\t'))
    key = lambda row: tuple(row[k] for k in ('model','seq','past','producer','consumer'))
    old = {key(row):row for row in before}
    if len(old) != len(before) or set(old) != {key(row) for row in after}:
        raise RuntimeError('before/after point universe differs')
    with (here/'production_comparison.tsv').open('w') as stream:
        fields = list(after[0]) + ['old_wait_sum','changed']
        writer = csv.DictWriter(stream, fields, delimiter='\t', lineterminator='\n')
        writer.writeheader()
        for row in after:
            prior = old[key(row)]
            if any(row[k] != prior[k] for k in ('fanout_sum','count','volume')):
                raise RuntimeError('unexpected fanout/count/volume change')
            writer.writerow({**row,'old_wait_sum':prior['wait_sum'],
                             'changed':int(prior['wait_sum'] != row['wait_sum'])})
    (out/'status.txt').write_text('PASS CPU derivation and normalization audit\n')


if __name__ == '__main__':
    main()
