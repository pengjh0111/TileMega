#!/usr/bin/env python3
"""Inspect historical inputs and instructions for distinguishable R-1 drift."""
import argparse
import csv
from pathlib import Path

from resources_sm90d1_sass import inspect
from sm90d1_common import ROOT, RUN, read, sha, write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-table', type=Path,
                        default=RUN/'collected-final/SM90D1_reference.csv')
    args = parser.parse_args()
    output = []
    for row in csv.DictReader(args.reference_table.open()):
        if (row['status']!='measured' or
                abs(float(row['difference_pct']))<=float(row['indistinguishable_pct'])):
            continue
        stem = f"{row['model']}_b{row['batch']}"
        config = read(ROOT/f'configs/serving/sm90r1/{stem}.json')
        original = ROOT/config['libraries']['decode']['path']
        rebuilt = RUN/f"build/reference2/{row['model']}-B{row['batch']}/decode.so"
        item = dict(config=stem, original=str(original), rebuilt=str(rebuilt),
                    difference_pct=row['difference_pct'], threshold_pct=row['indistinguishable_pct'])
        if not original.exists():
            item.update(status='historical_binary_unavailable', evidence='inspection unavailable')
            output.append(item)
            continue
        expected = config['libraries']['decode']['sha256']
        if sha(original)!=expected:
            raise ValueError('historical binary hash mismatch: '+str(original))
        old_plan, new_plan = read(str(original)+'.plan.json'), read(str(rebuilt)+'.plan.json')
        defines = lambda p: [s for s in Path(str(p)+'.cu').read_text().splitlines()
                             if s.startswith('#define ')]
        old_sass, new_sass = inspect(original), inspect(rebuilt)
        kernel = next(k for k in old_sass['instruction_encoding_sha256'] if 'l1_loop_kernel' in k)
        item.update(status='inspected',original_sha256=expected,rebuilt_sha256=sha(rebuilt),
            original_source_head=config['libraries']['decode']['source_head'],
            rebuilt_source_head=read(str(rebuilt)+'.identity.json')['source']['head'],
            gemm_geometry_identical=old_plan['gemms']==new_plan['gemms'],
            buffer_recipes_identical=old_plan['buffers']==new_plan['buffers'],
            compile_defines_identical=defines(original)==defines(rebuilt),
            l1_loop_instruction_encoding_identical=(old_sass['instruction_encoding_sha256'][kernel]==
                                                   new_sass['instruction_encoding_sha256'][kernel]),
            original_instruction_sha256=old_sass['instruction_encoding_sha256'][kernel],
            rebuilt_instruction_sha256=new_sass['instruction_encoding_sha256'][kernel],
            original_local=old_sass['functions'].get(kernel,{}),
            rebuilt_local=new_sass['functions'].get(kernel,{}),
            evidence='verified static comparison; cross-session drift not uniquely attributed')
        output.append(item)
    write(RUN/'reference-drift-audit.json',output)
    print(f'Inspected {len(output)} distinguishable reference cells')


if __name__ == '__main__':
    main()
