#!/usr/bin/env python3
"""Capture a complete ctest baseline or reject failures added after a repair."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import xml.etree.ElementTree as ET


def classify(tests):
    failed, not_run, skipped = [], [], []
    for test in tests:
        name = test.attrib['name']
        skip = test.find('skipped')
        if test.find('failure') is not None or test.find('error') is not None:
            failed.append(name)
        elif skip is not None and skip.get('message') == 'SKIP_RETURN_CODE=77':
            skipped.append(name)
        elif test.get('status') == 'notrun' or skip is not None:
            # CTest also encodes missing executables as <skipped>. They must
            # remain failures for the no-new-failure comparison.
            not_run.append(name)
    return dict(failed=failed, not_run=not_run, skipped=skipped,
                unsuccessful=sorted(failed + not_run))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--baseline',type=Path)
    p.add_argument('--expected-tests',type=int,default=315)
    a=p.parse_args()
    from tilemega.serving.baseline_common import require_guard
    gpu=int(require_guard(tp=1)[0])
    if gpu==6:raise ValueError('GPU6 is excluded')
    from dnn1_baselines import passive_context_drain,sha
    a.out.mkdir(parents=True,exist_ok=False)
    command=['ctest','--test-dir',str(a.build.resolve()),'--output-on-failure',
             '--output-junit',str((a.out/'ctest.xml').resolve()),'-j','1']
    started=time.time()
    try:
        with (a.out/'ctest.log').open('w') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        tests=ET.parse(a.out/'ctest.xml').getroot().findall('.//testcase')
        if len(tests)!=a.expected_tests:
            raise RuntimeError(f'incomplete ctest suite: {len(tests)} != {a.expected_tests}')
        counts=classify(tests)
        baseline=json.loads(a.baseline.read_text()) if a.baseline else None
        if baseline and 'unsuccessful' not in baseline:
            raise ValueError('baseline must distinguish missing executables from intentional skips')
        added=sorted(set(counts['unsuccessful'])-set(baseline['unsuccessful'])) if baseline else None
        record=dict(evidence='verified complete ctest execution; baseline capture is not a passing-suite claim',
            status=('regression' if added else 'no_new_failures') if baseline else 'baseline_recorded',
            command=command,build=str(a.build.resolve()),gpu=gpu,pid=os.getpid(),started=started,
            ended=time.time(),tests=len(tests),ctest_exit_code=result.returncode,
            **counts,added_failures=added,
            baseline_sha256=sha(a.baseline) if a.baseline else None,
            junit_sha256=sha(a.out/'ctest.xml'),runner_sha256=sha(__file__),
            exit_context_drain=passive_context_drain(dict(gpu=gpu)))
        (a.out/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k!='exit_context_drain'}),flush=True)
        return 1 if added else 0
    except BaseException:
        passive_context_drain(dict(gpu=gpu))
        raise


if __name__=='__main__':raise SystemExit(main())
