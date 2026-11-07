#!/usr/bin/env python3
"""Pack audited F3 measurements into an isolated solve target, without fallback."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

ROOT=Path(__file__).resolve().parents[1]
UNITS={'f3_k64_ns':'ns','f3_epilogue_ns':'ns','f3_splitk_ns':'ns',
       'f3_attention_task_ns':'ns/task','f3_small_ns':'ns/work_item'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def selector(row):
    metric=row['metric']
    tile=dict(tm=str(row['tile_m']),tn=str(row['tile_n'])) if 'tile_m' in row else {}
    if metric=='f3_k64_ns':fields=dict(mode=row['mode'],**tile)
    elif metric=='f3_epilogue_ns':fields=dict(**{'class':row['class']},**tile)
    elif metric=='f3_splitk_ns':fields=dict(**tile,slices=str(row['slices']))
    elif metric=='f3_small_ns':fields=dict(kind=row['kind'])
    elif metric=='f3_attention_task_ns':fields={}
    else:raise ValueError('unknown F3 metric '+metric)
    return ';'.join(k+'='+fields[k] for k in sorted(fields))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,nargs='+',required=True,
                   help='explicit ordered result files; later repaired probes replace earlier duplicate keys')
    p.add_argument('--target',type=Path,default=ROOT/'configs/targets/sm_90_h800.json')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--dependency-audit',type=Path,
                   help='pre-change nvcc dependency proof for headers unused by every measured probe')
    args=p.parse_args()
    if args.out.resolve()==args.target.resolve():
        raise ValueError('pack to an isolated target; install only after the full20 solve/compile gate')
    dependency=None;covered={};unused_changes={}
    if args.dependency_audit:
        dependency=json.loads(args.dependency_audit.read_text())
        if (dependency['status']!='passed' or dependency['target_sha256']!=sha(args.target)
                or dependency['script_sha256']!=sha(ROOT/'scripts/dnn3_calibration_dependencies.py')):
            raise ValueError('dependency audit differs from the measured target or audit script')
        for row in dependency['rows']:
            if sha(Path(row['dependencies']))!=row['dependencies_sha256']:
                raise ValueError('raw nvcc dependency evidence changed')
            if any(sha(ROOT/'include'/name)!=digest for name,digest in row['headers'].items()):
                raise ValueError('a measured probe dependency changed; remeasure it')
            covered[(row['build_sha256'],row['binary_sha256'])]=row
    tables={};headers=None;sources=[];rejected=[];duplicates=[]
    for path in args.results:
        path=path.resolve();d=json.loads(path.read_text())
        if d['status']!='passed' or d['gpu']!=1:
            raise ValueError('require terminal five-process GPU1 calibration: '+str(path))
        guard=path.parent.parent/'guard_result.json'
        receipt=json.loads(guard.read_text())
        if receipt['code'] or receipt['acceptance'] not in ('accepted','provisional_container_pid_namespace'):
            raise ValueError('rejected calibration guard: '+str(guard))
        build_path=Path(d['build']);build=json.loads(build_path.read_text())
        if sha(build_path)!=d['build_sha256']:
            raise ValueError('calibration build receipt changed')
        if dependency and any((sha(build_path),p['binary_sha256']) not in covered
                              for p in build['probes']):
            raise ValueError('dependency audit omits a measured probe')
        current={}
        for name,digest in build['inputs'].items():
            source=Path(name)
            if '/include/' not in name:continue
            if sha(source)!=digest:
                relative=name.split('/include/',1)[1]
                if not dependency or any(relative in r['headers'] for r in dependency['rows']):
                    raise ValueError('calibration header changed: '+name)
                unused_changes[relative]=dict(measured_sha256=digest,current_sha256=sha(source))
            relative=name.split('/include/',1)[1]
            if relative in current and current[relative]!=digest:
                raise ValueError('duplicate header path with different contents')
            current[relative]=digest
        if headers is not None and headers!=current:
            raise ValueError('cannot blend different F3 header versions')
        headers=current
        source=dict(result=str(path.relative_to(ROOT)),sha256=sha(path),
                    build_sha256=sha(build_path),guard_sha256=sha(guard),method=d.get('method','see process records'))
        sources.append(source)
        rejected.extend(d.get('rejected',d.get('rejected_builds',
            build.get('rejected',build.get('rejected_builds',[])))))
        for row in d['rows']:
            processes=row['processes'];metric=row['metric'];key=selector(row)
            if row['passes']!='5/5' or len(processes)!=5:
                raise ValueError('requires exactly five fresh processes per key')
            medians=[]
            for process in processes:
                if 'rows' in process:
                    matches=[r for r in process['rows'] if r['metric']==metric]
                    if len(matches)!=1:raise ValueError('ambiguous process metric')
                    value=matches[0]['ns']
                else:value=process['ns']
                if not math.isfinite(value) or value<=0:raise ValueError('nonpositive calibration')
                medians.append(value)
            median=statistics.median(medians)
            if median!=row['ns']:raise ValueError('stored median differs from five raw process medians')
            table=tables.setdefault(metric,{})
            if key in table:duplicates.append(dict(metric=metric,key=key,old=table[key]['raw_source'],new=source['result']))
            table[key]=dict(median=median,processes=5,unit=UNITS[metric],
                unstable=(max(medians)-min(medians))/median>.1,
                fresh_process_medians=medians,binary_sha256=row['binary_sha256'],raw_source=source['result'])
    section=dict(schema=1,variant='dnn3-f3-no-call-predicate-v2',
        method='GPU1; median of five fresh-process medians; production regions; no interpolation',
        header_sha256=headers,raw_sources=sources,rejected_probe_builds=rejected,
        duplicate_resolution='last explicitly listed repaired result; no averaging or fastest selection',
        replaced_keys=duplicates,qualification='provisional_container_pid_namespace',
        by_gpu={'1':tables},coverage='measured keys only; full20 solve/compile acceptance is separate')
    if dependency:
        section['dependency_audit']=dict(source=str(args.dependency_audit.resolve()),
            sha256=sha(args.dependency_audit),probes=len(dependency['rows']),
            unused_header_changes=unused_changes)
    target=json.loads(args.target.read_text());native=target['sm90_native_constants']
    target['sm90_f3_constants']=section
    assert target['sm90_native_constants']==native
    args.out.parent.mkdir(parents=True,exist_ok=True)
    if args.out.exists():raise ValueError('retain previous table; choose a new output')
    args.out.write_text(json.dumps(target,indent=2)+'\n')
    print(json.dumps(dict(target=str(args.out),keys={k:len(v) for k,v in tables.items()},
        rejected_builds=len(rejected),duplicate_replacements=len(duplicates),full20_solve_compile='pending')))


if __name__=='__main__':
    main()
