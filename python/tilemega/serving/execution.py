"""Execution choices live beside immutable, content-addressed plan manifests."""
from pathlib import Path
import json
import warnings

CLI_FEATURES=frozenset(('decode_executor','decode_loop','prefill_executor'))

def compiler_features(features):
    return {k:v for k,v in features.items() if k not in CLI_FEATURES}

def read_execution(binary):
    path=Path(str(binary)+'.serving.json')
    if not path.exists():return None
    data=json.loads(path.read_text())
    if data.get('decode_mode') not in ('L1','L2') or data.get('prefill_mode') not in ('L1','L2') or data.get('decode_loop') not in (0,1,False,True):
        raise ValueError('invalid serving execution sidecar: '+str(path))
    return data

def write_execution(binary,decode_mode,decode_loop,prefill_mode,**evidence):
    if decode_mode not in ('L1','L2') or prefill_mode not in ('L1','L2') or decode_loop not in (0,1):
        raise ValueError('invalid serving execution choice')
    path=Path(str(binary)+'.serving.json')
    data=dict(decode_mode=decode_mode,decode_loop=int(decode_loop),prefill_mode=prefill_mode,**evidence)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
    return data

def resolve_execution(binary,mode,loop,prefill_mode,paged):
    needs_auto=mode=='auto' or loop=='auto' or prefill_mode=='auto'
    data=read_execution(binary) if needs_auto else None
    if needs_auto and data is None:
        warnings.warn('serving sidecar missing; auto uses legacy L2 and paged loop defaults',RuntimeWarning,stacklevel=2)
    if mode=='auto':mode=data['decode_mode'] if data else 'L2'
    if loop=='auto':loop=data['decode_loop'] if data else int(paged)
    if prefill_mode=='auto':prefill_mode=data['prefill_mode'] if data else mode
    if prefill_mode is None:prefill_mode=mode
    if mode not in ('L1','L2') or prefill_mode not in ('L1','L2') or loop not in (0,1,False,True):
        raise ValueError('invalid serving execution options')
    return mode,bool(loop),prefill_mode

def execution_combinations(pg, executor="measure", loop="measure"):
    """Only combinations supported by the two serving executors are eligible."""
    valid=[("L1",0),("L2",0),("L2",1) if pg=="pages" else ("L1",1)]
    return [(mode,used) for mode,used in valid
            if executor in ("measure",mode) and loop in ("measure",used)]

def select_execution(candidates):
    import statistics
    if not candidates:
        raise ValueError("no eligible serving execution combination")
    valid=[c for c in candidates if len(c["samples_ms"])==3 and not c.get("error")]
    if not valid:
        raise RuntimeError("no serving execution combination passed three rounds")
    for c in valid:
        c["median_ms"]=statistics.median(c["samples_ms"])
    return min(valid,key=lambda c:(c["median_ms"],c["pg"],c["mode"],c["loop"]))

def pin_prefill(manifest, class_file, directory):
    """Lock the preregistered prefill winner by actual per-GEMM geometry."""
    import csv
    data=json.loads(Path(manifest).read_text());gemms={g['index']:g for g in data['gemms']}
    groups={}
    with Path(class_file).open() as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            groups.setdefault(int(row['class']),[]).append(int(row['gemm']))
    if sorted(groups)!=list(range(len(groups))) or set(gemms)!={g for v in groups.values() for g in v}:
        raise ValueError('prefill pin class coverage differs from manifest')
    shapes=[];fields=('tile_m','tile_n','tile_k','stages','split_k')
    for _,members in sorted(groups.items()):
        values=[{k:int(gemms[g][k]) for k in fields} for g in members]
        if any(v!=values[0] for v in values):raise ValueError('prefill pin differs within class')
        shapes.append(values[0])
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'prefill_domain.json').write_text(json.dumps(dict(geometries=shapes))+'\n')
    (directory/'prefill_cases.json').write_text(json.dumps(dict(cases=[dict(geometries=shapes,
        kappa=data['kappa'],residency=data['residency'])]))+'\n')
    return ['--search-domain',str(directory/'prefill_domain.json'),
            '--evaluate-configs',str(directory/'prefill_cases.json'),
            '--serve-kv-block',str(data['attention_kv_block']),
            '--serve-query-rows',str(data['attention_query_rows'])]
