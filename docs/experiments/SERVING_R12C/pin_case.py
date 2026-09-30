#!/usr/bin/env python3
"""Pin measured per-GEMM geometry to a DN-compatible class partition."""
import argparse,csv,json
from pathlib import Path
FIELDS=('tile_m','tile_n','tile_k','stages','split_k')
def classes(path):
    groups={}
    with Path(path).open() as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            groups.setdefault(int(row['class']),[]).append(int(row['gemm']))
    return {c:sorted(v) for c,v in sorted(groups.items())}

def pin(manifest,source_classes,target_classes,out,overrides=None):
    manifest=Path(manifest);data=json.loads(manifest.read_text());source=classes(source_classes);target=classes(target_classes)
    if sorted(source.values())!=sorted(target.values()):
        raise ValueError('source and DN-compatible target class GEMM sets differ')
    gemms={g['index']:g for g in data['gemms']}
    if set(gemms)!={g for members in target.values() for g in members}:raise ValueError('class partition does not cover manifest GEMMs')
    shapes=[]
    for c,members in target.items():
        if c!=len(shapes):raise ValueError('class IDs must be contiguous')
        values=[{k:int(gemms[g][k]) for k in FIELDS} for g in members]
        if any(g!=values[0] for g in values):raise ValueError(f'measured geometry differs within class {c}')
        shapes.append(values[0])
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    (out/'cases.json').write_text(json.dumps({'cases':[dict(geometries=shapes,kappa=data['kappa'],residency=data['residency'])]},indent=2)+'\n')
    unique={tuple(g[k] for k in FIELDS):g for g in shapes}
    (out/'domain.json').write_text(json.dumps(dict(geometries=list(unique.values())),indent=2)+'\n')
    pages=data.get('pages') or {};opts=dict(pg=data.get('pg','off'),sync=data.get('sync','legacy'),weight_layout='tiled' if data.get('pg')=='pages' else 'row',page_bytes=pages.get('page_bytes',16384),lookahead_bytes=pages.get('lookahead_bytes',0),serve_kv_block=data['attention_kv_block'],serve_query_rows=data['attention_query_rows'])
    opts.update(overrides or {});arguments=['--search-domain',str(out/'domain.json'),'--evaluate-configs',str(out/'cases.json')]
    for k,v in opts.items():arguments+=['--'+k.replace('_','-'),str(v)]
    record=dict(source_manifest=str(manifest),source_classes=str(source_classes),target_classes=str(target_classes),options=arguments,source_residency=data['residency'],source_grid=data['grid'],placeholder_measurement=True)
    (out/'record.json').write_text(json.dumps(record,indent=2)+'\n');return record

def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--classes',required=True);p.add_argument('--target-classes');p.add_argument('--out',required=True);p.add_argument('--overrides',default='{}');a=p.parse_args()
    print(json.dumps(pin(a.manifest,a.classes,a.target_classes or a.classes,a.out,json.loads(a.overrides))))
if __name__=='__main__':main()
