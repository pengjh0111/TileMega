"""Pinned attention variants for the integrated decode selection objective."""
import csv
import itertools
import json
from pathlib import Path


def variants(manifest, include_hopper=False):
    """Execution modes are orthogonal; each tuple names one compiled artifact."""
    capacity=int(manifest['capacity'])
    extents=sorted(set((32,64,128,256,512,capacity)))
    paged=manifest['pg']=='pages'
    impls=['mma16','pvswap','pvswap2']
    if 1<=int(manifest.get('attention_q_per_kv',0))<=4 and int(manifest.get('attention_head_dim',0)) in (64,128):
        impls.append('simt')
    result = [dict(ec=ec,attention_impl=impl,attention_page='chunk16',nonpaged_la=la,paged_attention_load='loader')
              for ec,impl,la in itertools.product(extents,impls,(0,) if paged else (0,1))]
    caps=manifest.get('target_spec',{}).get('caps',{})
    d=int(manifest.get('attention_head_dim',0))
    if include_hopper and caps.get('wgmma') and manifest.get('arch_paths','auto')=='auto' and \
            1<=int(manifest.get('attention_q_per_kv',0))<=8 and d in (64,128):
        result += [dict(ec=ec,attention_impl='wgmma',attention_page='uniform',nonpaged_la=0,
                        paged_attention_load='loader',page_bytes=4*d*64) for ec in extents]
    return result


def matches(manifest,variant):
    return (manifest.get('merge','stage')==variant.get('merge','stage') and
            (not variant.get('cluster_dim') or int(manifest.get('cluster_dim',1))==variant['cluster_dim']) and
            (not variant.get('page_bytes') or manifest.get('pg')!='pages' or
             int((manifest.get('pages') or {}).get('page_bytes',0))==variant['page_bytes']) and
            int(manifest['attention_kv_block'])==variant['ec'] and
            manifest.get('attention_impl','mma16')==variant['attention_impl'] and
            manifest.get('attention_page','uniform' if manifest.get('attention_impl')=='wgmma' else 'chunk16')==
                variant.get('attention_page','uniform' if variant['attention_impl']=='wgmma' else 'chunk16') and
            int(manifest.get('nonpaged_la',False))==variant['nonpaged_la'] and
            manifest.get('paged_attention_load','loader')==variant['paged_attention_load'])


def label(variant):
    return f"ec{variant['ec']}-{variant['attention_impl']}-la{variant['nonpaged_la']}-{variant['paged_attention_load']}"


def pinned_geometry(manifest,class_file,directory):
    """Read the measured winner, never the predicted classes.tsv geometry."""
    gemms={g['index']:g for g in manifest['gemms']};groups={}
    with Path(class_file).open() as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            groups.setdefault(int(row['class']),[]).append(int(row['gemm']))
    if sorted(groups)!=list(range(len(groups))) or sorted(gemms)!=sorted(g for members in groups.values() for g in members):
        raise ValueError('pinned attention variant has incomplete GEMM class coverage')
    fields=('tile_m','tile_n','tile_k','stages','split_k')
    shapes=[]
    for _,members in sorted(groups.items()):
        values=[dict({k:int(gemms[g][k]) for k in fields},impl=gemms[g].get('impl','mma16')) for g in members]
        if any(v!=values[0] for v in values):raise ValueError('winner geometry differs within semantic class')
        shapes.append(values[0])
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    (directory/'domain.json').write_text(json.dumps(dict(geometries=shapes))+'\n')
    (directory/'cases.json').write_text(json.dumps(dict(cases=[dict(geometries=shapes,
        kappa=manifest['kappa'],residency=manifest['residency'])]))+'\n')
    return shapes


def compile_options(original,variant,directory):
    """Preserve target, packing and geometry while replacing structural flags."""
    directory=Path(directory)
    replace={'--serve-kv-block':variant['ec'],'--attention-impl':variant['attention_impl'],
        '--attention-page':variant.get('attention_page','uniform' if variant['attention_impl']=='wgmma' else 'chunk16'),
        '--nonpaged-la':variant['nonpaged_la'],'--search-passes':1,'--top-m':1,'--measure-top':1,
        '--search-domain':directory/'domain.json','--evaluate-configs':directory/'cases.json',
        '--candidate-mode':'L1','--candidate-loop':0,'--dump-cg':directory/'selected.mlir'}
    if variant.get('page_bytes'):
        replace['--page-bytes']=variant['page_bytes']
    # The assisted transport enters this list only after its conditional gate.
    if variant['paged_attention_load']!='loader':
        replace['--paged-attention-load']=variant['paged_attention_load']
    remove=set(replace)|{'--serving-warm-start','--paged-seed-from'}
    result=[str(original[0]),str(directory/'plan.so')]
    if (len(original)-2)%2:raise ValueError('expected compile option/value pairs')
    for key,value in zip(original[2::2],original[3::2]):
        if key not in remove:result.extend((str(key),str(value)))
    for key,value in replace.items():result.extend((key,str(value)))
    return result


def cluster_merge_variants(manifest):
    """Explicit H4d seed candidates; ordinary/default search remains unchanged."""
    target=manifest.get('target_spec',{})
    caps=target.get('caps',{})
    if manifest.get('phase','decode')!='decode' or manifest.get('pg') not in ('off','l2') or \
            manifest.get('arch_paths','auto')!='auto' or not caps.get('wgmma') or not caps.get('dsmem') or \
            not 1<=int(manifest.get('attention_q_per_kv',0))<=8 or \
            int(manifest.get('attention_head_dim',0)) not in (64,128):
        return []
    capacity=int(manifest['capacity'])
    result=[]
    for ranks in (2,4):
        ec=((capacity+ranks*64-1)//(ranks*64))*64
        if (capacity+ec-1)//ec!=ranks:continue
        result.append(dict(ec=ec,attention_impl='wgmma',attention_page='uniform',
            nonpaged_la=1,paged_attention_load='loader',merge='cluster',cluster_dim=ranks,
            l2_prefetch_depth=0,combine='stage'))
    return result


def cluster_merge_resources(head_dim,queries,ranks,attention_shared_bytes):
    """Exact storage/traffic, with latency explicitly unfitted until guarded calibration."""
    if head_dim not in (64,128) or not 1<=queries<=8 or ranks not in (2,4):
        raise ValueError('illegal cluster attention shape')
    elements=queries*head_dim+((queries+3)&~3)
    # Storage: full[2][r-1], ready/ack[2], episode, 16B-align, two peer slots.
    metadata=2*(ranks-1)*8+4*8+4
    mailbox=(( ((metadata+15)//16)*16+2*(ranks-1)*elements*4 +1023)//1024)*1024
    return dict(mailbox_bytes=mailbox,shared_bytes=((attention_shared_bytes+1023)//1024)*1024+mailbox,
        peer_bytes_per_episode=(ranks-1)*elements*4,rank_order=list(range(ranks)),
        latency_ns=None,latency_evidence='not_calibrated',storage_evidence='inferred_from_cpp_layout')
