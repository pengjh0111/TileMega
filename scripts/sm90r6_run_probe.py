#!/usr/bin/env python3
"""Run one isolated native probe in a fresh guarded process and retain its ledger."""
import argparse
import csv
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import statistics


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_part(binary,ledger_path,batch,prompts,out,development,prefilled_kv=None,reference=None):
    ledger=json.loads(ledger_path.read_text())
    if not development:
        identity=json.loads(Path(str(binary)+'.probe-identity.json').read_text())
        if (identity['binary_sha256']!=sha(binary) or identity['ledger_sha256']!=sha(ledger_path) or
                identity['source_sha256']!=sha(str(binary)+'.cu') or not identity['inputs_unchanged'] or
                identity['entry_instructions']>identity['limit'] or
                identity['input_hashes_sha256']!=sha(identity['input_hashes'])):
            raise ValueError('probe artifact differs from its stable build or instruction limit')
    import gc
    import torch
    from tilemega.serving.engine import ServingEngine
    checkpoint=Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT'])
    out.mkdir(parents=True,exist_ok=False)
    prompt=torch.tensor(json.loads(prompts.read_text())[:batch],dtype=torch.int32)
    if tuple(prompt.shape)!=(batch,64):raise ValueError('probe requires B frozen ISL64 prompts')
    with ServingEngine(checkpoint,None,binary,batch,prompt_len=64,max_new_tokens=1,
                       mode='L1',decode_loop=False,step_events=False) as engine:
        if prefilled_kv is None:engine.state.kv_storage.zero_()
        else:
            if engine.state.kv_storage.shape!=prefilled_kv.shape:raise ValueError('A0 and probe KV layouts differ')
            engine.state.kv_storage.copy_(prefilled_kv)
        engine.state.tokens.zero_()
        engine.state.tokens[:,:64].copy_(prompt)
        engine.decode.set_steps([63])
        torch.cuda.synchronize()
        engine.decode.launch(0,1,torch.cuda.current_stream().cuda_stream)
        torch.cuda.synchronize()
        dump=engine.decode_lib.lib.tm_plan_dump_serving_trace
        dump.argtypes=[C.c_void_p,C.c_char_p];dump.restype=C.c_int
        if dump(engine.decode.handle,os.fsencode(out.resolve())):raise RuntimeError('probe trace dump failed')
        grid=engine.decode_lib.info.grid
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    by_stage={}
    with (out/'stage_trace.tsv').open() as stream:
        for row in csv.DictReader(stream,delimiter='\t'):
            if int(row['iteration'])!=0 or int(row['past'])!=63:raise ValueError('unexpected probe launch in trace')
            by_stage.setdefault(int(row['stage']),[]).append(row)
    expected={s for v in ledger['variants'] for s in v['stages']}
    if set(by_stage)!=expected:raise ValueError('probe ledger does not cover exactly the measured instances')
    measurements=[]
    for variant in ledger['variants']:
        samples=[]
        for stage in variant['stages']:
            rows=by_stage[stage]
            if len(rows)!=grid or {int(r['worker']) for r in rows}!=set(range(grid)):
                raise ValueError('probe stage lost or duplicated CTA records')
            if any(not 0<int(r['t_begin'])<int(r['t_tasks_end'])<=int(r['t_release']) for r in rows):
                raise ValueError('nonmonotonic stage ledger')
            # End-to-end completion of this isolated stage, including its V3
            # join; use one globaltimer domain across the full worker grid.
            samples.append(max(int(r['t_release']) for r in rows)-min(int(r['t_begin']) for r in rows))
        if len(samples)!=ledger['layers']:raise ValueError('probe did not repeat exactly L times')
        measurements.append(dict(family=ledger['family'],geometry=variant['geometry'],stage_ns=samples,
            median_ns=statistics.median(samples),process_id=f'{os.getpid()}:{out.resolve()}',
            fresh_process=True,binary_sha256=sha(binary),ledger_sha256=sha(ledger_path),
            gpu=int(os.environ['TILEMEGA_DEVICE_INDEX']),development=development,
            input_state='A0 prompt-prefilled KV' if prefilled_kv is not None else 'development zero KV',
            reference_sha256=sha(reference) if reference else None,
            raw=str((out/'stage_trace.tsv').resolve()),raw_sha256=sha(out/'stage_trace.tsv')))
    (out/'measurements.json').write_text(json.dumps(measurements,indent=2)+'\n')
    (out/'result.json').write_text(json.dumps(dict(passed=True,variants=len(measurements),grid=grid,
        stage_records=sum(map(len,by_stage.values())),prompt_sha256=sha(prompts),
        checkpoint=json.loads((checkpoint/'random_checkpoint.json').read_text())['checkpoint_sha256'],
        development=development),indent=2)+'\n')

    return measurements,json.loads((out/'result.json').read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--binary',type=Path)
    mode.add_argument('--manifest',type=Path)
    p.add_argument('--ledger',type=Path)
    p.add_argument('--reference',type=Path,help='A0 binary that produces the frozen prompt KV outside the timed probe')
    p.add_argument('--batch',type=int,required=True)
    p.add_argument('--prompts',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--development',action='store_true')
    a=p.parse_args()
    if a.binary and not a.ledger:p.error('--binary requires --ledger')
    if a.manifest and a.ledger:p.error('--manifest supplies each part ledger')
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD')!='1':raise ValueError('probe requires the physical GPU guard')
    if int(os.environ['TILEMEGA_DEVICE_INDEX']) not in (0,1,2,3,4,5,7):raise ValueError('probe timing excludes GPU6')
    if not a.development and a.reference is None:p.error('formal probes require the selected A0 input reference')
    prefilled_kv=None
    if a.reference:
        import gc
        import torch
        from tilemega.serving.engine import ServingEngine
        prompt=torch.tensor(json.loads(a.prompts.read_text())[:a.batch],dtype=torch.int32)
        with ServingEngine(Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT']),None,a.reference,a.batch,
                prompt_len=64,max_new_tokens=1,mode='L1',decode_loop=False,step_events=False) as engine:
            engine.state.kv_storage.zero_()
            engine.state.tokens.fill_(-1)
            generation=engine.generate(prompt,1)
            prefilled_kv=engine.state.kv_storage.clone()
            torch.cuda.synchronize()
        del generation,engine
        gc.collect()
        torch.cuda.empty_cache()
    if a.binary:
        run_part(a.binary,a.ledger,a.batch,a.prompts,a.out,a.development,prefilled_kv,a.reference)
        return
    manifest=json.loads(a.manifest.read_text());a.out.mkdir(parents=True,exist_ok=False)
    measurements=[];parts=[];geometries=set()
    for index,part in enumerate(manifest['parts']):
        if sha(part['binary'])!=part['binary_sha256']:raise ValueError('probe manifest binary changed')
        rows,result=run_part(Path(part['binary']),Path(part['ledger']),a.batch,a.prompts,
                             a.out/f'part-{index}',a.development,prefilled_kv,a.reference)
        for row in rows:
            if row['family']!=manifest['family']:raise ValueError('probe manifest mixes families')
            key=json.dumps(row['geometry'],sort_keys=True)
            if key in geometries:raise ValueError('probe manifest duplicates a geometry')
            geometries.add(key)
        measurements.extend(rows);parts.append(result)
    if len(measurements)!=manifest['geometries']:raise ValueError('probe manifest omits a geometry')
    checkpoints={r['checkpoint'] for r in parts}
    if len(checkpoints)!=1:raise ValueError('probe checkpoint changed between parts')
    (a.out/'measurements.json').write_text(json.dumps(measurements,indent=2)+'\n')
    (a.out/'result.json').write_text(json.dumps(dict(passed=True,variants=len(measurements),
        parts=len(parts),checkpoint=checkpoints.pop(),prompt_sha256=sha(a.prompts),
        reference_sha256=sha(a.reference) if a.reference else None,
        development=a.development),indent=2)+'\n')


if __name__=='__main__':main()
