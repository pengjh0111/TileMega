#!/usr/bin/env python3
"""One guarded attribution launch from an identity-bound native DNN artifact."""
from __future__ import annotations
import argparse
import ctypes as C
import hashlib
import json
import os
from pathlib import Path
import sys
import traceback


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('library','export','bridge','inputs','out'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--logical-plan',type=Path)
    p.add_argument('--batch',type=int,choices=(1,2,4,8,16),required=True)
    p.add_argument('--warmup',type=int,choices=(50,),default=50)
    p.add_argument('--synthetic-weights',action='store_true',
                   help='same fixed synthetic state as check_generated; otherwise require a random-weight export')
    p.add_argument('--process-index',type=int,required=True)
    a=p.parse_args()
    from tilemega.serving.baseline_common import require_guard,observe
    physical=require_guard(tp=1)[0]
    if int(physical) not in (0,1,2,3,4,5,7):raise ValueError('DNN1 capture excludes GPU6')
    a.out.mkdir(parents=True,exist_ok=False)
    try:
        from dnn1_work import describe,generated_tables
        from dnn1_marks import ForwardTrace
        source_path=Path(str(a.library)+'.cu');manifest_path=Path(str(a.library)+'.plan.json')
        identity_path=Path(str(a.library)+'.identity.json')
        identity=json.loads(identity_path.read_text())
        if not identity['trace'] or identity['binary_sha256']!=sha(a.library):
            raise ValueError('capture requires the identified TRACE artifact, never an E2E timing artifact')
        root=Path(__file__).resolve().parents[1]
        for rel in ('include/tilemega/Backend/Sm90ForwardMark.cuh',
                    'include/tilemega/Backend/Sm90ForwardSmall.cuh'):
            if identity['source']['source_files'].get(rel)!=sha(root/rel):
                raise ValueError('trace observation header changed; rebuild the frozen attribution artifact: '+rel)
        work=describe(source_path,manifest_path,identity_path,a.logical_plan,a.inputs)
        if work['runtime_batch']!=a.batch:raise ValueError('artifact batch differs')
        (a.out/'work.json').write_text(json.dumps(work,indent=2)+'\n')
        descriptors,_=generated_tables(source_path.read_text())
        allocation=sum(b['width']*(b['constant']+a.batch*b['per_batch']) for b in descriptors)
        source_manifest=json.loads(manifest_path.read_text())
        if source_manifest.get('export_sha256') and source_manifest['export_sha256']!=sha(a.bridge):
            raise ValueError('bridge differs from the artifact compile input')
        export_manifest=json.loads((a.export/'manifest.json').read_text())
        if export_manifest['artifacts']['exported_program.pt2']!=sha(a.export/'exported_program.pt2'):
            raise ValueError('export archive hash differs')
        checkpoint_sha=None
        if not a.synthetic_weights:
            checkpoint_sha=sha(a.export/'checkpoint/model.safetensors')
            if export_manifest['checkpoint']['sha256']!=checkpoint_sha:
                raise ValueError('random checkpoint hash differs from export manifest')
        if not a.synthetic_weights and not export_manifest.get('random_weights',False):
            raise ValueError('DNN1 needs a random-weight export or explicit fixed synthetic state')
        if export_manifest['model']=='nafnet' and allocation>70_000_000_000:
            raise ValueError(f'STOP NAFNet pre-plan buffer estimate exceeds 70GB: {allocation}')
        import torch
        from safetensors.torch import load_file
        from tilemega.serving.plan import FORWARD,PlanLibrary
        from tilemega.serving.weights import load_weights,_packed_gpu
        from tilemega.dnn.check_generated import synthetic_state
        torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False
        observe(a.out/'guard.jsonl','before-initialization')
        library=PlanLibrary(a.library)
        if library.info.phase!=FORWARD or library.info.batch_lo!=a.batch or library.info.batch_hi!=a.batch:
            raise ValueError('trace plan has another execution domain')
        bridge=json.loads(a.bridge.read_text());nodes={n['name']:n for n in bridge['nodes']}
        signature=bridge['signature']
        inputs=[x['name'] for x in signature['inputs'] if x['kind']=='USER_INPUT']
        outputs=[x['name'] for x in signature['outputs'] if x['kind']=='USER_OUTPUT']
        supplied=load_file(str(a.inputs),device='cpu')
        if set(supplied)!=set(inputs):raise ValueError('frozen input names differ from bridge')
        if a.synthetic_weights:
            if export_manifest['model']=='bert':import transformers.modeling_outputs
            archive=a.export/'exported_program.pt2'
            if export_manifest['artifacts']['exported_program.pt2']!=sha(archive):
                raise ValueError('archive hash differs')
            module=torch.export.load(archive).module()
            state=synthetic_state(module)
            sources={name:value.cuda() for name,value in state.items()}
            tensors={b.name:_packed_gpu(json.loads(b.pack_json),sources.__getitem__).contiguous()
                     for b in library.buffers if b.pack_json}
            del sources
        else:tensors=load_weights(a.export/'checkpoint',library,device='cuda')
        for name in inputs:
            record=nodes[name];shape=(a.batch,*(int(x) for x in record['shape'][1:]))
            dtype={'torch.bfloat16':torch.bfloat16,'torch.int64':torch.int64}[record['dtype']]
            value=supplied[name]
            if tuple(value.shape)!=shape or value.dtype!=dtype:raise ValueError('frozen input differs: '+name)
            tensors[name]=value.cuda()
        for name in outputs:
            shape=(a.batch,*(int(x) for x in nodes[name]['shape'][1:]))
            tensors[name]=torch.empty(shape,dtype=torch.bfloat16,device='cuda')
        external={b.name for b in library.buffers if b.role==1}
        if external!=set(tensors):raise ValueError('trace bindings differ from native external buffers')
        trace=ForwardTrace(library.lib)
        if trace.info['stages']!=len(work['stages']) or trace.info['grid']!=work['grid']:
            raise ValueError('trace array dimensions differ from compiled work description')
        with library.create(a.batch,{name:value.data_ptr() for name,value in tensors.items()},0) as plan:
            plan.set_steps([0]);stream=torch.cuda.current_stream().cuda_stream
            for _ in range(a.warmup):plan.launch(0,1,stream)
            torch.cuda.synchronize()
            for name in outputs:tensors[name].fill_(float('nan'))
            trace.reset();observe(a.out/'guard.jsonl','before-trace')
            plan.launch(0,1,stream);torch.cuda.synchronize()
            after=observe(a.out/'guard.jsonl','after-trace')
            trace.save(a.out/'raw',metadata=dict(model=export_manifest['model'],batch=a.batch,
                gpu_index=int(physical),process_index=a.process_index,
                artifact_id=identity['artifact_id'],library_sha256=sha(a.library),
                source_sha256=sha(source_path),bridge_sha256=sha(a.bridge),input_sha256=sha(a.inputs),
                work_sha256=sha(a.out/'work.json'),warmup=50,after=after,
                small_phase_aggregation='first-to-last occurrence windows; interleaved phases are not additive'))
            finite={name:bool(torch.isfinite(tensors[name]).all().item()) for name in outputs}
        receipt=dict(evidence='verified attribution launch; no latency or correctness claim',
            model=export_manifest['model'],batch=a.batch,physical_gpu=int(physical),process_index=a.process_index,
            pid=os.getpid(),cpu_affinity=sorted(os.sched_getaffinity(0)),allocation_bytes=allocation,
            artifact_id=identity['artifact_id'],library_sha256=sha(a.library),identity_sha256=sha(identity_path),
            input_sha256=sha(a.inputs),bridge_sha256=sha(a.bridge),synthetic_weights=a.synthetic_weights,
            source_checkpoint_sha256=checkpoint_sha,archive_sha256=sha(a.export/'exported_program.pt2'),
            weight_seed=20261010 if a.synthetic_weights else export_manifest.get('initialization_seed'),
            output_finite=finite,correctness='not evaluated; known original batch defects remain visible in work.json',
            trace_directory=str((a.out/'raw').resolve()),work_json=str((a.out/'work.json').resolve()),
            command=[sys.executable,*sys.argv],timing_eligible=False)
        (a.out/'capture.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(receipt),flush=True)
    except BaseException as error:
        (a.out/'capture_error.json').write_text(json.dumps({'error_type':type(error).__name__,
            'error':str(error),'traceback':traceback.format_exc(),'command':[sys.executable,*sys.argv]},indent=2)+'\n')
        raise


if __name__=='__main__':
    from tilemega.serving.baseline_common import guarded_main
    guarded_main(main)
