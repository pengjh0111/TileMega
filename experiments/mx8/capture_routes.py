"""Capture one common 1024-position trajectory, not engine-specific DRAM traffic."""
import argparse
import ctypes as C
import json
from pathlib import Path
import sys
sys.path=[p for p in sys.path if Path(p or '.').resolve()!=Path(__file__).resolve().parent]
import numpy as np
import torch
from tilemega.build.identity import sha,verify
from tilemega.serving.engine import ServingEngine


def read(plan,layers,batch):
    result=np.zeros((1024,layers,batch,8),dtype=np.uint32)
    function=plan.library.lib.tm_plan_read_buffer_named
    function.argtypes=[C.c_void_p,C.c_char_p,C.c_void_p,C.c_uint64];function.restype=C.c_int
    if function(plan.handle,b'mx.route_debug',result.ctypes.data,result.nbytes):
        raise ValueError('1024-position route read failed')
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cell',type=Path,required=True)
    parser.add_argument('--decode',type=Path,required=True)
    parser.add_argument('--prefill',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    cell=json.loads(args.cell.read_text());config=json.loads((Path(cell['model_dir'])/'config.json').read_text())
    layers=config['num_hidden_layers'];batch=cell['batch'];experts=config.get('num_local_experts',config.get('num_experts'))
    for p in (args.decode,args.prefill):verify(p)
    prompts=torch.tensor(json.loads(Path(cell['prompt']).read_text())[:batch],dtype=torch.int32)
    with ServingEngine(cell['model_dir'],args.prefill,args.decode,batch,mode='L1',prefill_mode='L1',
            decode_loop=True,step_events=False) as engine:
        tokens=engine.generate(prompts).tokens
        routes=read(engine.decode,layers,batch)
        routes[0]=read(engine.prefill,layers,batch)[0]
        reference=json.loads(Path(cell['tokens']['path']).read_text())
        if not torch.equal(tokens[:,:len(reference[0])],torch.tensor(reference,dtype=torch.int32)):
            raise ValueError('DIAG reference trajectory differs from frozen main tokens')
    if not np.all(routes<experts) or np.any(np.diff(np.sort(routes,axis=-1),axis=-1)==0):
        raise ValueError('capture has missing or invalid top-8 entries')
    active=np.array([[len(np.unique(routes[s,l])) for l in range(layers)] for s in range(1024)])
    np.save(args.out/'routes.npy',routes)
    result=dict(model=cell['model'],batch=batch,positions=1024,layers=layers,
        position_semantics='prompt past63 and decode past64..1086; common TileMega reference used for all systems',
        active_experts_per_layer_mean=active.mean(axis=0).tolist(),active_experts_mean=float(active.mean()),
        minimum=int(active.min()),maximum=int(active.max()),
        capture_sha256=sha(args.out/'routes.npy'),decode_sha256=sha(args.decode),prefill_sha256=sha(args.prefill),
        frozen_tokens_exact=True)
    (args.out/'RESULT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,positions=1024,active_experts_mean=result['active_experts_mean'])))


if __name__=='__main__':main()
