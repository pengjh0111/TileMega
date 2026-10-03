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
