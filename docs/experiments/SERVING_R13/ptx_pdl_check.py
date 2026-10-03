#!/usr/bin/env python3
"""Fail closed on missing entry-point boundaries or incorrectly placed PDL."""
import argparse,json,re
from pathlib import Path
def entries(text):
    out={}
    for m in re.finditer(r'\.entry\s+(\S+)\s*\(',text):
        begin=text.find('{',m.end());depth=1;end=begin+1
        while depth and end<len(text):
            depth+=(text[end]=='{')-(text[end]=='}');end+=1
        out[m.group(1)]=text[begin:end]
    return out
def check(text,trigger):
    rows=[]
    for wanted in ('tilemega_l1_kernel','tilemega_l2_kernel','tilemega_l1_loop_kernel'):
        found=[(name,body) for name,body in entries(text).items() if wanted in name]
        if len(found)!=1:raise ValueError('missing or ambiguous kernel '+wanted)
        name,body=found[0];wait=body.find('griddepcontrol.wait')
        early=body.find('TILEMEGA_PDL_IMMUTABLE_PREFETCH_BEGIN')
        mutable=body.find('TILEMEGA_PDL_MUTABLE_ACCESS_BEGIN')
        launch=[m.start() for m in re.finditer('griddepcontrol.launch_dependents',body)]
        if not (0<=early<wait<mutable) or len(launch)!=1:raise ValueError('PDL boundary missing: '+name)
        # Before the annotated immutable-only descriptor/weight region, there
        # must be no global access (loading the Params pointer itself is param).
        global_access=r'\b(?:ld|st|atom|red)(?:\.[\w:]+)*\.global\b'
        if re.search(global_access,body[:early]):raise ValueError('global access before immutable region: '+name)
        prewait=body[early:wait]
        if re.search(r'\b(?:st|atom|red)(?:\.[\w:]+)*\.global\b',prewait):
            raise ValueError('global write before dependency wait: '+name)
        if re.search(r'\bcall(?:\.[\w:]+)*\b',prewait):
            raise ValueError('out-of-line prefetch must be audited before approving '+name)
        if trigger==1 and not early<launch[0]<wait:raise ValueError('early trigger misplaced: '+name)
        if trigger==0 and launch[0]<mutable:raise ValueError('late trigger misplaced: '+name)
        # The last trigger must follow all body global writes/atomics.
        if trigger==0 and re.search(r'\b(?:st|atom|red)(?:\.[\w:]+)*\.global\b',body[launch[0]:]):
            raise ValueError('publication follows late trigger: '+name)
        rows.append(dict(kernel=name,wait_offset=wait,trigger_offset=launch[0],pass_=True,
                         immutable_region=prewait,qualification='region is restricted by ServingPdlEnter to descriptor loads and NextStage weight prefetch; source audit required'))
    return rows
def main():
    p=argparse.ArgumentParser();p.add_argument('ptx',type=Path);p.add_argument('--trigger',type=int,choices=(0,1),default=0)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    try:r=dict(pass_=True,kernels=check(a.ptx.read_text(),a.trigger))
    except ValueError as error:r=dict(pass_=False,error=str(error))
    a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r));raise SystemExit(0 if r['pass_'] else 1)
if __name__=='__main__':main()
