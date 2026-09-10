#!/usr/bin/env python3
"""Real two-stage CG fusion calibration through the production harness."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import torch

parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--export",type=Path,required=True)
parser.add_argument("--tile-m",type=int,required=True)
args=parser.parse_args()
repo=Path(__file__).resolve().parents[3]
out=args.out.resolve()
out.mkdir(exist_ok=False)
(out/".gitignore").write_text("bin/\nfixture/\n")
for name in ("bin","fixture","logs","build"):
    (out/name).mkdir()
status=out/"status.txt"
status.write_text("RUNNING\n")
torch.set_num_threads(1)

def run(command,log,timeout=600):
    result=subprocess.run(command,cwd=repo,text=True,capture_output=True,timeout=timeout,
                          env=dict(os.environ,TILEMEGA_ISL_AUDIT="1"))
    log.write_text(result.stdout+result.stderr)
    result.check_returncode()
    return result

def record(text,label):
    lines=[line for line in text.splitlines() if line.startswith(label+" ")]
    if len(lines)!=1:
        raise RuntimeError("missing or ambiguous "+label)
    return dict(re.findall(r"(\w+)=([^\s]+)",lines[0]))

try:
    manifests=[]
    for consumer in ("add","norm"):
        chain=out/consumer
        run([str(repo/"build-portable/tools/tilemega-fusion-chain"),str(args.export.resolve()),
             consumer,str(args.tile_m),str(chain)],out/"build"/(consumer+"_cg.txt"))
        shape=json.loads((chain/"shape.json").read_text())
        if shape["dtype"]!="bf16":
            raise RuntimeError("this calibration gate requires BF16 source dimensions")
        n,k=shape["n"],shape["k"]
        for seq in (4,128):
            fixture=out/"fixture"/f"{consumer}_s{seq}"
            fixture.mkdir()
            generator=torch.Generator().manual_seed(20260910+seq)
            a=(torch.randn(seq,k,generator=generator)*0.2).bfloat16()
            b=(torch.randn(n,k,generator=generator)*0.2).bfloat16()
            c=(torch.randn(seq,n,generator=generator)*0.2).bfloat16() if consumer=="add" else (
                1+torch.randn(n,generator=generator)*0.1).bfloat16()
            intermediate=torch.matmul(a,b.T)
            reference=(intermediate.float()+c.float()).bfloat16() if consumer=="add" else (
                (intermediate.float()*torch.rsqrt(intermediate.float().square().mean(-1,keepdim=True)+1e-6)).bfloat16().float()*c.float()).bfloat16()
            for name,value in (("input",a),("weight",b),("consumer_input",c),("reference",reference)):
                (fixture/(name+".bin")).write_bytes(bytes(value.contiguous().untyped_storage()))
            (fixture/"manifest.json").write_text(json.dumps(dict(seq=seq,past=3))+"\n")
            manifests.append(dict(consumer=consumer,seq=seq,n=n,k=k,seed=20260910+seq,
                golden="PyTorch CPU BF16 matmul; FP32 add or RMS reduction; BF16 output",
                torch_version=torch.__version__,reference_sha256=hashlib.sha256((fixture/"reference.bin").read_bytes()).hexdigest()))
        for state in ("separate","fused"):
            tag=consumer+"_"+state
            binary=out/"bin"/tag
            command=["/usr/local/cuda/bin/nvcc","-std=c++17","-O2","-arch=sm_89","-lineinfo",
                "-Xptxas=-v,--warn-on-spills","--expt-relaxed-constexpr",
                *["-I"+str(repo/p) for p in ("include","third_party/cutlass/include",
                    "third_party/cutlass/tools/util/include","third_party/cutlass/test")],
                str(chain/(state+".cu")),str(repo/"build-portable/libtilemega.a"),
                "/tmp/tilemega-vf-build/llvm-project/lib/libLLVMSupport.a",
                "/tmp/tilemega-vf-build/llvm-project/lib/libLLVMDemangle.a",
                *[str(repo/p) for p in ("build-barvinok/.libs/libbarvinok.a","build-isl/.libs/libisl.a",
                                       "build-polylib/.libs/libpolylibgmp.a")],
                "-lntl","-lgmp","-lz","-lrt","-ldl","-lm","-L/usr/local/cuda/lib64","-lcudart","-o",str(binary)]
            run(command,out/"build"/(tag+"_ptxas.txt"))
            manifests.append(dict(tag=tag,command=command,binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest()))
            print(tag,"built",flush=True)
    (out/"manifest.json").write_text(json.dumps(manifests,indent=2)+"\n")
    cases=[(consumer,seq,state) for consumer in ("add","norm") for seq in (4,128) for state in ("separate","fused")]
    fields=["round","execution_index","consumer","seq","state","l05_ms","l1_ms","l2_ms",
            "resource_json","schedule_json","hash_json"]
    with (out/"paired.tsv").open("x") as stream:
        writer=csv.DictWriter(stream,fields,delimiter="\t",lineterminator="\n")
        writer.writeheader()
        for round_id in range(50):
            offset=round_id%len(cases)
            hashes={}
            for index,(consumer,seq,state) in enumerate(cases[offset:]+cases[:offset]):
                tag=consumer+"_"+state
                result=subprocess.run([str(out/"bin"/tag),str(out/"fixture"/f"{consumer}_s{seq}")],
                    cwd=repo,text=True,capture_output=True,timeout=180,
                    env=dict(os.environ,TILEMEGA_WARMUP="5",TILEMEGA_REPEAT="11",TILEMEGA_ISL_AUDIT="1"))
                text=result.stdout+result.stderr
                (out/"logs"/f"{tag}_s{seq}_r{round_id:03d}.txt").write_text(text)
                result.check_returncode()
                if "RESULT status=PASS" not in text:
                    raise RuntimeError("chain correctness failed")
                current=record(text,"E2E_HASH")
                if len(set(current.values()))!=1:
                    raise RuntimeError("chain execution levels differ")
                key=(consumer,seq)
                if key in hashes and hashes[key]!=current:
                    raise RuntimeError("shared fusion changed separate output bits")
                hashes[key]=current
                times=record(text,"E2E_TIME")
                writer.writerow(dict(round=round_id,execution_index=index,consumer=consumer,seq=seq,state=state,
                    **{key:times[key] for key in ("l05_ms","l1_ms","l2_ms")},
                    resource_json=json.dumps(record(text,"E2E_RESOURCE")),
                    schedule_json=json.dumps(record(text,"E2E_SCHEDULE")),hash_json=json.dumps(current)))
                stream.flush()
            if (round_id+1)%10==0:
                print(f"{(round_id+1)*8}/400 fresh BF16 chain processes PASS",flush=True)
    status.write_text("PASS 400/400 BF16 chain processes; model prediction sign analysis separate\n")
except BaseException as error:
    status.write_text(f"STOP {error}\n")
    raise
