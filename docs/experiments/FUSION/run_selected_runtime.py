#!/usr/bin/env python3
"""Select, lower and verify real fusion with state-rotated production controls."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument("--out",type=Path,required=True)
parser.add_argument("--selected-plans",type=Path,
                    help="reuse verified selected MLIR; all binaries are rebuilt")
args=parser.parse_args()
repo=Path(__file__).resolve().parents[3]
out=args.out.resolve()
out.mkdir(exist_ok=False)
for folder in ("bin","src","build","logs","plans"):
    (out/folder).mkdir()
(out/".gitignore").write_text("bin/\n")
status=out/"status.txt"
status.write_text("RUNNING\n")
if any(key.startswith("TILEMEGA_") for key in os.environ):
    raise RuntimeError("remove inherited TILEMEGA overrides")

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def execute(command,log,timeout):
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
    builds=[]
    for model in ("gqa2","mha4"):
        source=repo/f"docs/experiments/SEQSCAN/raw/export/{model}.json"
        selected=out/"plans"/(model+".mlir")
        command=[str(repo/"build-portable/tools/tilemega-fusion-dp"),str(source),
                 str(repo/"configs/targets/sm_89.json"),
                 str(repo/f"docs/experiments/OCCUPANCY/raw/ptxas/{model}_occ1_full.build"),
                 "4","3",str(selected)]
        if args.selected_plans:
            saved=args.selected_plans.resolve()/(model+".mlir")
            selected.write_bytes(saved.read_bytes())
            (out/"build"/(model+"_selection.txt")).write_text(
                f"REUSED_SELECTED_CG path={saved} sha256={sha(saved)}\n")
        else:
            print(model,"selecting fusion intervals",flush=True)
            execute(command,out/"build"/(model+"_selection.txt"),3600)
        for state in (0,1):
            tag=f"{model}_f{state}"
            cuda=out/"src"/(tag+".cu")
            execute([str(repo/"build-portable/tools/tilemega-compile"),
                     str(selected if state else source),str(cuda)],
                    out/"build"/(tag+"_codegen.txt"),1200)
            binary=out/"bin"/tag
            command=["/usr/local/cuda/bin/nvcc","-std=c++17","-O2","-arch=sm_89","-lineinfo",
                     "-Xptxas=-v,--warn-on-spills","--expt-relaxed-constexpr",
                     *["-I"+str(repo/p) for p in ("include","third_party/cutlass/include",
                        "third_party/cutlass/tools/util/include","third_party/cutlass/test")],
                     str(cuda),str(repo/"build-portable/libtilemega.a"),
                     "/tmp/tilemega-vf-build/llvm-project/lib/libLLVMSupport.a",
                     "/tmp/tilemega-vf-build/llvm-project/lib/libLLVMDemangle.a",
                     *[str(repo/p) for p in ("build-barvinok/.libs/libbarvinok.a",
                        "build-isl/.libs/libisl.a","build-polylib/.libs/libpolylibgmp.a")],
                     "-lntl","-lgmp","-lz","-lrt","-ldl","-lm",
                     "-L/usr/local/cuda/lib64","-lcudart","-o",str(binary)]
            execute(command,out/"build"/(tag+"_ptxas.txt"),600)
            builds.append(dict(tag=tag,command=command,source_sha256=sha(cuda),binary_sha256=sha(binary)))
            print(tag,"built",flush=True)
    (out/"manifest.json").write_text(json.dumps(builds,indent=2)+"\n")
    cases=[(model,seq,state) for model in ("gqa2","mha4") for seq in (4,128) for state in (0,1)]
    fields=["round","execution_index","model","seq","state","l05_ms","l1_ms","l2_ms",
            "resource_json","schedule_json","hash_json"]
    with (out/"paired.tsv").open("x") as stream:
        writer=csv.DictWriter(stream,fields,delimiter="\t",lineterminator="\n")
        writer.writeheader()
        for round_id in range(50):
            shift=round_id%len(cases)
            hashes={}
            for index,(model,seq,state) in enumerate(cases[shift:]+cases[:shift]):
                tag=f"{model}_f{state}"
                run=subprocess.run([str(out/"bin"/tag),str(repo/f"docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3")],
                    cwd=repo,text=True,capture_output=True,timeout=180,
                    env=dict(os.environ,TILEMEGA_WARMUP="5",TILEMEGA_REPEAT="11",TILEMEGA_ISL_AUDIT="1"))
                text=run.stdout+run.stderr
                (out/"logs"/f"{tag}_s{seq}_r{round_id:03d}.txt").write_text(text)
                run.check_returncode()
                if "RESULT status=PASS" not in text:
                    raise RuntimeError("full-model fusion correctness did not pass")
                current=record(text,"E2E_HASH")
                if len(set(current.values()))!=1:
                    raise RuntimeError("inter-level fusion outputs differ")
                key=(model,seq)
                if key in hashes and hashes[key]!=current:
                    raise RuntimeError("fusion changes baseline output bits")
                hashes[key]=current
                times=record(text,"E2E_TIME")
                writer.writerow(dict(round=round_id,execution_index=index,model=model,seq=seq,state=state,
                    **{name:times[name] for name in ("l05_ms","l1_ms","l2_ms")},
                    resource_json=json.dumps(record(text,"E2E_RESOURCE")),
                    schedule_json=json.dumps(record(text,"E2E_SCHEDULE")),hash_json=json.dumps(current)))
                stream.flush()
            if (round_id+1)%10==0: print(f"{(round_id+1)*8}/400 fresh production processes PASS",flush=True)
    status.write_text("PASS 400/400 BF16 selected RoPE/KV fusion/control processes; GEMM fusion sign gate separate\n")
except BaseException as error:
    status.write_text(f"STOP {error}\n")
    raise
