#!/usr/bin/env python3
"""BF16 production placement controls, complete state rotation, fresh processes."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def record(text,label):
    rows=[dict(re.findall(r"(\w+)=([^\s]+)",line)) for line in text.splitlines()
          if line.startswith(label+" ")]
    if len(rows)!=1:
        raise RuntimeError(f"missing/ambiguous {label}")
    return rows[0]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--phase",choices=("build","run"),required=True)
    args=parser.parse_args()
    repo=Path(__file__).resolve().parents[3]
    out=args.out.resolve()
    if any(key.startswith("TILEMEGA_") for key in os.environ):
        raise RuntimeError("remove inherited TILEMEGA overrides")
    arms=[(m,s) for m in ("gqa2","mha4") for s in (0,4)]
    if args.phase=="build":
        out.mkdir(parents=True,exist_ok=False)
        for folder in ("src","bin","build","plans","logs"):
            (out/folder).mkdir()
        (out/".gitignore").write_text("bin/\n")
        (out/"source.diff").write_bytes(subprocess.check_output(["git","diff","--binary"],cwd=repo))
        builds=[]
        for model,state in arms:
            tag=f"{model}_p{state}"
            plan=dict(schema="tilemega.runtime_variants.v1",variants=[dict(
                seq_begin=1,seq_end=128,balanced_placement=state==4,
                uniform=dict(tile_m=128,tile_n=128,tile_k=16,stages=3,split_k=1))])
            plan_path=out/"plans"/(tag+".json")
            plan_path.write_text(json.dumps(plan,indent=2)+"\n")
            source=out/"src"/(tag+".cu")
            command=[str(repo/"build-portable/tools/tilemega-compile"),
                     str(repo/f"docs/experiments/SEQSCAN/raw/export/{model}.json"),
                     str(source),"--variants",str(plan_path)]
            with (out/"build"/(tag+"_codegen.txt")).open("w") as log:
                subprocess.run(command,stdout=log,stderr=log,check=True,timeout=600)
            binary=out/"bin"/tag
            command=["/usr/local/cuda/bin/nvcc","-std=c++17","-O2","-arch=native","-lineinfo",
                     "-Xptxas=-v,--warn-on-spills","--expt-relaxed-constexpr",
                     "-DTILEMEGA_FP32_PARTIALS=1","-DTILEMEGA_CG_SPLIT_TASK_ORDER=1",
                     f"-DTILEMEGA_PLACEMENT={state}",
                     *[f"-I{repo/p}" for p in ("include","third_party/cutlass/include",
                        "third_party/cutlass/tools/util/include","third_party/cutlass/test")],
                     str(source),str(repo/"build-portable/libtilemega.a"),
                     "-L/usr/local/cuda/lib64","-lcudart","-o",str(binary)]
            with (out/"build"/(tag+"_ptxas.txt")).open("w") as log:
                subprocess.run(command,stdout=log,stderr=log,check=True,timeout=600)
            builds.append(dict(tag=tag,binary_sha256=sha(binary),source_sha256=sha(source),command=command))
        (out/"manifest.json").write_text(json.dumps(dict(builds=builds),indent=2)+"\n")
        (out/"status.txt").write_text("BUILT\n")
        return
    manifest=json.loads((out/"manifest.json").read_text())
    for build in manifest["builds"]:
        if sha(out/"bin"/build["tag"])!=build["binary_sha256"]:
            raise RuntimeError("placement binary changed")
    cases=[(m,s,q) for q in (4,128) for m,s in arms]
    fields=["round","execution_index","model","state","seq","status","l05_ms","l1_ms","l2_ms",
            "resource_json","schedule_json","balanced_json","hash_json","log_sha256"]
    try:
        with (out/"paired.tsv").open("x") as stream:
            writer=csv.DictWriter(stream,fields,delimiter="\t",lineterminator="\n")
            writer.writeheader()
            for round_ in range(50):
                shift=round_%len(cases)
                for index,(model,state,seq) in enumerate(cases[shift:]+cases[:shift]):
                    tag=f"{model}_p{state}"
                    fixture=repo/f"docs/experiments/SEQSCAN/raw/fixture/{model}_s{seq}_p3"
                    run=subprocess.run([str(out/"bin"/tag),str(fixture)],capture_output=True,text=True,
                        timeout=180,env=dict(os.environ,TILEMEGA_WARMUP="5",TILEMEGA_REPEAT="11"))
                    text=run.stdout+run.stderr
                    log=out/"logs"/f"{tag}_s{seq}_r{round_}.txt"
                    log.write_text(text)
                    if run.returncode or "RESULT status=PASS" not in text:
                        raise RuntimeError(f"correctness stopped at {log}")
                    hashes=record(text,"E2E_HASH")
                    if len(set(hashes.values()))!=1:
                        raise RuntimeError("cross-level hash mismatch")
                    times=record(text,"E2E_TIME")
                    row=dict(round=round_,execution_index=index,model=model,state=state,seq=seq,status="PASS",
                             log_sha256=sha(log),hash_json=json.dumps(hashes),
                             resource_json=json.dumps(record(text,"E2E_RESOURCE")),
                             schedule_json=json.dumps(record(text,"E2E_SCHEDULE")),
                             balanced_json=json.dumps(record(text,"E2E_BALANCED") if state==4 else {}))
                    row.update({k:times[k] for k in ("l05_ms","l1_ms","l2_ms")})
                    writer.writerow(row)
                    stream.flush()
                (out/"status.txt").write_text(f"RUNNING {round_+1}/50 rounds\n")
        (out/"status.txt").write_text("PASS 400/400 fresh processes; prediction/sign gate pending\n")
    except BaseException as error:
        (out/"status.txt").write_text(f"STOP {error}\n")
        raise


if __name__=="__main__":
    main()
