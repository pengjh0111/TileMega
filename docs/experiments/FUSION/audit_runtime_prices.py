#!/usr/bin/env python3
"""Reprice compiled chain states and check exact task/event receipts."""
import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

parser=argparse.ArgumentParser()
parser.add_argument("root",type=Path)
parser.add_argument("--label",default="price")
args=parser.parse_args()
root=args.root.resolve()
repo=Path(__file__).resolve().parents[3]
rows=list(csv.DictReader((root/"paired.tsv").open(),delimiter="\t"))
results=[]
for consumer in ("add","norm"):
    for seq in (4,128):
        command=[str(repo/"build-portable/tools/tilemega-fusion-dp"),str(root/consumer/"separate.mlir"),
            str(repo/"configs/targets/sm_89.json"),str(root/"build"/(consumer+"_separate_ptxas.txt")),
            str(seq),"3","-","--fused-ptxas",str(root/"build"/(consumer+"_fused_ptxas.txt"))]
        run=subprocess.run(command,cwd=repo,text=True,capture_output=True,timeout=600)
        with (root/"build"/f"{consumer}_s{seq}_{args.label}.txt").open("x") as stream:
            stream.write(run.stdout+run.stderr)
        run.check_returncode()
        prices=list(csv.DictReader(io.StringIO(run.stdout),delimiter="\t"))
        if len(prices)!=2:
            raise RuntimeError("chain DP did not evaluate both compiled states")
        for price in prices:
            state="fused" if int(price["fusion_count"]) else "separate"
            for row in rows:
                if row["consumer"]!=consumer or int(row["seq"])!=seq or row["state"]!=state:
                    continue
                resource=json.loads(row["resource_json"])
                schedule=json.loads(row["schedule_json"])
                for predicted,observed in ((price["task_refs"],schedule["task_refs"]),
                    (price["waits"],schedule["waits"]),(price["ctas_per_sm"],resource["ctas_per_sm"]),
                    (price["registers"],resource["reg"]),(price["shared"],resource["task_smem"])):
                    if int(predicted)!=int(observed):
                        raise RuntimeError(f"{consumer}/{seq}/{state}: model/resource mismatch {predicted} != {observed}")
            results.append(dict(consumer=consumer,seq=seq,state=state,**price))
with (root/(args.label+"s.json")).open("x") as stream:
    json.dump(dict(checks="2000/2000",results=results),stream,indent=2)
    stream.write("\n")
print("CHAIN_PRICE task/wait/register/shared/residency 2000/2000")
