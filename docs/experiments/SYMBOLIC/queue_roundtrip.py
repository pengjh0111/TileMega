#!/usr/bin/env python3
"""Compare actual template/native queue vectors at all S5 validation points.

Consumes the already proved maps, without replacing their ISL certificates.
The original four families have forty endpoint/interior cases; each fitted
current winner adds five cases at its actual grid. No GPU execution occurs.
"""
import concurrent.futures,csv,hashlib,json,shutil,subprocess,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
OUT=HERE/'queue_roundtrip'
def rows(path):return list(csv.DictReader(path.open(),delimiter='\t'))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(OUT).free//2**20
    print(f'DISK NEED_MIB=8192 FREE_MIB={free}',flush=True)
    if free<8192:raise RuntimeError('insufficient compile space')
    driver=OUT/'bin/check_queues'
    subprocess.run(['python3',str(HERE.parent/'JOINT2/build_driver.py'),str(HERE/'check_queues.cpp'),str(driver)],cwd=REPO,check=True)
    jobs=[]
    original=HERE/'complete';cg=REPO/json.loads((original/'provenance.json').read_text())['source_cg']
    jobs.append(('original',cg,[(r,original) for r in rows(original/'samples.tsv')]))
    for root in sorted((HERE/'bounded_certificates').iterdir()):
        if not (root/'manifest.json').exists():continue
        metadata=json.loads((root/'manifest.json').read_text())
        samples=[(r,root/shard) for shard in metadata['shards'] for r in rows(root/shard/'samples.tsv')]
        jobs.append((root.name,Path(metadata['source']),samples))
    def run(job):
        name,cg,samples=job;folder=OUT/name;folder.mkdir(exist_ok=True)
        if (folder/'command.json').exists():raise RuntimeError('refusing overwrite '+str(folder))
        files={str(cg):sha(cg)};lines=['family\tgrid\tseq\tmap\n']
        for row,root in samples:
            path=root/f'{row["family"]}_g{row["grid"]}.pi_sigma.isl'
            files[str(path)]=sha(path)
            lines.append(f'{row["family"]}\t{row["grid"]}\t{row["seq"]}\t{path}\n')
        manifest=folder/'inputs.tsv';manifest.write_text(''.join(lines))
        command=[str(driver),str(cg),str(manifest),str(folder)]
        meta=dict(command=command,inputs=files,binary_sha256=sha(driver),started_ns=time.time_ns())
        with (folder/'run.log').open('w') as log:
            result=subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
        meta.update(exit_code=result.returncode,elapsed_ns=time.time_ns()-meta['started_ns'])
        (folder/'command.json').write_text(json.dumps(meta,indent=2)+'\n')
        result.check_returncode();print('QUEUE_ROUNDTRIP',name,len(samples),'PASS',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        for unused in pool.map(run,jobs):pass
if __name__=='__main__':main()
