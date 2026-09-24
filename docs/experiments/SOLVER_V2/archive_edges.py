#!/usr/bin/env python3
"""Losslessly archive final simulator edge dumps after a completed solve.

Usage: archive_edges.py ARM_DIRECTORY [...]
Keeps local originals, writes deterministic .edges.tsv.gz and a byte-hash
manifest. Commit compressed copies only; plan_statistics.py reads them directly.
These files contain raw edges, not summarized statistics.
"""
import argparse,gzip,hashlib,json,pathlib,shutil

def digest(path,compressed=False):
    h=hashlib.sha256()
    with (gzip.open(path,'rb') if compressed else path.open('rb')) as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('arms',nargs='+',type=pathlib.Path);a=ap.parse_args()
    for arm in a.arms:
        if not (arm/'selected.cu.timing.tsv').exists():raise RuntimeError('solve is not complete: '+str(arm))
        records=[]
        for raw in sorted(arm.glob('selected.cu.final*.edges.tsv')):
            compressed=pathlib.Path(str(raw)+'.gz');temporary=pathlib.Path(str(compressed)+'.tmp')
            original_hash=digest(raw)
            if not compressed.exists() or digest(compressed,True)!=original_hash:
                with raw.open('rb') as src,temporary.open('wb') as dst:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=dst,mtime=0,compresslevel=6) as zipped:shutil.copyfileobj(src,zipped)
                if digest(temporary,True)!=original_hash:raise RuntimeError('archive round trip mismatch: '+str(raw))
                temporary.replace(compressed)
            records.append(dict(original=raw.name,archive=compressed.name,sha256=original_hash,archive_sha256=digest(compressed),bytes=raw.stat().st_size,archive_bytes=compressed.stat().st_size))
        (arm/'edge_archives.json').write_text(json.dumps(records,indent=2)+'\n')
        print(arm,'raw edge archives=',len(records),'all decompressed hashes match')

if __name__=='__main__':main()
