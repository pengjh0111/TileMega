#!/usr/bin/env python3
"""Restore the losslessly compressed oversized trace before local analysis.

The raw file is deliberately not a Git blob (>100 MiB). Existing raw files
are preserved; a mismatch is an error, never overwritten.
"""
import gzip,hashlib,json,pathlib,shutil,tempfile

E=pathlib.Path(__file__).resolve().parent

def checksum(source):
    digest=hashlib.sha256()
    for block in iter(lambda:source.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()

def restore():
    manifest=E/'trace/compressed.json'
    for item in json.loads(manifest.read_text()):
        raw=E/item['raw'];archive=E/item['archive']
        if raw.exists():
            with raw.open('rb') as f:digest=checksum(f)
            if digest!=item['sha256']:raise RuntimeError('raw trace checksum mismatch: '+str(raw))
            continue
        with tempfile.NamedTemporaryFile(dir=raw.parent,delete=False) as tmp:
            temporary=pathlib.Path(tmp.name)
            with gzip.open(archive,'rb') as source:shutil.copyfileobj(source,tmp)
        with temporary.open('rb') as f:digest=checksum(f)
        if digest!=item['sha256']:
            temporary.unlink();raise RuntimeError('compressed trace checksum mismatch: '+str(archive))
        temporary.replace(raw)

if __name__=='__main__':restore()
