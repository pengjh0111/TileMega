"""Execute a frozen five-process TP device test in a guarded card group."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--build',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    identity=json.loads((args.build/'BUILD.json').read_text())
    source=json.loads((args.build.parent/'SOURCE.json').read_text())
    sha=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if identity['returncode'] or identity['source_digest']!=source['source_digest']:
        raise ValueError('device test build is not source-bound')
    for name,digest in source['files'].items():
        if sha(args.build.parent/'frozen'/name)!=digest:raise ValueError('frozen test source changed')
    if sha(identity['binary'])!=identity['binary_sha256']:raise ValueError('test binary changed')
    if len(os.environ.get('TILEMEGA_DEVICE_GROUP','').split(','))!=identity['tp']:
        raise ValueError('test TP differs from guarded group')
    subprocess.run([sys.executable,'experiments/mx7/run_probe.py','--exe',identity['binary'],
                    '--source',identity['source'],'--out',str(args.out),'--'],check=True)
    (args.out/'build_identity.json').write_text(json.dumps(identity,indent=2)+'\n')


if __name__=='__main__':main()
