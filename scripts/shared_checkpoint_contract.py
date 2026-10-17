"""Validate the explicit zero-copy checkpoint lease without allocating weights."""
import fcntl,hashlib,json,os
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def validate(candidate,live=False):
    spec=candidate['shared_checkpoint'];model=Path(spec['model']);lease=model/'lease.json'
    if sha(lease)!=spec['lease_sha256']:raise ValueError('lease proof changed')
    proof=read(lease);manifest=read(model/'random_checkpoint.json')
    if proof['state']!='live' or proof['owner_pid']!=spec['owner_pid'] or str(proof['owner_start_ticks'])!=spec['owner_start_ticks']:
        raise ValueError('lease owner differs')
    for name,digest in spec['metadata_sha256'].items():
        if sha(model/name)!=digest or proof['metadata_sha256'][name]!=digest:raise ValueError('shared checkpoint metadata differs')
    if (manifest['checkpoint_sha256']!=spec['checkpoint_sha256'] or proof['checkpoint_sha256']!=spec['checkpoint_sha256'] or
        manifest['seed']!=candidate['seed'] or manifest['config_sha256']!=candidate['config_sha256'] or
        manifest['tensor_count']!=18867 or manifest['total_bytes']!=61064245248 or len(manifest['shards'])!=114):
        raise ValueError('shared checkpoint is not the exact full-model seed')
    rows={r['file']:r for r in manifest['shards']}
    if len(proof['shards'])!=114 or {r['file'] for r in proof['shards']}!=set(rows):raise ValueError('lease shard coverage differs')
    if live:
        stat=Path(f'/proc/{proof["owner_pid"]}/stat').read_text().rsplit(')',1)[1].split()
        if stat[0]=='Z' or stat[19]!=spec['owner_start_ticks']:raise ValueError('lease owner expired or PID reused')
        for r in proof['shards']:
            p=model/r['file'];target=f'/proc/{proof["owner_pid"]}/fd/{Path(r["lease_fd_target"]).name}'
            if os.readlink(p)!=target or r['lease_fd_target']!=target or r['manifest_full_sha256']!=rows[r['file']]['sha256']:
                raise ValueError('lease FD provenance differs')
            info=Path(f'/proc/{proof["owner_pid"]}/fdinfo/{Path(target).name}').read_text()
            flags=int(next(line.split()[1] for line in info.splitlines() if line.startswith('flags:')),8)
            if flags&os.O_ACCMODE!=os.O_RDONLY:raise ValueError('lease owner FD is not read-only')
            fd=os.open(p,os.O_RDONLY|os.O_CLOEXEC)
            try:
                s=os.fstat(fd)
                if (s.st_dev,s.st_ino,s.st_size)!=(r['device'],r['inode'],r['bytes']) or s.st_size!=rows[r['file']]['bytes']:
                    raise ValueError('shared weight inode/size differs')
                sample=hashlib.sha256(os.pread(fd,4096,0)+os.pread(fd,4096,max(0,s.st_size-4096))).hexdigest()
                if sample!=r['sample_sha256'] or fcntl.fcntl(fd,fcntl.F_GET_SEALS)!=r['memfd_seals']:raise ValueError('lease sample or seals changed')
            finally:os.close(fd)
    return {'passed':True,'live_validated':live,'contract':spec,'helper_sha256':sha(__file__),
        'proof_scope':'exact metadata plus original generator full digest, same memfd inode/size and read-only lease FD; samples checked; full 61GB not rehashed and memfd not write-sealed'}

def measurement(candidate,path):
    p=Path(path);value=read(p);receipt=read(p.parent.parent/'shared-checkpoint-lease-validation.json');config=value['configuration'];spec=candidate['shared_checkpoint']
    if (receipt['passed'] is not True or receipt['live_validated'] is not True or receipt['contract']!=spec or
        receipt['helper_sha256']!=sha(__file__) or config['checkpoint_sha256']!=spec['checkpoint_sha256'] or
        str(Path(config['checkpoint_manifest']))!=spec['resolved_manifest']):
        raise ValueError('measurement used another shared checkpoint or lacks lease validation')
    return receipt
