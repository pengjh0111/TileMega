#!/usr/bin/env python3
"""Reuse verified synthetic CPU weights while retaining fresh guarded GPU children."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

from sm90d1_common import ROOT, PYTHON, read, sha, write

RUN = ROOT / 'runs/sm90x1'
POLICY = ROOT / 'configs/bench/sm90x1_parallel_policy.json'
IDENTITY_KEYS = ('algorithm', 'config_sha256', 'seed', 'aliases', 'tensors')
TERMINAL = {'complete', 'finished_with_failures', 'stopped', 'blocked'}


def process_start(pid):
    try:
        fields = Path('/proc', str(pid), 'stat').read_text().rsplit(')', 1)[1].split()
        return fields[19] if fields[0] not in ('Z', 'X') else None
    except OSError:
        return None


def identity(manifest):
    data = {key: manifest[key] for key in IDENTITY_KEYS}
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def hash_fd(fd):
    digest = hashlib.sha256(); offset = 0
    while True:
        chunk = os.pread(fd, 8 << 20, offset)
        if not chunk:
            return digest.hexdigest()
        digest.update(chunk); offset += len(chunk)


def borrow(source, reference, destination):
    source, reference, destination = Path(source), Path(reference), Path(destination)
    manifest, accepted = read(source/'random_checkpoint.json'), read(reference)
    if manifest['state'] not in ('live', 'anchored_unverified'):
        raise ValueError('checkpoint source is not live')
    expected = accepted['checkpoint_sha256']
    if identity(manifest) != expected or identity(accepted) != expected or manifest['seed'] != 0:
        raise ValueError('checkpoint differs from accepted seed-zero weights')
    if manifest['shards'] != accepted['shards']:
        raise ValueError('checkpoint shards differ from accepted reference')
    destination.mkdir(parents=True, exist_ok=False)
    descriptors = []
    try:
        for shard in manifest['shards']:
            fd = os.open(source/shard['file'], os.O_RDONLY)
            descriptors.append(fd)
            if os.fstat(fd).st_size != shard['bytes']:
                raise ValueError('checkpoint shard size mismatch')
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            digests = list(pool.map(hash_fd, descriptors))
        if digests != [shard['sha256'] for shard in accepted['shards']]:
            raise ValueError('checkpoint shard content mismatch')
        if sha(source/'config.json') != accepted['config_sha256']:
            raise ValueError('checkpoint config mismatch')
        for name in ('config.json', 'generation_config.json', 'model.safetensors.index.json'):
            if (source/name).exists():shutil.copyfile(source/name, destination/name)
        for shard, fd in zip(manifest['shards'], descriptors):
            (destination/shard['file']).symlink_to(f'/proc/{os.getpid()}/fd/{fd}')
        write(destination/'validated.json', dict(passed=True, checkpoint_sha256=expected,
              reference=str(reference.resolve()), reference_sha256=sha(reference),
              shard_sha256=digests, at=time.time()))
        manifest.update(state='live', supervisor_pid=os.getpid(),
                        supervisor_start=process_start(os.getpid()), source_checkpoint=str(source.resolve()))
        write(destination/'random_checkpoint.json', manifest)
        return descriptors
    except BaseException:
        for fd in descriptors:os.close(fd)
        shutil.rmtree(destination)
        raise


def validate(cache, config, expected, max_gib):
    cache = Path(cache); manifest = read(cache/'random_checkpoint.json')
    proof = read(cache/'validated.json')
    if manifest['state'] != 'live' or process_start(manifest['supervisor_pid']) != manifest['supervisor_start']:
        raise ValueError('checkpoint cache supervisor is unavailable')
    if not proof.get('passed') or proof['checkpoint_sha256'] != expected or identity(manifest) != expected:
        raise ValueError('checkpoint cache identity is unverified')
    if manifest['checkpoint_sha256'] != expected or manifest['seed'] != 0:
        raise ValueError('checkpoint cache seed or identity mismatch')
    if sha(Path(config)/'config.json') != manifest['config_sha256']:
        raise ValueError('checkpoint cache model mismatch')
    if manifest['total_bytes'] > max_gib * (1 << 30):
        raise ValueError('checkpoint exceeds declared memory limit')
    for shard in manifest['shards']:
        if (cache/shard['file']).stat().st_size != shard['bytes']:
            raise ValueError('checkpoint cache shard unavailable')
    return manifest


def cached_command(config, out, max_gib, child, scope=None):
    if not Path(scope or out).resolve().is_relative_to(RUN):return None
    entry = read(POLICY).get('checkpoint_cache') if POLICY.exists() else None
    if not entry or sha(Path(config)/'config.json') != entry['config_sha256']:return None
    return [PYTHON, str(Path(__file__).resolve()), 'run', '--cache', entry['path'],
            '--config', str(config), '--expected', entry['checkpoint_sha256'],
            '--out', str(out), '--max-gib', str(max_gib), '--', *child]


def run(args):
    if os.environ.get('TILEMEGA_GPU_LOCK_HELD') != '1':
        raise ValueError('checkpoint GPU child requires the scheduler guard')
    with (args.cache/'readers.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        manifest = validate(args.cache, args.config, args.expected, args.max_gib)
        args.out.mkdir(parents=True, exist_ok=False)
        for name in ('config.json', 'generation_config.json', 'model.safetensors.index.json'):
            if (args.cache/name).exists():shutil.copyfile(args.cache/name, args.out/name)
        for shard in manifest['shards']:
            (args.out/shard['file']).symlink_to(args.cache.resolve()/shard['file'])
        manifest.update(reused_cpu_checkpoint=str(args.cache), reader_pid=os.getpid(),
                        cpu_checkpoint_runner_sha256=sha(__file__))
        write(args.out/'random_checkpoint.json', manifest)
        child = args.child[1:] if args.child[:1] == ['--'] else args.child
        try:
            return subprocess.call(child, env=dict(os.environ, TILEMEGA_RANDOM_CHECKPOINT=str(args.out.resolve())))
        finally:
            for shard in manifest['shards']:(args.out/shard['file']).unlink(missing_ok=True)
            manifest['state'] = 'closed'
            write(args.out/'random_checkpoint.json', manifest)


def workers_live():
    names = ['ready-phase2-process.json', 'post-phase2-process.json', 'phase2-pipeline-process.json']
    if (RUN/'extra-worker-metadata.json').exists():names += read(RUN/'extra-worker-metadata.json')
    return any((RUN/name).exists() and (pid := read(RUN/name)['pid']) != os.getpid()
               and process_start(pid) is not None for name in names)


def serve(args):
    with (args.out.parent/'checkpoint-service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        descriptors = borrow(args.source, args.reference, args.out)
        try:
            while not (RUN/'stop-checkpoint-cache').exists():
                if read(RUN/'campaign-state.json')['status'] in TERMINAL and not workers_live():break
                time.sleep(2)
            with (args.out/'readers.lock').open('a') as readers:
                fcntl.flock(readers, fcntl.LOCK_EX)
                manifest = read(args.out/'random_checkpoint.json');manifest['state'] = 'closed'
                write(args.out/'random_checkpoint.json', manifest)
        finally:
            for fd in descriptors:os.close(fd)


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='action', required=True)
    p = sub.add_parser('serve');p.add_argument('--source', type=Path, required=True)
    p.add_argument('--reference', type=Path, required=True);p.add_argument('--out', type=Path, required=True)
    p = sub.add_parser('run');p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--config', type=Path, required=True);p.add_argument('--expected', required=True)
    p.add_argument('--out', type=Path, required=True);p.add_argument('--max-gib', type=float, required=True)
    p.add_argument('child', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if args.action == 'serve':serve(args);return 0
    return run(args)


if __name__ == '__main__':
    raise SystemExit(main())
