#!/usr/bin/env python3
"""Hold read-only references to existing checkpoint memfds without copying weights."""
import argparse
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import select
import shutil
import time


def sha(data):
    return hashlib.sha256(data).hexdigest()


def start_ticks(pid):
    return (Path('/proc') / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[19]


class CheckpointLease:
    def __init__(self, source, destination, expected_sha, source_pid, source_start_ticks):
        self.source, self.destination = Path(source), Path(destination)
        self.expected_sha, self.source_pid = expected_sha, source_pid
        self.source_start_ticks = str(source_start_ticks)
        self.fds, self.links = [], []
        self.proof = None

    def __enter__(self):
        manifest = json.loads((self.source / 'random_checkpoint.json').read_text())
        identity = {key: manifest[key] for key in
                    ('algorithm', 'config_sha256', 'seed', 'aliases', 'tensors')}
        fingerprint = sha(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode())
        if (manifest['state'] != 'live' or manifest['checkpoint_sha256'] != self.expected_sha or
                fingerprint != self.expected_sha or manifest['supervisor_pid'] != self.source_pid or
                start_ticks(self.source_pid) != self.source_start_ticks):
            raise ValueError('source checkpoint identity or live owner differs')
        if sha((self.source / 'config.json').read_bytes()) != manifest['config_sha256']:
            raise ValueError('source checkpoint configuration differs')
        names = [row['file'] for row in manifest['shards']]
        index = json.loads((self.source / 'model.safetensors.index.json').read_text())
        if (not names or len(set(names)) != len(names) or
                set(index['weight_map'].values()) != set(names) or
                set(index['weight_map']) != {row['name'] for row in manifest['tensors']} or
                sum(row['data_bytes'] for row in manifest['shards']) != manifest['total_bytes']):
            raise ValueError('source checkpoint index or shard counts differ')
        self.destination.mkdir(parents=True, exist_ok=False)
        metadata, shards = {}, []
        try:
            for name in ('config.json', 'generation_config.json', 'model.safetensors.index.json',
                         'random_checkpoint.json'):
                source = self.source / name
                if source.is_file():
                    shutil.copyfile(source, self.destination / name)
                    digest = sha(source.read_bytes())
                    if sha((self.destination / name).read_bytes()) != digest:
                        raise ValueError('checkpoint metadata changed while leasing')
                    metadata[name] = digest
            for row in manifest['shards']:
                name = row['file']
                if Path(name).name != name or not name.endswith('.safetensors'):
                    raise ValueError('invalid shard filename')
                source = self.source / name
                target = Path(os.readlink(source))
                if target.parent != Path(f'/proc/{self.source_pid}/fd') or not target.name.isdigit():
                    raise ValueError('shard is not owned by the declared checkpoint supervisor')
                before = source.stat()
                fd = os.open(source, os.O_RDONLY | os.O_CLOEXEC)
                self.fds.append(fd)
                actual = os.fstat(fd)
                if ((before.st_dev, before.st_ino, before.st_size) !=
                        (actual.st_dev, actual.st_ino, actual.st_size) or actual.st_size != row['bytes'] or
                        fcntl.fcntl(fd, fcntl.F_GETFL) & os.O_ACCMODE != os.O_RDONLY):
                    raise ValueError('lease does not reference the same read-only memfd')
                tail = max(0, actual.st_size - 4096)
                sample = sha(os.pread(fd, 4096, 0) + os.pread(fd, 4096, tail))
                with source.open('rb') as stream:
                    first = stream.read(4096)
                    stream.seek(tail)
                    if sha(first + stream.read(4096)) != sample:
                        raise ValueError('source and lease byte samples differ')
                leased_target = f'/proc/{os.getpid()}/fd/{fd}'
                link = self.destination / name
                link.symlink_to(leased_target)
                self.links.append((link, leased_target))
                seals = fcntl.fcntl(fd, fcntl.F_GET_SEALS)
                source_info = (Path('/proc') / str(self.source_pid) / 'fdinfo' / target.name).read_text()
                source_flags = int(next(line.split()[1] for line in source_info.splitlines()
                                        if line.startswith('flags:')), 8)
                shards.append(dict(file=name, source_fd_target=str(target), lease_fd_target=leased_target,
                    device=actual.st_dev, inode=actual.st_ino, bytes=actual.st_size,
                    same_inode_verified=True, read_only=True, sample_sha256=sample,
                    memfd_seals=seals, write_sealed=bool(seals & fcntl.F_SEAL_WRITE),
                    lease_fd_access_mode=os.O_RDONLY, source_fd_access_mode=source_flags & os.O_ACCMODE,
                    manifest_full_sha256=row['sha256'], full_sha256_rehashed=False))
            self.proof = dict(schema_version=1, state='live', created=time.time(),
                source=str(self.source), destination=str(self.destination),
                source_owner_pid=self.source_pid, source_owner_start_ticks=self.source_start_ticks,
                owner_pid=os.getpid(), owner_start_ticks=start_ticks(os.getpid()),
                checkpoint_sha256=self.expected_sha, config_sha256=manifest['config_sha256'],
                tensor_count=manifest['tensor_count'], weight_bytes=manifest['total_bytes'],
                metadata_sha256=metadata, shards=shards, additional_weight_bytes_allocated=0,
                byte_identity_proof='same memfd inode and size; read-only descriptors; matching byte samples; original generator full SHA retained',
                metadata_byte_preserved=True, no_cuda_context_created=True)
            for name in ('runtime/random_checkpoint.py', 'supervisor.py'):
                path = self.source.parent / name
                if path.is_file():
                    self.proof[name.replace('/', '_') + '_sha256'] = sha(path.read_bytes())
            (self.destination / 'lease.json').write_text(json.dumps(self.proof, indent=2) + '\n')
            return self
        except BaseException:
            self.close('acquisition_failed')
            raise

    def close(self, reason='context_exit'):
        for path, target in self.links:
            if path.is_symlink() and os.readlink(path) == target:
                path.unlink()
        self.links.clear()
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()
        if self.destination.exists():
            (self.destination / 'lease-closed.json').write_text(json.dumps(dict(
                closed=time.time(), owner_pid=os.getpid(), reason=reason)) + '\n')

    def __exit__(self, *error):
        self.close('exception' if error[0] else 'released')


def terminal_plan(plan, states):
    names = plan.get('jobs', [])
    if not isinstance(names, list) or len(set(names)) != len(names):
        raise ValueError('lease jobs must be a list of unique names')
    return (plan.get('sealed') is True and plan.get('release_when_terminal') is True and bool(names) and
            all(name in states and states[name].get('status') in
                ('done', 'failed', 'skipped', 'not_run', 'cancelled') for name in names))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--expected-sha', required=True)
    parser.add_argument('--source-pid', type=int, required=True)
    parser.add_argument('--source-start-ticks', required=True)
    parser.add_argument('--jobs-file', type=Path, required=True)
    parser.add_argument('--state-file', type=Path, required=True)
    parser.add_argument('--release-file', type=Path, required=True)
    parser.add_argument('--ready-out', type=Path, required=True)
    args = parser.parse_args()
    with CheckpointLease(args.source, args.out, args.expected_sha,
                         args.source_pid, args.source_start_ticks) as lease:
        ready = {key: lease.proof[key] for key in ('destination', 'owner_pid', 'owner_start_ticks',
            'checkpoint_sha256', 'config_sha256', 'tensor_count', 'weight_bytes',
            'additional_weight_bytes_allocated', 'metadata_sha256')}
        ready['lease_sha256'] = sha((args.out / 'lease.json').read_bytes())
        libc = ctypes.CDLL(None, use_errno=True)
        fd = libc.inotify_init1(os.O_CLOEXEC | os.O_NONBLOCK)
        if fd < 0:
            raise OSError(ctypes.get_errno(), 'inotify_init1')
        for directory in {args.jobs_file.parent, args.state_file.parent, args.release_file.parent}:
            if libc.inotify_add_watch(fd, os.fsencode(directory), 8 | 128) < 0:
                raise OSError(ctypes.get_errno(), 'inotify_add_watch')
        args.ready_out.write_text(json.dumps(ready, indent=2) + '\n')
        print(json.dumps(ready), flush=True)
        while not args.release_file.exists():
            try:
                plan = json.loads(args.jobs_file.read_text()) if args.jobs_file.exists() else {}
                states = json.loads(args.state_file.read_text())
                if terminal_plan(plan, states):
                    break
            except (json.JSONDecodeError, FileNotFoundError, ValueError, TypeError):
                pass
            select.select([fd], [], [], 60)
            try:
                os.read(fd, 65536)
            except BlockingIOError:
                pass
        os.close(fd)


if __name__ == '__main__':
    main()
