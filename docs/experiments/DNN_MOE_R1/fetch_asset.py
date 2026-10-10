#!/usr/bin/env python3
"""Resume an official asset download and publish its verified file identity."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def git_blob(path):
    value = hashlib.sha1()
    value.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            value.update(block)
    return value.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--bytes', type=int)
    parser.add_argument('--sha256')
    parser.add_argument('--git-blob', help='Git blob identity for a non-LFS file')
    parser.add_argument('--attempts', type=int, default=100)
    parser.add_argument('--attempt-seconds', type=int, default=180)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report = Path(str(args.out) + '.download.json')
    if report.exists():
        old = json.loads(report.read_text())
        if (old['url'] != args.url or not args.out.exists() or
                args.out.stat().st_size != old['bytes'] or
                digest(args.out) != old['sha256']):
            raise ValueError('asset changed since its completed download')
        if args.bytes is not None and old['bytes'] != args.bytes:
            raise ValueError('completed asset has the wrong size')
        if args.sha256 and not old['sha256'].startswith(args.sha256):
            raise ValueError('completed asset has the wrong digest')
        if args.git_blob and git_blob(args.out) != args.git_blob:
            raise ValueError('completed asset has the wrong Git blob identity')
        print(json.dumps(dict(event='asset_verified', **old)), flush=True)
        return
    log = Path(str(args.out) + '.download.log')
    if args.bytes is not None and (args.sha256 or args.git_blob) and args.out.exists() and \
            args.out.stat().st_size == args.bytes:
        sha = digest(args.out)
        if (args.sha256 and not sha.startswith(args.sha256)) or \
                (args.git_blob and git_blob(args.out) != args.git_blob):
            raise ValueError('existing asset has the wrong digest; retained for inspection')
        value = dict(url=args.url, path=str(args.out.resolve()), bytes=args.bytes,
                     sha256=sha, evidence='verified', attempts=0)
        report.write_text(json.dumps(value, indent=2) + '\n')
        print(json.dumps(dict(event='asset_verified', **value)), flush=True)
        return
    for attempt in range(1, args.attempts + 1):
        # A fresh curl computes the current resume offset after every failure.
        # A retry of the original command may reuse its original offset.
        command = ['curl', '--http1.1', '--fail', '--location', '--silent',
                   '--show-error', '--connect-timeout', '30', '--max-time',
                   str(args.attempt_seconds), '--continue-at', '-', '--output',
                   str(args.out), args.url]
        with log.open('a') as stream:
            stream.write(f'attempt {attempt}\n')
            stream.flush()
            code = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT).returncode
        if code == 0:
            size = args.out.stat().st_size
            if args.bytes is not None and size != args.bytes:
                raise ValueError(f'wrong asset size: {size}, expected {args.bytes}')
            sha = digest(args.out)
            if args.sha256 and not sha.startswith(args.sha256):
                raise ValueError('downloaded asset has the wrong digest')
            if args.git_blob and git_blob(args.out) != args.git_blob:
                raise ValueError('downloaded asset has the wrong Git blob identity')
            value = dict(url=args.url, path=str(args.out.resolve()), bytes=size,
                         sha256=sha, evidence='verified', attempts=attempt)
            report.write_text(json.dumps(value, indent=2) + '\n')
            print(json.dumps(dict(event='asset_download_complete', **value)), flush=True)
            return
        if code not in (5, 6, 7, 18, 28, 35, 52, 55, 56, 92):
            raise RuntimeError(f'non-resumable curl failure {code}; see {log}')
        time.sleep(min(attempt, 30))
    raise RuntimeError(f'download attempts exhausted; retained partial asset at {args.out}')


if __name__ == '__main__':
    main()
