"""Specialize the selected C0 access primitives without changing the LL protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--store', type=int, choices=(0, 1), required=True)
    parser.add_argument('--load', type=int, choices=range(7), required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    original = Path('experiments/mx7/ll_litmus.cu')
    text = original.read_text()
    read = 'Word Read(Word const* p, int method) {'
    write = 'void Write(Word* p, Word value, int method) {'
    if text.count(read) != 1 or text.count(write) != 1:
        raise ValueError('LL primitive definitions changed; inspect before specializing')
    text = text.replace(read, read + f'\n  method = {args.load};')
    text = text.replace(write, write + f'\n  method = {args.store};')
    source = args.out / 'll_static.cu'
    source.write_text(text)
    binary = args.out / 'll_litmus'
    command = ['/opt/maca/tools/cu-bridge/bin/cucc', '-std=c++17', '-O3',
               '-Xcompiler=-resource-usage', '-fno-strict-aliasing', str(source), '-o', str(binary)]
    with (args.out / 'build.log').open('x') as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    record = dict(original_source=str(original), original_sha256=sha(original),
                  generated_source=str(source), generated_sha256=sha(source),
                  binary=str(binary), binary_sha256=sha(binary) if binary.exists() else None,
                  store=args.store, load=args.load, command=command, returncode=result.returncode,
                  note='Only access-method parameters become compile-time constants; epochs, packets, CTA participation and polling bound are unchanged.')
    (args.out / 'IDENTITY.json').write_text(json.dumps(record, indent=2) + '\n')
    if result.returncode:
        raise RuntimeError('static LL build failed; preserve compiler log')


if __name__ == '__main__':
    main()
