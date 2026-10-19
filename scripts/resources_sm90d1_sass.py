#!/usr/bin/env python3
"""Inspect local instructions by function, with an explicit line-info surrogate."""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
from pathlib import Path
import re
import subprocess

from sm90d1_common import RUN, read, sha, write


def inspect(binary, lineinfo=False):
    identity = read(str(binary)+'.identity.json')
    folder = RUN / 'sass' / binary.parent.parent.name / binary.parent.name / binary.stem
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / 'sass.txt'
    if not path.exists():
        with path.open('w') as output:
            subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--dump-sass', str(binary)],
                stdout=output, env=dict(os.environ, CUDA_VISIBLE_DEVICES=''), check=True)
    functions, function, encoding = defaultdict(Counter), None, defaultdict(list)
    usage_path=folder/'resource-usage.txt'
    if not usage_path.exists():
        with usage_path.open('w') as output:
            subprocess.run(['/usr/local/cuda/bin/cuobjdump','--dump-resource-usage',str(binary)],
                           stdout=output,check=True)
    usage={}
    function=None
    for line in usage_path.open():
        match=re.search(r'Function (\S+):',line)
        if match:
            function=match[1]
        elif function and 'REG:' in line:
            usage[function]={k:int(v) for k,v in re.findall(r'(REG|STACK|SHARED|LOCAL):(\d+)',line)}
    function=None
    for line in path.open():
        match = re.search(r'Function\s*:\s*(\S+)', line)
        if match:
            function = match[1]
        if function:
            for opcode in re.findall(r'\b(LDL|STL)(?:\.|\s)', line):
                functions[function][opcode] += 1
            encoding[function] += re.findall(r'/\* (0x[0-9a-f]+) \*/', line)
    mapped = defaultdict(Counter)
    if lineinfo:
        extract = folder / 'elf'
        extract.mkdir(exist_ok=True)
        subprocess.run(['/usr/local/cuda/bin/cuobjdump', '--extract-elf', 'all', str(binary)],
                       cwd=extract, stdout=subprocess.DEVNULL, check=True)
        cubin = max(extract.glob('*.cubin'), key=lambda p:p.stat().st_size)
        disassembly = folder / 'line.sass'
        if not disassembly.exists():
            with disassembly.open('w') as output:
                subprocess.run(['/usr/local/cuda/bin/nvdisasm','-g','-gi',str(cubin)],
                               stdout=output, check=True)
        function, source, location, in_locations = None, None, None, False
        hot = ('ServingTiledMainloop','ServingGemmTaskBody','ServingGemv',
               'PagedAttentionTaskBody','ServingAttentionSimt','ServingAttentionMma',
               'ServingEpilogue','ServingDmEpilogue','Sm90Gmma','Sm90Attention')
        for line in disassembly.open():
            match = re.match(r'\.text\.(\S+):', line)
            if match:
                function = match[1]
                source, location = None, None
                in_locations = False
            match = re.search(r'//## File "([^"]+)", line (\d+)', line)
            if match:
                # nvdisasm lists the innermost source first, then callers.
                if not in_locations:
                    source, location = match[1], int(match[2])
                in_locations = True
            elif re.search(r'/\*[0-9a-f]+\*/',line):
                in_locations = False
            if function and re.search(r'\b(?:LDL|STL)(?:\.|\s)', line):
                region = 'hot' if source and any(part in source for part in hot) else 'dispatch_or_other'
                mapped[function][region] += 1
                mapped[function][f'{Path(source).name if source else "unknown"}:{location}'] += 1
    result = dict(binary=str(binary), binary_sha256=sha(binary), trace=identity['trace'],
                  resources=identity['resources'], functions={k:dict(v) for k,v in functions.items()},
                  cubin_resource_usage=usage,
                  instruction_encoding_sha256={k:__import__('hashlib').sha256(''.join(v).encode()).hexdigest() for k,v in encoding.items()},
                  line_mapping={k:dict(v) for k,v in mapped.items()})
    write(folder / 'resources.json', result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', type=int, default=4)
    p.add_argument('--tag', action='append')
    a = p.parse_args()
    binaries = [p for tag in (a.tag or ['reference2','lineinfo2','repeat2'])
                for p in (RUN / 'build' / tag).glob('*/*.so')
                if Path(str(p)+'.identity.json').exists()]
    with ThreadPoolExecutor(max_workers=a.jobs) as pool:
        futures = [pool.submit(inspect, p, 'lineinfo' in p.parent.parent.name or
                   (Path(str(p)+'.diagnostic.json').exists() and
                    read(str(p)+'.diagnostic.json').get('lineinfo',False))) for p in binaries]
        for future in as_completed(futures):
            result = future.result()
            print(result['binary'], flush=True)


if __name__ == '__main__':
    main()
