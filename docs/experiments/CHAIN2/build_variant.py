#!/usr/bin/env python3
"""Compile a solver diagnostic without mutating the shared production library."""
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUILD = REPO / 'build-portable'


def main():
    raw = HERE / 'raw/minimal'
    (raw / 'plan').mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(raw).free // 2**20
    print(f'DISK NEED_MIB=2048 FREE_MIB={free}', flush=True)
    if free < 2048:
        raise RuntimeError('insufficient disk before compilation')
    commands = subprocess.check_output(['ninja', '-C', str(BUILD), '-t', 'commands', 'libtilemega.a'], text=True).splitlines()
    cmd = next(c for c in commands if ' -c ' in c and '/Solver/ChainPlacement.cpp' in c)
    obj = str(raw / 'ChainPlacement.o')
    cmd = cmd.replace('CMakeFiles/tilemega.dir/lib/Solver/ChainPlacement.cpp.o', obj).replace(
        str(REPO / 'lib/Solver/ChainPlacement.cpp'), str(HERE / 'minimal_variant.cpp'))
    subprocess.run(shlex.split(cmd), cwd=BUILD, check=True)
    link = json.loads((HERE / 'raw/build_commands.json').read_text())[1]
    link = link.replace('-o '+str(HERE / 'raw/place_chain'), '-o '+str(raw / 'place_chain'))
    link = link.replace(' libtilemega.a ', ' '+shlex.quote(obj)+' libtilemega.a ')
    subprocess.run(['bash', '-c', link], cwd=BUILD, check=True)
    (raw / 'build.json').write_text(json.dumps(dict(commands=[cmd, link],
        source_sha256=hashlib.sha256((HERE / 'minimal_variant.cpp').read_bytes()).hexdigest()), indent=2)+'\n')


if __name__ == '__main__':
    main()
