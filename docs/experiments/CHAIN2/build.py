#!/usr/bin/env python3
"""Build the R4 driver with the existing target's compiler and linker recipe."""
from pathlib import Path
import json
import hashlib
import shlex
import shutil
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
BUILD = REPO / 'build-portable'


def main():
    free = shutil.disk_usage(HERE).free // 2**20
    print(f'DISK NEED_MIB=2048 FREE_MIB={free}', flush=True)
    if free < 2048:
        raise RuntimeError('insufficient disk before compilation')
    subprocess.run(['ninja', '-C', str(BUILD), 'chain_placement_test', 'tools/tilemega-place-chain'], check=True)
    subprocess.run([str(BUILD / 'chain_placement_test')], check=True)
    commands = subprocess.check_output(['ninja', '-C', str(BUILD), '-t', 'commands',
                                       'tools/tilemega-place-chain'], text=True).splitlines()
    compile_command = next(c for c in commands if ' -c ' in c and '/CHAIN/place_chain.cpp' in c)
    link_command = commands[-1]
    obj = str(HERE / 'raw/place_chain.o')
    binary = str(HERE / 'raw/place_chain')
    old_obj = 'CMakeFiles/tilemega-place-chain.dir/docs/experiments/CHAIN/place_chain.cpp.o'
    compile_command = compile_command.replace(old_obj, obj).replace(
        str(REPO / 'docs/experiments/CHAIN/place_chain.cpp'), str(HERE / 'place_chain.cpp'))
    link_command = link_command.replace(old_obj, obj).replace('-o tools/tilemega-place-chain', '-o '+binary)
    # Ninja's linker recipe uses ':' shell sentinels. Preserve the recipe from
    # the trusted local build rather than synthesizing an MLIR library list.
    (HERE / 'raw').mkdir(exist_ok=True)
    (HERE / 'raw/build_commands.json').write_text(json.dumps(
        [compile_command, link_command], indent=2)+'\n')
    subprocess.run(shlex.split(compile_command), cwd=BUILD, check=True)
    subprocess.run(['bash', '-c', link_command], cwd=BUILD, check=True)
    sources = [REPO / 'lib/Solver/ChainPlacement.cpp', REPO / 'include/tilemega/Solver/ChainPlacement.h',
               HERE / 'place_chain.cpp', REPO / 'docs/experiments/SIMULATOR/cell_inputs.h',
               REPO / 'docs/experiments/SIMULATOR/hop_ns.tsv', REPO / 'configs/targets/sm_89.json']
    (HERE / 'raw/build_inputs.json').write_text(json.dumps({str(p.relative_to(REPO)):
        hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}, indent=2)+'\n')


if __name__ == '__main__':
    main()
