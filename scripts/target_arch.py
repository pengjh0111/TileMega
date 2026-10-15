"""Use the package compiler-target mapping from standalone validation scripts."""
from pathlib import Path
import runpy

mapping = runpy.run_path(str(Path(__file__).resolve().parents[1] /
                               'python/tilemega/build/arch.py'))
nvcc_arch = mapping['nvcc_arch']
nvcc_flag = mapping['nvcc_flag']

if __name__ == '__main__':
    import sys
    print(nvcc_flag(sys.argv[2]) if sys.argv[1]=='--flag' else nvcc_arch(sys.argv[1]))
