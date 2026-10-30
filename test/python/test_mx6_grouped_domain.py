"""Compile the boundary cases that distinguish V3 from the V2 legal domain."""
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[2]


def test_versioned_geometry_boundaries(tmp_path):
    out=tmp_path/'geometry'
    subprocess.run(['c++','-std=c++17','-I'+str(ROOT/'include'),
        str(ROOT/'test/unit/mx/decode_geometry_v3.cpp'),'-o',str(out)],check=True)
    subprocess.run([str(out)],check=True)
