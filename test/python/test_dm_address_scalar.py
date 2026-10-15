# SPDX-License-Identifier: BSD-3-Clause
"""Both address implementations preserve the descriptor's coordinate map."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
#include <tilemega/Backend/DmTensorAddress.h>
#include <cassert>
#include <random>
using namespace tilemega;
int main() {
  std::mt19937 random(16);
  for (unsigned rank=1; rank<=4; ++rank)
    for (unsigned kind=0; kind<2; ++kind)
      for (unsigned sample=0; sample<1000; ++sample) {
        codegen::DmBufferLayout layout{};
        layout.rank=rank; layout.kind=codegen::DmLayout(kind);
        layout.halo_top=random()%8; layout.halo_left=random()%8;
        std::uint64_t row=0, expected=0;
        unsigned column=random()%17;
        for (unsigned axis=0; axis<rank; ++axis) {
          layout.logical[axis]=1+random()%31;
          layout.strides[axis]=1+random()%100000;
          unsigned coordinate=axis+1==rank?column:random()%layout.logical[axis];
          if (axis+1<rank) row=row*layout.logical[axis]+coordinate;
          if (kind && axis+1<rank) {
            if (axis==1) coordinate+=layout.halo_top;
            if (axis==2) coordinate+=layout.halo_left;
          }
          expected+=coordinate*layout.strides[axis];
        }
        assert(backend::DmTensorAddress<codegen::DmWriteKind::kDense>::Offset(
            layout,row,column,713)==expected);
      }
  codegen::DmBufferLayout absent{};
  assert(backend::DmTensorAddress<codegen::DmWriteKind::kDense>::Offset(
      absent,13,7,713)==13*713+7);
}
'''

class AddressScalarTest(unittest.TestCase):
    def test_descriptor_coordinates_with_both_feature_settings(self):
        with tempfile.TemporaryDirectory(prefix='tilemega-address-') as directory:
            source = Path(directory) / 'address.cpp'
            source.write_text(SOURCE)
            for feature in (0, 1):
                with self.subTest(dwpw=feature):
                    binary = Path(directory) / f'address-{feature}'
                    subprocess.run([os.environ.get('CXX', 'c++'), '-O2', '-std=c++17',
                                    '-I', str(ROOT / 'include'),
                                    f'-DTILEMEGA_DWPW_WGMMA={feature}', str(source),
                                    '-o', str(binary)], check=True)
                    subprocess.run([str(binary)], check=True,
                                   env={**os.environ, 'CUDA_VISIBLE_DEVICES': ''})

if __name__ == '__main__':
    unittest.main()
