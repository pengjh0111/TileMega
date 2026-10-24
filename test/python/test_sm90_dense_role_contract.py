# SPDX-License-Identifier: BSD-3-Clause
"""Compile the actual host role contract, including unsupported launch rejection."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]


class NativeRoleContract(unittest.TestCase):
    def test_fixed_launch_shape(self):
        prefix = '#include <tilemega/Backend/Sm90DenseRoleContract.h>\n'
        with tempfile.TemporaryDirectory() as tmp:
            source, output = Path(tmp) / 'role.cpp', Path(tmp) / 'role.o'
            for shape, supported in [('160,1,1', True), ('128,1,1', False),
                                     ('192,1,1', False), ('160,2,1', False),
                                     ('160,1,2', False)]:
                with self.subTest(shape=shape):
                    source.write_text(prefix +
                        f'using Launch=tilemega::backend::Sm90DenseRoleLaunch<{shape}>;\n'
                        'static_assert(sizeof(Launch)>0);\n')
                    command = [*shlex.split(os.environ.get('CXX', 'c++')),
                               '-std=c++17', '-I' + str(ROOT / 'include'),
                               '-c', str(source), '-o', str(output)]
                    result = subprocess.run(command, stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE, text=True, timeout=60)
                    if supported:
                        self.assertEqual(result.returncode, 0, result.stderr)
                    else:
                        self.assertNotEqual(result.returncode, 0)
                        self.assertIn('160x1x1 thread block', result.stderr)


if __name__ == '__main__':
    unittest.main()
