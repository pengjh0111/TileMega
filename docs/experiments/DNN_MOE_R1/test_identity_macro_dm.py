import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import tempfile
import unittest

module_path = Path(os.environ.get('TILEMEGA_DM_IDENTITY_MODULE',
                                 str(Path(__file__).with_name('identity_dm.py'))))
spec = importlib.util.spec_from_file_location('identity_under_test', module_path)
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


class CompleteIdentityTests(unittest.TestCase):
    def fixture(self, base):
        compiler = base / 'build-dm/tools/tilemega'
        compiler.parent.mkdir(parents=True)
        compiler.write_bytes(b'synthetic plan compiler identity fixture')
        header = base / 'defaults.h'
        header.write_text('#define INCLUDED_DEFAULT 17\n')
        support = base / 'support.cpp'
        support.write_text('#include "defaults.h"\n#define SUPPORT_ONLY 31\n')
        so = base / 'plan.so'
        so.write_bytes(b'synthetic shared object identity fixture; never loaded')
        Path(str(so) + '.cu').write_text('#include "defaults.h"\n#define MAIN_ONLY 29\n')
        command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', '-arch=sm_89', '-shared',
                   '-Xcompiler=-fPIC', '-DCLI_VALUE=7', '-I' + str(base),
                   '-x', 'cu', str(so) + '.cu', '-x', 'cu', str(support), '-o', str(so)]
        Path(str(so) + '.build_command.txt').write_text(shlex.join(command))
        Path(str(so) + '.ptxas.log').write_text(
            "ptxas info : Compiling entry function 'synthetic_kernel' for 'sm_89'\n"
            'ptxas info : Used 32 registers, 0 bytes spill stores, 0 bytes spill loads\n')
        plan = dict(pg='l2', phase='forward', attention_kv_block=64, attention_query_rows=1,
                    gemms=[], kappa=1, grid=1, residency=1, pages={})
        Path(str(so) + '.plan.json').write_text(json.dumps(plan))
        source = dict(root=str(base), head='synthetic source head', diff_sha256='synthetic diff',
                      compiler_binary_sha256=identity.sha(compiler),
                      source_files={'defaults.h': identity.sha(header),
                                    'support.cpp': identity.sha(support)})
        return so, source

    def test_complete_tables_verify_and_detect_tampering(self):
        with tempfile.TemporaryDirectory(prefix='dm identity macros ') as directory:
            so, source = self.fixture(Path(directory))
            record = identity.generate(so, source)
            self.assertEqual(identity.verify(so)['artifact_id'], record['artifact_id'])
            units = record['complete_macros']['translation_units']
            self.assertEqual(len(units), 4)
            self.assertTrue(all(unit['macros']['INCLUDED_DEFAULT'] == '17' for unit in units))
            self.assertTrue(all(unit['macros']['CLI_VALUE'] == '7' for unit in units))
            self.assertEqual({unit['phase'] for unit in units}, {'host', 'device_890'})
            table = Path(units[0]['macros_file'])
            saved = table.read_bytes(); table.write_bytes(saved + b'#define TAMPERED 1\n')
            with self.assertRaisesRegex(ValueError, 'macro table changed'):
                identity.verify(so)
            table.write_bytes(saved)
            self.assertEqual(identity.verify(so)['artifact_id'], record['artifact_id'])
            # Historical identities keep their recorded schema and artifact key.
            del record['complete_macros']; del record['artifact_id']
            record['artifact_id'] = hashlib.sha256(json.dumps(
                record, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            Path(str(so) + '.identity.json').write_text(json.dumps(record))
            self.assertEqual(identity.verify(so)['artifact_id'], record['artifact_id'])

    def test_before_build_input_drift_rejects(self):
        with tempfile.TemporaryDirectory(prefix='dm identity drift ') as directory:
            base = Path(directory); so, source = self.fixture(base)
            (base / 'defaults.h').write_text('#define INCLUDED_DEFAULT 18\n')
            with self.assertRaisesRegex(ValueError, 'build source changed'):
                identity.generate(so, source)
            self.assertFalse(Path(str(so) + '.macro_capture').exists())
            (base / 'defaults.h').write_text('#define INCLUDED_DEFAULT 17\n')
            (base / 'build-dm/tools/tilemega').write_bytes(b'changed synthetic compiler')
            with self.assertRaisesRegex(ValueError, 'plan compiler changed'):
                identity.generate(so, source)
            self.assertFalse(Path(str(so) + '.identity.json').exists())


if __name__ == '__main__':
    unittest.main()
