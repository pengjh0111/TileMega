import tempfile
import unittest
from pathlib import Path

from capture_macros_dm import capture, macro_definitions, preprocessor_invocations


class MacroCaptureTests(unittest.TestCase):
    def test_preserve_function_empty_and_quoted_macros(self):
        self.assertEqual(macro_definitions('#define F(a, b) ((a) + (b))\n'
                                          '#define EMPTY\n#define STR "hello world"\n'),
                         {'F(a, b)': '((a) + (b))', 'EMPTY': '', 'STR': '"hello world"'})
        for text in ('', '#define A 1\n#define A 2\n', 'unparsed output\n'):
            with self.assertRaises(ValueError):
                macro_definitions(text)

    def test_require_real_paired_preprocessors(self):
        host = '#$ gcc -E -x c++ "/path with space/a.cu" -o h.ii\n'
        device = '#$ gcc -E -D __CUDA_ARCH__=890 -x c++ "/path with space/a.cu" -o d.ii\n'
        rows = preprocessor_invocations('#$ _HERE_=/usr/local/cuda/bin\n' + host + device)
        self.assertEqual([row['phase'] for row in rows], ['host', 'device_890'])
        self.assertEqual(rows[0]['source'], '/path with space/a.cu')
        for text in (host, device, host + device + host,
                     host + device.replace('gcc', 'sh'),
                     host + device.replace('d.ii', 'd.ii ; echo exposed'),
                     host + device.replace('d.ii', '$(secret)'),
                     host + device.replace('-o d.ii', ''),
                     host + device.replace('=890', '=unknown')):
            with self.assertRaises(ValueError):
                preprocessor_invocations(text)

    def test_actual_host_device_and_support_unit_defaults(self):
        with tempfile.TemporaryDirectory(prefix='dm macro capture ') as directory:
            base = Path(directory)
            header = base / 'defaults.h'
            header.write_text('#define INCLUDED_DEFAULT 17\n'
                              '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900\n'
                              '#define DEVICE_CAPABLE 1\n#else\n#define DEVICE_CAPABLE 0\n#endif\n')
            kernel = base / 'main.cu'
            support = base / 'support.cpp'
            kernel.write_text('#include "defaults.h"\n#define MAIN_ONLY 29\n')
            support.write_text('#include "defaults.h"\n#define SUPPORT_ONLY 31\n')
            for arch in (89, 100):
                command = ['/usr/local/cuda/bin/nvcc', '-std=c++17', f'-arch=sm_{arch}',
                           '-DCLI_VALUE=7', '-shared', '-Xcompiler=-fPIC', '-I' + str(base),
                           '-x', 'cu', str(kernel), '-x', 'cu', str(support),
                           '-o', str(base / f'never-compiled-{arch}.so')]
                record = capture(command, base / f'capture-{arch}')
                self.assertFalse(Path(command[-1]).exists())
                self.assertEqual(len(record['translation_units']), 4)
                for row in record['translation_units']:
                    definitions = row['macros']
                    self.assertEqual(definitions['INCLUDED_DEFAULT'], '17')
                    self.assertEqual(definitions['CLI_VALUE'], '7')
                    device = row['phase'].startswith('device_')
                    self.assertEqual(definitions['DEVICE_CAPABLE'], str(int(device and arch >= 90)))
                    if device:
                        self.assertEqual(definitions['__CUDA_ARCH__'], str(arch * 10))
                    else:
                        self.assertNotIn('__CUDA_ARCH__', definitions)
                    self.assertEqual('MAIN_ONLY' in definitions, row['source'] == str(kernel))
                    self.assertEqual('SUPPORT_ONLY' in definitions, row['source'] == str(support))


if __name__ == '__main__':
    unittest.main()
