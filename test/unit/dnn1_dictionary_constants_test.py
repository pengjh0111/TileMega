import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

MODULE = Path(__file__).resolve().parents[2] / 'scripts/dnn1_dictionary_constants.py'
spec = importlib.util.spec_from_file_location('dictionary_constants', MODULE)
codec = importlib.util.module_from_spec(spec)
spec.loader.exec_module(codec)


class DictionaryConstantsTest(unittest.TestCase):
    def section(self):
        return dict(schema=2, variant='unchanged', processes=5,
            keys=[['a', 'tm=032;factor=1e-3', 'ns'], ['a', 'tm=64;factor=0.001', 'ns'],
                  ['b', 'factor=1e-3;tm=032', 'ns/k64']],
            median_by_gpu={g:[float.fromhex('0x1.0000000000001p+0'), 2.0, 3.0]
                           for g in ('0', '5', '7')},
            unstable=[dict(gpu=5, metric='a', key='tm=032;factor=1e-3')],
            supplements=[dict(variant='source', key_indices=[1, 2])])

    def test_exact_strings_values_indices_and_metadata(self):
        source=self.section();saved=copy.deepcopy(source);encoded=codec.encode(source)
        self.assertEqual(source, saved)
        self.assertEqual(len(encoded['key_layouts']), 2)
        self.assertEqual(encoded['keys'][0], [0, '032', '1e-3'])
        for field in ('variant', 'median_by_gpu', 'unstable', 'supplements'):
            self.assertEqual(encoded[field], source[field])

    def test_invalid_selectors(self):
        for key in ('tm=32;tm=64', '=32', 'tm=', 'tm'):
            source=self.section();source['keys'][0][1]=key
            with self.assertRaises(ValueError):codec.encode(source)

    def test_native_prefix_and_exclusive_output(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'input.json';output=Path(directory)/'output.json'
            prefix='{\n  "sm90_native_constants": {"literal": 1.0000000000000000}'
            source.write_text(prefix+',\n  "sm90_forward_constants": '+json.dumps(self.section())+'\n}\n')
            proof=codec.convert(source,output);self.assertEqual(proof['medians'],9)
            self.assertTrue(output.read_text().startswith(prefix+',\n'))
            original=output.read_bytes()
            with self.assertRaises(FileExistsError):codec.convert(source,output)
            self.assertEqual(output.read_bytes(),original)


if __name__=='__main__':unittest.main()
