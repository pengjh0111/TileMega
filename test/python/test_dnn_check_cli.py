import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tilemega.dnn import check as checker


class DnnCheckCliTest(unittest.TestCase):
    def test_nested_output_preserves_both_pass_and_failure_receipts(self):
        for passed in (True, False):
            with self.subTest(passed=passed), tempfile.TemporaryDirectory() as folder:
                out=Path(folder)/'nested/model/correctness.json'
                args=['--model','mbv1','--library','model.so','--export','export',
                      '--bridge','bridge.json','--data','data','--batch','2','--out',str(out)]
                result=dict(passed=passed,model='mbv1',samples=1000,
                            artifact_id='identity',metrics={'cosine':.999})
                def run(parsed):
                    self.assertTrue(out.parent.is_dir())
                    return result
                with patch.object(checker,'check',side_effect=run), contextlib.redirect_stdout(io.StringIO()):
                    if passed:
                        checker.main(args)
                    else:
                        with self.assertRaises(SystemExit) as exit:
                            checker.main(args)
                        self.assertEqual(exit.exception.code,1)
                self.assertEqual(json.loads(out.read_text()),result)


if __name__=='__main__':
    unittest.main()
