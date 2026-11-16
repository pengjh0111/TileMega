"""Missing CTest executables must never become passing regression evidence."""
from pathlib import Path
import sys
import unittest
import xml.etree.ElementTree as ET

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from dnn1_ctest import classify


class CtestEvidence(unittest.TestCase):
    def test_ctest_notrun_distinguishes_skip_code_from_missing_binary(self):
        tests = ET.fromstring('''<testsuite>
          <testcase name="passed" status="run"/>
          <testcase name="tp_disabled" status="notrun"><skipped message="SKIP_RETURN_CODE=77"/></testcase>
          <testcase name="missing_dense" status="notrun"><skipped message="Unable to find executable"/></testcase>
          <testcase name="failed_contract" status="fail"><failure/></testcase>
          <testcase name="launcher_error"><error/></testcase>
          <testcase name="unexplained_notrun" status="notrun"/>
        </testsuite>''')
        result = classify(tests)
        self.assertEqual(result['skipped'], ['tp_disabled'])
        self.assertEqual(result['not_run'], ['missing_dense', 'unexplained_notrun'])
        self.assertEqual(result['unsuccessful'],
            ['failed_contract', 'launcher_error', 'missing_dense', 'unexplained_notrun'])


if __name__ == '__main__':
    unittest.main()
