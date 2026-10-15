"""A child cwd must not redirect the scheduler's raw evidence directory."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class EvidencePath(unittest.TestCase):
    def test_cpu_child_receives_an_absolute_attempt_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'queue').mkdir();(root/'child').mkdir()
            (root/'policy.json').write_text('{}')
            job=dict(name='cpu_evidence',gpu=False,timing=False,timeout_s=10,
                cwd=str(root/'child'),command=[sys.executable,'-c',
                    'from pathlib import Path; import sys; p=Path(sys.argv[1]); assert p.is_absolute(); (p/"evidence.txt").write_text("ok")',
                    '{attempt_dir}'])
            (root/'queue/queue_test.json').write_text(json.dumps([job]))
            subprocess.run([sys.executable,'-m','tilemega.bench.scheduler',
                '--queue-dir','queue','--out','scheduler','--policy','policy.json'],
                cwd=root,env={**os.environ,'PYTHONPATH':str(ROOT/'python')},
                stdout=subprocess.PIPE,stderr=subprocess.STDOUT,check=True,timeout=20)
            state=json.loads((root/'scheduler/state.json').read_text())['cpu_evidence']
            self.assertEqual(state['status'],'done')
            self.assertEqual(Path(state['out']),root/'raw/cpu_evidence/attempt1')
            self.assertEqual((Path(state['out'])/'evidence.txt').read_text(),'ok')
            self.assertFalse((root/'child/raw').exists())


if __name__=='__main__':unittest.main()
