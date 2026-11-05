"""Replay diagnostics cannot overwrite snapshots or run without CPU guards."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'scripts'))
import dnn1_tvm_trace_replay as replay


class ReplayGuardTest(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);self.snapshot=self.root/'immutable';self.snapshot.mkdir()

    def call(self,out,env):
        args=['replay','--snapshot',str(self.snapshot),'--tvm-root',str(self.root/'source'),
              '--out',str(out),'--expected','success']
        with patch.object(sys,'argv',args), patch.dict(replay.os.environ,env,clear=True):
            return replay.main()

    def test_snapshot_and_nested_output_are_rejected_before_any_import(self):
        for out in (self.snapshot,self.snapshot/'nested'):
            with self.assertRaisesRegex(ValueError,'immutable snapshot'):
                self.call(out,{})
        self.assertEqual(list(self.snapshot.iterdir()),[])

    def test_existing_output_is_retained_without_overwrite(self):
        out=self.root/'retained';out.mkdir();marker=out/'receipt.json';marker.write_text('old evidence')
        with self.assertRaises(FileExistsError):
            self.call(out,{'CUDA_VISIBLE_DEVICES':'','LD_PRELOAD':'reject_cuda.so'})
        self.assertEqual(marker.read_text(),'old evidence')

    def test_missing_cpu_guard_cannot_create_output(self):
        out=self.root/'new'
        with self.assertRaisesRegex(ValueError,'CPU replay requires'):
            self.call(out,{'CUDA_VISIBLE_DEVICES':'0'})
        self.assertFalse(out.exists())


if __name__=='__main__':unittest.main()
