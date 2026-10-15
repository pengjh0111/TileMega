"""CPU tests for fresh-process readiness; no GPU or framework imports."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE=Path(__file__).resolve().parents[2]/'python/tilemega/serving/baseline_common.py'
spec=importlib.util.spec_from_file_location('m0_baseline_common',SOURCE)
common=importlib.util.module_from_spec(spec);spec.loader.exec_module(common)


class M0BarrierTest(unittest.TestCase):
    def test_two_fresh_nonces_acknowledge_each_other(self):
        with tempfile.TemporaryDirectory() as directory, ThreadPoolExecutor(2) as executor:
            futures=[executor.submit(common.m0_pair_barrier,directory,gpu,(6,7),0,1,1,.005) for gpu in (6,7)]
            left,right=[future.result(timeout=2) for future in futures]
            self.assertEqual(left['peer_nonce'],right['nonce'])
            self.assertEqual(right['peer_nonce'],left['nonce'])
            self.assertEqual(left['run'],1)

    def test_dead_stale_and_wrong_run_peers_cannot_release(self):
        for case in ('dead','start_mismatch','old_nonce','wrong_run'):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as temporary:
                root=Path(temporary)/'round0/N1024-run1';root.mkdir(parents=True)
                peer=dict(gpu='7',pid=os.getpid(),process_start=common._process_start(os.getpid()),
                    nonce='old',round=0,run=1,participants=['6','7'],ready_against={'6':'old-own-nonce'})
                if case=='dead':peer.update(pid=99999999,process_start='old-start')
                if case=='start_mismatch':peer['process_start']='different-start'
                if case=='wrong_run':peer['run']=2
                (root/'gpu7.json').write_text(json.dumps(peer))
                with self.assertRaises(common.InvalidTiming) as raised:
                    common.m0_pair_barrier(temporary,6,(6,7),0,1,.03,.005)
                self.assertFalse(raised.exception.retryable)
                self.assertEqual(len(list(root.glob('gpu6.*.failed.json'))),1)

    def test_optional_path_is_disabled_for_normal_runs_and_warmup(self):
        args=argparse.Namespace(warmup=1,round_index=0)
        with patch.dict(os.environ,{},clear=True),patch.object(common,'require_guard') as guard:
            self.assertIsNone(common.optional_m0_barrier(args,1024,1));guard.assert_not_called()
        with patch.dict(os.environ,{'TILEMEGA_M0_BARRIER_DIR':'unused'},clear=True), \
                patch.object(common,'require_guard') as guard:
            self.assertIsNone(common.optional_m0_barrier(args,1,1))
            self.assertIsNone(common.optional_m0_barrier(args,1024,0));guard.assert_not_called()


if __name__=='__main__':unittest.main()
