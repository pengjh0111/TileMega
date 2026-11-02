"""CPU lifecycle checks for the explicitly owned TP stream adapter."""
from contextlib import nullcontext
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tilemega.serving.tp_stream import OwnedTpStream
from tilemega.serving.tp_engine import TpServingEngine


class Runtime:
    def __init__(self):
        self.calls = []
    def create(self, device):
        self.calls.append(('create', device))
        return 0x12340
    def destroy(self, device, handle):
        self.calls.append(('destroy', device, handle))


class OwnedStreamTests(unittest.TestCase):
    @patch('tilemega.serving.tp_stream.torch.cuda.device', side_effect=lambda d: nullcontext())
    def test_sync_then_destroy_exactly_once(self, device):
        sdk = Runtime()
        def wrap(handle, *, device):
            return SimpleNamespace(cuda_stream=handle, synchronize=lambda: sdk.calls.append(('sync', device)))
        owned = OwnedTpStream(2, sdk=sdk, wrap=wrap)
        owned.close()
        owned.close()
        self.assertEqual(sdk.calls, [('create', 2), ('sync', 2), ('destroy', 2, 0x12340)])

    @patch('tilemega.serving.tp_stream.torch.cuda.device', side_effect=lambda d: nullcontext())
    def test_wrapper_failure_frees_unused_stream(self, device):
        sdk = Runtime()
        def wrap(*args, **kwargs):
            raise ValueError('wrapper failed')
        with self.assertRaisesRegex(ValueError, 'wrapper failed'):
            OwnedTpStream(1, sdk=sdk, wrap=wrap)
        self.assertEqual(sdk.calls, [('create', 1), ('destroy', 1, 0x12340)])

    @patch('tilemega.serving.tp_stream.torch.cuda.device', side_effect=lambda d: nullcontext())
    def test_poisoned_cohort_abandons_live_stream(self, device):
        sdk = Runtime()
        owned = OwnedTpStream(0, sdk=sdk, wrap=lambda handle, **kw: SimpleNamespace(cuda_stream=handle))
        owned.abandon()
        owned.close()
        self.assertEqual(sdk.calls, [('create', 0)])

    @patch('tilemega.serving.tp_stream.torch.cuda.device', side_effect=lambda d: nullcontext())
    def test_sync_failure_does_not_destroy_live_work(self, device):
        sdk = Runtime()
        def fail():
            raise RuntimeError('sync failed')
        owned = OwnedTpStream(0, sdk=sdk, wrap=lambda handle, **kw: SimpleNamespace(cuda_stream=handle, synchronize=fail))
        with self.assertRaisesRegex(RuntimeError, 'sync failed'):
            owned.close()
        self.assertEqual(sdk.calls, [('create', 0)])


class CohortCloseTests(unittest.TestCase):
    @patch('tilemega.serving.tp_engine.torch.cuda.device', side_effect=lambda d: nullcontext())
    def test_ready_cohort_closes_stream_before_plan(self, device):
        calls = []
        engine = TpServingEngine.__new__(TpServingEngine)
        engine.poisoned = False
        engine.ranks = [dict(device=d,
            stream_owner=SimpleNamespace(close=lambda d=d: calls.append(('stream', d))),
            plan=SimpleNamespace(close=lambda d=d: calls.append(('plan', d)))) for d in range(2)]
        engine.close()
        self.assertEqual(calls, [('stream', 0), ('plan', 0), ('stream', 1), ('plan', 1)])
        self.assertEqual(engine.ranks, [])

    @patch('tilemega.serving.tp_engine.torch.cuda.device', side_effect=lambda d: nullcontext())
    def test_one_busy_peer_preserves_all_receive_buffers(self, device):
        calls = []
        engine = TpServingEngine.__new__(TpServingEngine)
        engine.poisoned = True
        ranks = [dict(device=d, stream=SimpleNamespace(query=lambda d=d: d == 0),
            stream_owner=SimpleNamespace(abandon=lambda d=d: calls.append(('abandon', d))),
            plan=SimpleNamespace(handle=object())) for d in range(2)]
        engine.ranks = ranks
        engine.close()
        self.assertEqual(calls, [('abandon', 0), ('abandon', 1)])
        self.assertTrue(all(rank['plan'].handle is None for rank in ranks))
        self.assertEqual(engine.ranks, [])


if __name__ == '__main__':
    unittest.main()
