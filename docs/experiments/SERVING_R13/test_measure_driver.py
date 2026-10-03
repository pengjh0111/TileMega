#!/usr/bin/env python3
"""Exercise launch and event counts without creating a CUDA context."""
from types import SimpleNamespace
from unittest.mock import Mock, patch
import unittest
from tilemega.serving import engine as module

class Tensor:
    shape = (1, 64)
    def to(self, **_): return self
    def contiguous(self): return self
    def is_pinned(self): return True
    def __getitem__(self, _): return self
    def copy_(self, *_, **__): return self

class DriverTests(unittest.TestCase):
    def generate(self, step_events, loop):
        events = []
        class Event:
            def __init__(self, **_): events.append(self)
            def record(self, _): pass
            def elapsed_time(self, _): return 1.0
        stream = SimpleNamespace(cuda_stream=0, synchronize=lambda: None)
        fake = SimpleNamespace(int32=0, empty=lambda *a, **k: Tensor(),
                               cuda=SimpleNamespace(Event=Event,
                                                    current_stream=lambda: stream))
        engine = module.ServingEngine.__new__(module.ServingEngine)
        engine.max_new_tokens=4; engine.batch=1; engine.prompt_len=64
        engine.decode_mode=2; engine.prefill_mode=1; engine.decode_loop=loop
        engine.decode_chunk=None; engine.step_events=step_events
        engine.decode_lib=SimpleNamespace(lib=SimpleNamespace(tm_plan_launch_steps=True))
        engine.prefill=Mock(); engine.decode=Mock()
        engine.decode.read_step_ns.return_value=[0, 1000000, 2000000, 3000000]
        engine.state=SimpleNamespace(tokens=Tensor())
        with patch.object(module, 'torch', fake): result=engine.generate(Tensor())
        return engine, result, events

    def test_without_step_events_keeps_ttft(self):
        engine,result,events=self.generate(False,False)
        self.assertEqual(len(events),3)  # start, TTFT, final
        self.assertEqual(result.step_ms,[1.0])
        self.assertEqual(engine.decode.launch.call_count,3)
        self.assertFalse(result.decode_loop_used)
        self.assertFalse(result.step_ns_read)
        self.assertEqual(engine.prefill.launch.call_args.args[1],1)

    def test_events_and_loop_metadata(self):
        engine,result,events=self.generate(True,False)
        self.assertEqual(len(events),6)
        self.assertEqual(len(result.step_ms),4)
        engine,result,events=self.generate(True,True)
        self.assertTrue(result.decode_loop_used)
        self.assertTrue(result.step_ns_read)
        self.assertEqual(engine.decode.launch_steps.call_count,1)
        self.assertEqual(len(result.step_ms),4)

    def test_loop_without_events_does_not_read_timestamps(self):
        engine,result,_=self.generate(False,True)
        self.assertTrue(result.decode_loop_used)
        self.assertFalse(result.step_ns_read)
        engine.decode.read_step_ns.assert_not_called()

if __name__=='__main__': unittest.main()
