"""Check C/Python forward metadata and step-zero restrictions without CUDA."""
import ctypes
from pathlib import Path
import subprocess
import tempfile
import unittest

from tilemega.serving.plan import FORWARD, PlanLibrary

ROOT = Path(__file__).resolve().parents[2]
SOURCE = r'''
#include <tilemega/Codegen/tasks/ServingAbi.h>
#ifndef SEQ
#define SEQ 1
#endif
#ifndef CAPACITY
#define CAPACITY 0
#endif
int tm_plan_query(tm_plan_info* p) {
  *p = (tm_plan_info){TM_SERVING_ABI_VERSION, TM_SERVING_FORWARD, 1, 64,
      SEQ, 0, 0, CAPACITY, 1, 1, TM_SERVING_L1 | TM_SERVING_L2, 1}; return 0;
}
int tm_plan_buffer(uint32_t i, tm_buffer_info* p) {
  if (i) return -1;
  *p = (tm_buffer_info){"input", TM_BUFFER_EXTERNAL, TM_DTYPE_BF16, 0, 384, 0}; return 0;
}
void* tm_plan_create(int b, void* const* p, int d) { return p[0] ? (void*)1 : 0; }
int tm_plan_set_steps(void* p, const int32_t* past, uint32_t n) {
  return n == 1 && past[0] == 0 ? 0 : -1;
}
int tm_plan_launch(void* p, uint32_t step, uint32_t mode, uint64_t i, void* s) {
  return step == 0 ? 0 : -1;
}
void tm_plan_destroy(void* p) {}
'''


class ForwardAbiTests(unittest.TestCase):
    def library(self, folder, seq=1, capacity=0):
        source = Path(folder) / f'forward-{seq}-{capacity}.c'
        binary = source.with_suffix('.so')
        source.write_text(SOURCE)
        subprocess.run(['cc', '-std=c11', '-shared', '-fPIC', '-I' + str(ROOT / 'include'),
                        f'-DSEQ={seq}', f'-DCAPACITY={capacity}', str(source), '-o', str(binary)],
                       check=True, capture_output=True)
        return PlanLibrary(binary)

    def test_forward_metadata_for_images_encoder_and_moe(self):
        with tempfile.TemporaryDirectory() as folder:
            for seq in (1, 128, 4096):
                lib = self.library(folder, seq)
                self.assertEqual(lib.info.phase, FORWARD)
                self.assertEqual(lib.info.seq, seq)
                self.assertEqual((lib.info.past_lo, lib.info.past_hi, lib.info.capacity), (0, 0, 0))
                self.assertEqual(lib.buffers[0].name, 'input')
                self.assertEqual(lib.buffers[0].elements_per_batch, 384)

    def test_only_step_zero_advances_iterations(self):
        with tempfile.TemporaryDirectory() as folder:
            lib = self.library(folder)
            with lib.create(8, {'input': 123}, 0) as plan:
                plan.set_steps([0])
                for steps in ([], [1], [0, 0]):
                    with self.assertRaises(ValueError): plan.set_steps(steps)
                with self.assertRaises(ValueError): plan.launch(1, 1, 0)
                self.assertEqual(plan.iteration, {1: 0, 2: 0})
                plan.launch(0, 1, 0); plan.launch(0, 2, 0)
                self.assertEqual(plan.iteration, {1: 1, 2: 1})

    def test_forward_rejects_kv_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(RuntimeError): self.library(folder, capacity=128)
            with self.assertRaises(RuntimeError): self.library(folder, seq=0)


if __name__ == '__main__':
    unittest.main()
