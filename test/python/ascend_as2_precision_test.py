"""Recipe precision changes cannot reuse replay tables, images or prices."""
import unittest
import tempfile
from pathlib import Path
from tilemega.bench.as2_models import RUNS
from tilemega.bench.as2_request import source_parts, compatible_source
from tilemega.bench.as2_precision import REPLAY, STREAMING, STREAMING_FP32, validate_plan_precision
from tilemega.bench.as2_body_profile import logical_bytes

class PrecisionTest(unittest.TestCase):
    def stage(self):
        s=[0]*33;s[17]=13;s[23:28]=[8,128,3,1088,48];s[31]=10
        return s

    def test_precision_requires_real_variant_and_legal_shape(self):
        s=self.stage()
        self.assertEqual(validate_plan_precision(STREAMING,[s]),2)
        with self.assertRaises(ValueError):validate_plan_precision(REPLAY,[s])
        with self.assertRaises(ValueError):validate_plan_precision(STREAMING,[])
        replay=s.copy();replay[31]=6
        self.assertEqual(validate_plan_precision(REPLAY,[replay]),1)
        with self.assertRaises(ValueError):validate_plan_precision(STREAMING,[s,replay])
        for index,value in [(23,0),(24,32),(25,0),(25,5),(26,128),(27,47)]:
            v=s.copy();v[index]=value
            with self.assertRaises(ValueError):validate_plan_precision(STREAMING,[v])

    def test_precision_cannot_cross_source_or_probability_implementations(self):
        s=self.stage();s[31]=11
        self.assertEqual(validate_plan_precision(STREAMING_FP32,[s]),3)
        with self.assertRaises(ValueError):validate_plan_precision(STREAMING,[s])
        with tempfile.TemporaryDirectory(dir=RUNS/'probes') as raw:
            root=Path(raw)
            geometry='#define TILEMEGA_LLM_VARIANT 1\n#define TILEMEGA_LLM_TM 16\n#define TILEMEGA_LLM_TN 128\n#define TILEMEGA_LLM_TK 128\n'
            for contract in (1,2,3):
                p=root/f'{contract}.cpp'
                p.write_text(geometry+f'#define TILEMEGA_LLM_PRECISION_CONTRACT {contract}\n'+
                    f'#define TILEMEGA_LLM_STREAMING_ATTENTION {int(contract>1)}\n#include <same_runtime.h>\n')
                self.assertEqual(source_parts(p)[1]['TILEMEGA_LLM_PRECISION_CONTRACT'],contract)
            for a,b in ((1,2),(2,3),(3,1)):
                with self.assertRaises(ValueError):compatible_source(root/f'{a}.cpp',root/f'{b}.cpp')
            (root/'wrong.cpp').write_text(geometry+'#define TILEMEGA_LLM_PRECISION_CONTRACT 3\n#include <same_runtime.h>\n')
            with self.assertRaises(ValueError):source_parts(root/'wrong.cpp')

    def test_streaming_traffic_tracks_valid_keys_without_capacity_planes(self):
        s=self.stage();m=[0]*15;m[9]=1
        early=logical_bytes(s,m,0,64);late=logical_bytes(s,m,0,1086)
        self.assertEqual(late-early,4*3*(1086-64)*128)
        self.assertGreater(early,0)
        m[9]=64
        self.assertGreater(logical_bytes(s,m,8,0),logical_bytes(s,m,0,0))

if __name__=='__main__':unittest.main()
