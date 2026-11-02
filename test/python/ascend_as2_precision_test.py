"""Recipe precision changes cannot reuse replay tables, images or prices."""
import unittest
from tilemega.bench.as2_precision import REPLAY, STREAMING, validate_plan_precision
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

    def test_streaming_traffic_tracks_valid_keys_without_capacity_planes(self):
        s=self.stage();m=[0]*15;m[9]=1
        early=logical_bytes(s,m,0,64);late=logical_bytes(s,m,0,1086)
        self.assertEqual(late-early,4*3*(1086-64)*128)
        self.assertGreater(early,0)
        m[9]=64
        self.assertGreater(logical_bytes(s,m,8,0),logical_bytes(s,m,0,0))

if __name__=='__main__':unittest.main()
