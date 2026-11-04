import unittest
from tilemega.dnn.debug_buffers import materialized_buffers

class NativeDebugTest(unittest.TestCase):
    def test_only_written_buffers_are_retained(self):
        source='''struct Bind_0 {
 kOutput=1u,kRowStats=3u,kChannelPartial=kAbsent;
};
struct Bind_1 {
 kOutput=4u,kRowStats=kAbsent,kChannelPartial=5u;
};'''
        manifest=dict(native_forward={},stages=[{},{}],buffers=[dict(name=str(i),role='external' if i==0 else 'internal') for i in range(6)])
        manifest['native_forward']=True
        value=materialized_buffers(source,manifest)
        self.assertEqual(set(value),{'0','1','3','4','5'})
        self.assertEqual(value['5']['writers'],[1])
        with self.assertRaises(ValueError):materialized_buffers(source.replace('kOutput=1u','kOutput=6u'),manifest)
        with self.assertRaises(ValueError):materialized_buffers(source.replace('Bind_1','Bind_3'),manifest)

if __name__=='__main__':unittest.main()
