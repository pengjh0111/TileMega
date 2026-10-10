"""Resolve the router buffer through both untouched export spellings."""
import copy
import json
from pathlib import Path
import unittest

from tilemega.moe.check_generated import router_output


class RouterBuffer(unittest.TestCase):
    def test_before_and_core(self):
        fixtures=Path(__file__).resolve().parents[1]/'fixtures/moe'
        for spelling,target in [('before','aten.linear.default'),('core','aten.mm.default')]:
            with self.subTest(spelling=spelling):
                bridge=json.loads((fixtures/('region_'+spelling+'.json')).read_text())
                gate=next(item['target'] for item in bridge['signature']['inputs']
                    if item['kind']=='PARAMETER' and item['target'].endswith('mlp.gate.weight'))
                name=router_output(bridge,gate)
                node=next(item for item in bridge['nodes'] if item['name']==name)
                self.assertEqual(node['target'],target)
                bad=copy.deepcopy(bridge)
                bad['nodes'].append(dict(node,name='second_router'))
                with self.assertRaises(ValueError):router_output(bad,gate)
                with self.assertRaises(ValueError):router_output(bridge,'missing.gate.weight')


if __name__=='__main__':unittest.main()
