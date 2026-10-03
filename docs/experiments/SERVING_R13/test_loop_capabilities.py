#!/usr/bin/env python3
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import Mock
import json,unittest
from tilemega.serving.plan import Plan
class LoopCapabilities(unittest.TestCase):
    def test_new_binary_uses_plan_specific_mask(self):
        query=Mock(return_value=1)
        plan=Plan(SimpleNamespace(lib=SimpleNamespace(tm_plan_loop_modes=query)),42)
        self.assertEqual(plan.loop_modes(),1);query.assert_called_once_with(42)
    def test_old_binary_fallback_requires_pages(self):
        with TemporaryDirectory() as tmp:
            so=Path(tmp)/'plan.so';manifest=Path(str(so)+'.plan.json')
            library=SimpleNamespace(path=so,lib=SimpleNamespace(),info=SimpleNamespace(phase=1))
            p=Plan(library,1)
            self.assertEqual(p.loop_modes(),0)
            manifest.write_text(json.dumps({'pg':'pages'}));self.assertEqual(p.loop_modes(),2)
            library.info.phase=0;self.assertEqual(p.loop_modes(),0)
if __name__=='__main__':unittest.main()
