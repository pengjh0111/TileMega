#!/usr/bin/env python3
"""CPU-only validation of the manual runner; never invokes a GPU binary."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("runner", Path(__file__).with_name("run_schedule_sm120.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def test_records(self):
        self.assertEqual(runner.record("E2E_TIME l2_ms=1\n", "E2E_TIME", ["l2_ms"]), {"l2_ms": "1"})
        for text in ("", "E2E_TIME l1_ms=1\n", "E2E_TIME l2_ms=1\nE2E_TIME l2_ms=2\n"):
            with self.assertRaises(ValueError):
                runner.record(text, "E2E_TIME", ["l2_ms"])

    def test_complete_matrix_and_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "fixture").mkdir()
            (root / "binary").write_text("test-only, never executed\n")
            (root / "binary").chmod(0o700)
            (root / "source").write_text("test fixture\n")
            (root / "ptxas").write_text("Used 218 registers\n0 bytes spill stores, 0 bytes spill loads\n")
            cases = []
            for state in ("stage_major", "affine_balanced"):
                (root / state).write_text(json.dumps(dict(state=state,total_ns=1,event_ns=1,ctas_per_sm=2)))
                for model in ("gqa2", "mha4"):
                    for seq in (4, 128):
                        case = dict(model=model,seq=seq,group="placement",state=state,fixture="fixture")
                        for field in ("binary", "source", "ptxas", "prediction"):
                            path = state if field == "prediction" else field
                            case[field] = path
                            case[field + "_sha256"] = runner.digest(root / path)
                        cases.append(case)
            manifest = dict(dtype="bf16",arch="sm_120",cases=cases)
            self.assertEqual(len(runner.validate(manifest, root, "placement")), 8)
            with self.assertRaises(ValueError):
                runner.validate(dict(manifest,cases=cases[:-1]), root, "placement")
            with self.assertRaises(ValueError):
                runner.validate(dict(manifest,cases=cases+[cases[0]]), root, "placement")
            with self.assertRaises(ValueError):
                runner.validate(dict(manifest,dtype="fp32"), root, "placement")
            (root / "binary").write_text("changed\n")
            with self.assertRaises(ValueError):
                runner.validate(manifest, root, "placement")


if __name__ == "__main__":
    unittest.main()
