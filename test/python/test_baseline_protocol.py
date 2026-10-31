"""CPU contract tests: invalid measurements cannot become accepted results."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SOURCE = Path(__file__).resolve().parents[2] / "python/tilemega/serving/baseline_common.py"
spec = importlib.util.spec_from_file_location("independent_baseline_common", SOURCE)
common = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)


class BaselineProtocolTest(unittest.TestCase):
    def test_guard_is_required_before_engine_import(self):
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            common.require_guard()
        with patch.dict(os.environ, {"TILEMEGA_DEVICE_INDEX": "3", "CUDA_VISIBLE_DEVICES": "4",
                                    "TILEMEGA_GPU_LOCK_HELD": "1"}, clear=True), self.assertRaises(ValueError):
            common.require_guard()
        with patch.dict(os.environ, {"TILEMEGA_DEVICE_INDEX": "3", "CUDA_VISIBLE_DEVICES": "3",
                                    "TILEMEGA_GPU_LOCK_HELD": "1"}, clear=True):
            self.assertEqual(common.require_guard(), ["3"])

    def test_prompt_and_token_contract(self):
        for ids in ([[1], [True]], [[1], [2, 3]], [[-1]], []):
            with self.assertRaises(ValueError):
                common.validate_prompts(ids, 1)
        self.assertEqual(common.validate_prompts([[1, 2], [3, 4]], 2), 2)
        for tokens in ([[1]], [[1, 2], [3, 4]], [[1, True]], [[1, -2]]):
            with self.assertRaises(AssertionError):
                common.check_tokens(tokens, 1, 2)
        common.check_tokens([[1, 2]], 1, 2)

    def test_tpot_and_throughput_are_request_based(self):
        rows = [{"N": n, "warmup": warmup, "wall_seconds": value}
                for n, warmup, value in ((1, True, 100), (1, False, 1), (4, False, 4))]
        result = common.summarize(rows, 2, 4)
        self.assertEqual(result["tpot_seconds"], 1)
        self.assertEqual(result["output_tokens_per_second"], 2)
        self.assertIsNone(common.summarize(rows, 2, 1)["tpot_seconds"])
        rows[-1]["wall_seconds"] = .5
        with self.assertRaises(common.InvalidMeasurement):
            common.summarize(rows, 2, 4)

    def test_telemetry_rejects_throttling_and_foreign_processes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, "require_guard", return_value=["2"]), \
                patch.object(common, "_own_pids", return_value={100}):
            path = Path(tmp)/"guard.jsonl"
            good = "400, 1800, 1593, 55, 1100, Not Active, Not Active, Not Active, Not Active, 0x0\n"
            with patch.object(common.subprocess, "check_output", side_effect=[good, "100, 1000\n"]):
                self.assertTrue(common.observe(path, "good")[0]["accepted"])
            with patch.object(common.subprocess, "check_output", side_effect=[
                    good.replace("Not Active", "Active", 1), "100, 1000\n"]):
                with self.assertRaises(common.InvalidMeasurement):
                    common.observe(path, "throttled")
            with patch.object(common.subprocess, "check_output", side_effect=[good, "200, 1000\n"]):
                with self.assertRaises(common.InvalidMeasurement):
                    common.observe(path, "foreign")
            self.assertEqual(len(path.read_text().splitlines()), 3)

    def test_namespace_limited_is_explicit_and_keeps_known_foreign_rejection(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, "require_guard", return_value=["2"]), \
                patch.object(common, "_own_pids", return_value={100}):
            path = Path(tmp)/"guard.jsonl"
            good = "400, 1800, 1593, 55, 1100, Not Active, Not Active, Not Active, Not Active, 0x0\n"
            with patch.dict(os.environ, {"TILEMEGA_GPU_PID_MODE": "strict"}), \
                    patch.object(common.subprocess, "check_output", side_effect=[good, "99999999, 1000\n"]):
                with self.assertRaises(common.InvalidMeasurement):
                    common.observe(path, "strict-unknown")
            with patch.dict(os.environ, {"TILEMEGA_GPU_PID_MODE": "namespace_limited"}), \
                    patch.object(common.subprocess, "check_output", side_effect=[good, "99999999, 1000\n"]):
                sample = common.observe(path, "limited-unknown")[0]
                self.assertFalse(sample["ownership_verified"])
                self.assertEqual(sample["unresolved_host_pids"], [99999999])
                self.assertEqual(sample["acceptance"], "provisional_container_pid_namespace")
            with patch.dict(os.environ, {"TILEMEGA_GPU_PID_MODE": "namespace_limited"}), \
                    patch.object(common.subprocess, "check_output", side_effect=[good, f"{os.getpid()}, 1000\n"]):
                with self.assertRaises(common.InvalidMeasurement):
                    common.observe(path, "limited-known-foreign")
            with patch.object(common.subprocess, "check_output", side_effect=[
                    good.replace("0x0", "0x20"), "100, 1000\n"]):
                with self.assertRaises(common.InvalidMeasurement):
                    common.observe(path, "software-thermal-mask")

    def test_semantic_invalid_preserves_raw_timings_and_exits_without_retry(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); model = root/"model"; model.mkdir()
            (model/"config.json").write_text('{"vocab_size":8}')
            prompts = root/"prompts.json"; prompts.write_text('[[1,2]]')
            args = argparse.Namespace(batch="1", out=root/"results", model=model,
                prompt_ids=prompts, max_tokens=4, warmup=1, repeats=1, round_index=0,
                load_format="dummy", policy=None)
            with patch.object(common, "observe", return_value={}), \
                    patch.object(common, "require_guard", return_value=["2"]), \
                    patch.object(common.time, "perf_counter", side_effect=[0,9,10,13,20,21,30,32]):
                with self.assertRaises(common.InvalidTiming):
                    common.measure(args, [[1,2]], "fake", lambda rows,n: [[3]*n], {})
            folder = args.out/"B1"
            invalid = json.loads((folder/"invalid_measurements.json").read_text())
            self.assertFalse(invalid["retryable"])
            self.assertEqual(invalid["diagnostics"]["e2e_seconds"], 2)
            self.assertEqual(invalid["diagnostics"]["ttft_seconds"], 3)
            self.assertEqual(len(invalid["generation_runs"]), 4)
            self.assertEqual(len((folder/"raw_timings.jsonl").read_text().splitlines()), 4)
            self.assertFalse((folder/"measurements.json").exists())
        for error, code in ((common.InvalidTiming("semantic", e2e_seconds=2,ttft_seconds=3), 3),
                            (common.InvalidMeasurement("guard interference"), 75)):
            def fail():
                raise error
            with patch("builtins.print"), self.assertRaises(SystemExit) as raised:
                common.guarded_main(fail)
            self.assertEqual(raised.exception.code, code)

    def test_one_process_outputs_independent_n1_and_full_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root/"model"; model.mkdir(); (model/"config.json").write_text('{"vocab_size":8}')
            prompts = root/"prompts.json"; prompts.write_text('[[1,2]]')
            args = argparse.Namespace(batch="1", out=root/"results", model=model,
                prompt_ids=prompts, max_tokens=4, warmup=1, repeats=2, round_index=2,
                load_format="dummy", policy=None)
            calls = []
            def generate(rows, count):
                calls.append(count)
                return [[3]*count for _ in rows]
            clock = [0, 1, 2, 3, 4, 5, 10, 14, 20, 24, 30, 34]
            with patch.object(common, "observe", return_value={}), patch.object(common, "require_guard", return_value=["2"]), \
                    patch.object(common, "versions", return_value={}), \
                    patch.object(common.time, "perf_counter", side_effect=clock), patch("builtins.print"):
                common.measure(args, [[1, 2]], "fake", generate, {})
            summary = json.loads((args.out/"B1/measurements.json").read_text())
            self.assertEqual(calls, [1, 1, 1, 4, 4, 4])
            self.assertEqual(summary["round_index"], 2)
            self.assertEqual(summary["tpot_seconds"], 1)
            self.assertEqual(len(summary["generation_runs"]), 6)
            self.assertEqual(json.loads((args.out/"B1/tokens_N4_run2.json").read_text()), [[3]*4])


if __name__ == "__main__":
    unittest.main()
