"""Queue coverage and safety gates, without importing a GPU framework."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SOURCE = Path(__file__).resolve().parents[2]/"scripts/run_r16_baselines.py"
spec = importlib.util.spec_from_file_location("r16_baseline_queue_driver", SOURCE)
driver = importlib.util.module_from_spec(spec)
spec.loader.exec_module(driver)


class BaselineQueueTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)/"repo"
        self.envs = Path(self.temp.name)/"envs"
        for model in driver.MODELS:
            path = self.root/"runs/r16/models"/model
            path.mkdir(parents=True)
            (path/"config.json").write_text('{"vocab_size":32}')
        prompts = self.root/"configs/prompts"; prompts.mkdir(parents=True)
        for name in ("llama3_isl64.json", "qwen_isl64.json"):
            (prompts/name).write_text(json.dumps([[1]*64 for _ in range(16)]))
        harnesses = self.root/"python/tilemega/serving"; harnesses.mkdir(parents=True)
        for system in driver.SYSTEMS:
            (harnesses/f"{system}_baseline.py").write_text(f"# {system}\n")
        for env in set(driver.ENVIRONMENTS.values()):
            path = self.envs/env/"bin"; path.mkdir(parents=True)
            (path/"python").touch()

    def test_complete_grid_has_three_fresh_processes_and_interleaves_systems(self):
        plan = driver.build_plan(self.root, self.envs)
        jobs = [job for card in plan["m1"].values() for job in card]
        self.assertEqual(len(jobs), 72)
        self.assertEqual(len({job["name"] for job in jobs}), 72)
        coverage = {}
        for job in jobs:
            command = job["command"]
            model = Path(command[command.index("--model")+1]).name
            harness = Path(command[command.index("-u")+1]).stem
            rnd = int(command[command.index("--round-index")+1])
            coverage.setdefault((model, harness), set()).add(rnd)
            self.assertEqual(command[command.index("--batch")+1], "all")
            self.assertEqual(command[command.index("--max-tokens")+1], "1024")
            self.assertEqual(command[command.index("--warmup")+1], "1")
            self.assertEqual(command[command.index("--repeats")+1], "3")
            self.assertNotIn("after", job)
            self.assertIn("--membind=1", command)
            self.assertEqual(job["env"]["TILEMEGA_GPU_PID_MODE"], "strict")
            self.assertEqual(job["env"]["CUDA_VISIBLE_DEVICES"], job["env"]["TILEMEGA_DEVICE_INDEX"])
        self.assertEqual(len(coverage), 24)
        self.assertTrue(all(rounds == {0, 1, 2} for rounds in coverage.values()))
        firsts = []
        for rnd in range(3):
            items = [job for job in jobs if job["protocol"]["fresh_process_round"] == rnd]
            firsts.append(min(items, key=lambda job: job["priority"])["name"])
        self.assertTrue("vllm" in firsts[0] and "sglang" in firsts[1] and "trtllm" in firsts[2])

    def test_m0_trials_and_recovery_are_preview_only(self):
        plan = driver.build_plan(self.root, self.envs, pid_mode="namespace_limited", cpu_bind_only=True)
        self.assertEqual(sum(map(len, plan["anchors_start"].values())), 6)
        for jobs in plan["m0_concurrent"].values():
            for job in jobs:
                self.assertEqual(job["env"]["TILEMEGA_NUMA_CONCURRENCY_TRIAL"], "1")
                self.assertNotIn("TILEMEGA_NUMA_CONCURRENCY_VERIFIED", job["env"])
                self.assertNotIn("--membind=1", job["command"])
        self.assertEqual(set(plan["recovery_preview"]), {6})
        self.assertEqual(len(plan["recovery_preview"][6]), 2)
        out = self.root/"preview"
        manifest = driver.write_plan(out, plan, self.root, "namespace_limited", True)
        self.assertFalse(manifest["enqueued"])
        self.assertIsNone(manifest["ownership_verified"])
        self.assertEqual(manifest["g_m0"], "not_run")
        with self.assertRaises(FileExistsError):
            driver.write_plan(out, plan, self.root, "namespace_limited", True)

    def test_recovery_avoids_mpi_spawn_and_keeps_long_generation_protocol(self):
        plan = driver.build_recovery_plan(self.root, self.envs)
        jobs = plan["recovery_mpi_and_prewarm"]
        self.assertEqual(jobs[6][0]["env"]["TLLM_WORKER_USE_SINGLE_PROCESS"], "1")
        self.assertEqual(jobs[6][0]["timeout_s"], 300)
        self.assertEqual(jobs[6][0]["priority"], 0)
        counts = {job["protocol"]["osl"] for job in jobs[7]}
        self.assertEqual(counts, {64,1024})
        self.assertTrue(all(job["protocol"]["repeats"] == 3 for job in jobs[7]))
        self.assertEqual(sum(map(len,plan["m1_recovered"].values())), 36)

    def test_invalid_device_topology_and_prompt_vocabulary_are_rejected(self):
        with self.assertRaises(ValueError):
            driver.build_plan(self.root, self.envs, gpus=(7, 7))
        with self.assertRaises(ValueError):
            driver.build_plan(self.root, self.envs, gpus=(0,))
        (self.root/"configs/prompts/llama3_isl64.json").write_text(json.dumps([[32]*64 for _ in range(16)]))
        with self.assertRaises(ValueError):
            driver.build_plan(self.root, self.envs)


if __name__ == "__main__":
    unittest.main()
