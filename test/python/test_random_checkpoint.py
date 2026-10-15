"""Tiny CPU fixtures for shared random memfd checkpoints and their lifetime."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from safetensors import safe_open
from transformers import AutoModelForCausalLM, LlamaConfig, Qwen3Config, Qwen3MoeConfig

from tilemega.serving.random_checkpoint import RandomCheckpoint, describe, validate_live_checkpoint
from tilemega.serving.weights import Checkpoint


class RandomCheckpointTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def source(self, kind, tied=False):
        config_class = {"llama": LlamaConfig, "qwen": Qwen3Config, "moe": Qwen3MoeConfig}[kind]
        config = config_class(vocab_size=32, hidden_size=16, intermediate_size=32,
            num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1,
            head_dim=8, tie_word_embeddings=tied, max_position_embeddings=32,
            moe_intermediate_size=8, num_experts=4, num_experts_per_tok=2)
        source = self.root/f"config-{kind}-{tied}"
        config.save_pretrained(source)
        return source

    def test_hf_and_tilemega_read_identical_llama_and_qwen_tensors(self):
        for kind, tied in (("llama", False), ("llama", True), ("qwen", False), ("qwen", True), ("moe", False)):
            with self.subTest(kind=kind, tied=tied):
                source = self.source(kind, tied)
                destination = self.root/f"shared-{kind}-{tied}"
                info = describe(source)
                self.assertGreater(info["tensor_count"], 5)
                with RandomCheckpoint(source, destination, shard_bytes=700, chunk_bytes=113) as shared:
                    manifest = validate_live_checkpoint(destination)
                    self.assertEqual(manifest["state"], "live")
                    self.assertGreater(len(manifest["shards"]), 1)
                    checkpoint = Checkpoint(destination)
                    model = AutoModelForCausalLM.from_pretrained(str(destination),
                        torch_dtype=torch.bfloat16, local_files_only=True).eval()
                    self.assertEqual(set(checkpoint.files), {item["name"] for item in manifest["tensors"]})
                    for name, parameter in model.named_parameters(remove_duplicate=False):
                        actual = checkpoint.tensor(name)
                        self.assertTrue(torch.equal(actual, parameter.detach()), name)
                        self.assertTrue(torch.isfinite(actual).all(), name)
                        if "norm" in name and name.endswith("weight"):
                            self.assertTrue(torch.equal(actual, torch.ones_like(actual)), name)
                    if tied:
                        self.assertNotIn("lm_head.weight", checkpoint.files)
                        self.assertEqual(manifest["aliases"]["lm_head.weight"], "model.embed_tokens.weight")
                    with torch.inference_mode():
                        logits = model(torch.tensor([[1, 2, 3]])).logits
                    self.assertTrue(torch.isfinite(logits).all())
                    opened = list(shared.fds)
                for fd in opened:
                    with self.assertRaises(OSError):
                        os.fstat(fd)
                self.assertFalse(list(destination.glob("*.safetensors")))
                with self.assertRaisesRegex(RuntimeError, "closed"):
                    validate_live_checkpoint(destination)
                with self.assertRaises(FileNotFoundError):
                    Checkpoint(destination)

    def test_reproducible_chunks_and_sharding_and_seed_changes(self):
        source = self.source("llama")
        identities, values = [], []
        for run, options in enumerate((dict(seed=9, shard_bytes=700, chunk_bytes=13),
                                       dict(seed=9, shard_bytes=100000, chunk_bytes=8192),
                                       dict(seed=10, shard_bytes=700, chunk_bytes=13))):
            with RandomCheckpoint(source, self.root/f"seed-{run}", **options) as shared:
                identities.append(shared.manifest["checkpoint_sha256"])
                values.append(Checkpoint(shared.directory).tensor("model.embed_tokens.weight").clone())
        self.assertEqual(identities[0], identities[1])
        self.assertTrue(torch.equal(values[0], values[1]))
        self.assertNotEqual(identities[0], identities[2])
        self.assertFalse(torch.equal(values[0], values[2]))

    def test_cpu_subprocess_uses_proc_path_and_supervisor_cleans_up(self):
        source = self.source("qwen")
        destination = self.root/"supervisor"
        module = Path(__file__).resolve().parents[2]/"python/tilemega/serving/random_checkpoint.py"
        child = ("import os,json; from pathlib import Path; from safetensors import safe_open; "
                 "p=Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT']); "
                 "f=next(p.glob('*.safetensors')); "
                 "s=safe_open(str(f),framework='pt'); assert len(s.keys())>0; "
                 "(p/'child-read-ok.json').write_text(json.dumps(s.keys()))")
        result = subprocess.run([sys.executable, str(module), "--config", str(source),
            "--out", str(destination), "--seed", "3", "--", sys.executable, "-c", child],
            text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue((destination/"child-read-ok.json").is_file())
        self.assertEqual(json.loads((destination/"random_checkpoint.json").read_text())["state"], "closed")
        self.assertFalse(list(destination.glob("*.safetensors")))

    def test_invalid_limits_and_closed_supervisor_are_explicit(self):
        source = self.source("llama")
        for options in (dict(max_bytes=1), dict(shard_bytes=0), dict(chunk_bytes=0)):
            with self.assertRaises(ValueError):
                with RandomCheckpoint(source, self.root/"invalid", **options):
                    pass
        self.assertFalse((self.root/"invalid").exists())
        destination = self.root/"broken"
        with RandomCheckpoint(source, destination) as shared:
            link = next(destination.glob("*.safetensors"))
            link.unlink(); link.symlink_to("/proc/99999999/fd/99")
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                validate_live_checkpoint(destination)
            # Restore only the owned link so cleanup can remove it safely.
            link.unlink(); link.symlink_to(shared.links[0][1])


if __name__ == "__main__":
    unittest.main()
