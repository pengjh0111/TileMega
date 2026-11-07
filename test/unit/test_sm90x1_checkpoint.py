"""CPU checkpoint reuse must preserve bytes and independent child processes."""
import hashlib
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from unittest.mock import Mock
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'scripts'))
import sm90x1_checkpoint as cache


class CheckpointTest(unittest.TestCase):
    def setUp(self):
        parent = ROOT/'runs/sm90x1/test-checkpoint-cache';parent.mkdir(parents=True, exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(dir=parent)
        self.root = Path(self.temporary.name);self.source = self.root/'source';self.source.mkdir()
        (self.source/'config.json').write_text('{"model_type":"fixture"}\n')
        (self.source/'model.safetensors.index.json').write_text('{}')
        self.data = b'deterministic CPU weights\n' * 100
        self.original = os.memfd_create('checkpoint-fixture');os.write(self.original, self.data)
        self.name = 'model.safetensors'
        (self.source/self.name).symlink_to(f'/proc/{os.getpid()}/fd/{self.original}')
        self.manifest = dict(algorithm='fixture', config_sha256=cache.sha(self.source/'config.json'),
            seed=0, aliases={}, tensors=[], state='live', supervisor_pid=os.getpid(),
            total_bytes=len(self.data), shards=[dict(file=self.name,bytes=len(self.data),
                sha256=hashlib.sha256(self.data).hexdigest())])
        self.manifest['checkpoint_sha256'] = cache.identity(self.manifest)
        cache.write(self.source/'random_checkpoint.json', self.manifest)
        self.reference = self.root/'accepted.json';cache.write(self.reference, self.manifest)
        self.destination = self.root/'cache';self.descriptors = []

    def tearDown(self):
        if self.original is not None:os.close(self.original)
        for fd in self.descriptors:os.close(fd)
        self.temporary.cleanup()

    def borrow(self):
        self.descriptors = cache.borrow(self.source, self.reference, self.destination)

    def test_five_fresh_readers_survive_original_supervisor_close(self):
        self.borrow();os.close(self.original);self.original = None
        program = """
import hashlib,json,os,sys,time
from pathlib import Path
p=Path(os.environ['TILEMEGA_RANDOM_CHECKPOINT'])
assert hashlib.sha256((p/'model.safetensors').read_bytes()).hexdigest()==sys.argv[1]
Path(sys.argv[2]).write_text(str(os.getpid()))
while not Path(sys.argv[3]).exists():time.sleep(.01)
"""
        children = []
        for index in range(5):
            command = [sys.executable, str(ROOT/'scripts/sm90x1_checkpoint.py'), 'run',
                '--cache', str(self.destination), '--config', str(self.source),
                '--expected', self.manifest['checkpoint_sha256'], '--out', str(self.root/f'reader{index}'),
                '--max-gib', '1', '--', sys.executable, '-c', program,
                hashlib.sha256(self.data).hexdigest(), str(self.root/f'pid{index}'), str(self.root/'release')]
            children.append(subprocess.Popen(command, env=dict(os.environ,TILEMEGA_GPU_LOCK_HELD='1'),
                                             stdout=subprocess.PIPE,stderr=subprocess.PIPE))
        try:
            deadline = time.monotonic()+10
            while not all((self.root/f'pid{i}').exists() for i in range(5)) and time.monotonic()<deadline:
                time.sleep(.01)
            self.assertTrue(all((self.root/f'pid{i}').exists() for i in range(5)))
            with (self.destination/'readers.lock').open('a') as lock:
                with self.assertRaises(BlockingIOError):fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        finally:
            (self.root/'release').touch()
            outputs = [p.communicate(timeout=20) for p in children]
        self.assertEqual([p.returncode for p in children], [0]*5, outputs)
        self.assertEqual(len({(self.root/f'pid{i}').read_text() for i in range(5)}), 5)
        for index in range(5):
            manifest = cache.read(self.root/f'reader{index}/random_checkpoint.json')
            self.assertEqual(manifest['checkpoint_sha256'], self.manifest['checkpoint_sha256'])
            self.assertEqual(manifest['state'], 'closed')
        with (self.destination/'readers.lock').open('a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)

    def test_changed_source_bytes_are_rejected(self):
        os.pwrite(self.original, b'X', 0)
        with self.assertRaisesRegex(ValueError, 'content mismatch'):
            self.borrow()
        self.assertFalse(self.destination.exists())

    def test_model_mismatch_and_reused_pid_are_rejected(self):
        self.borrow();other = self.root/'other';other.mkdir();(other/'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'model mismatch'):
            cache.validate(self.destination, other, self.manifest['checkpoint_sha256'], 1)
        manifest = cache.read(self.destination/'random_checkpoint.json');manifest['supervisor_start'] = 'wrong'
        cache.write(self.destination/'random_checkpoint.json', manifest)
        with self.assertRaisesRegex(ValueError, 'supervisor is unavailable'):
            cache.validate(self.destination, self.source, self.manifest['checkpoint_sha256'], 1)

    def test_policy_is_scoped_to_x1_and_matching_model(self):
        policy = self.root/'policy.json'
        cache.write(policy,dict(checkpoint_cache=dict(path=str(self.destination),
                    config_sha256=self.manifest['config_sha256'],checkpoint_sha256=self.manifest['checkpoint_sha256'])))
        with patch.object(cache, 'POLICY', policy):
            self.assertIsNotNone(cache.cached_command(self.source, self.root/'output', 1, ['child']))
            self.assertIsNone(cache.cached_command(self.source, '/tmp/non-x1-output', 1, ['child']))
            (self.source/'config.json').write_text('{}')
            self.assertIsNone(cache.cached_command(self.source, self.root/'output', 1, ['child']))

    def test_missing_guard_does_not_start_child(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, 'scheduler guard'):
                cache.run(None)

    def policy_for_moe(self):
        policy = self.root/'moe-policy.json'
        cache.write(policy,dict(checkpoint_cache=dict(path=str(self.destination),
            config_sha256=cache.sha(ROOT/'configs/serving/sm90d1/inputs/qwen3_30b_a3b/config.json'),
            checkpoint_sha256=self.manifest['checkpoint_sha256'])))
        return policy

    def test_trace_queue_retains_identity_wrapper_and_frozen_artifacts(self):
        import sm90x1 as base
        unit = self.root/'trace';handle = Mock()
        cell = dict(cell='fixture',model='qwen3_30b_a3b',batch=16,prefill='/frozen-prefill.so',**{'class':'trial'})
        with patch.object(cache,'POLICY',self.policy_for_moe()), \
             patch.object(base,'claim',return_value=(0,handle)), \
             patch.object(base.subprocess,'Popen'), \
             patch.object(base,'watch',side_effect=RuntimeError('queue-only fixture')):
            with self.assertRaisesRegex(RuntimeError,'queue-only fixture'):
                base.run_trace_unit(SimpleNamespace(phase=2),cell,'W',False,self.root,
                                    dict(head='frozen'),'/frozen-trace.so',unit)
        queue = next(unit.glob('try*/queue/queue_trace.json'))
        jobs = cache.read(queue);self.assertEqual(len(jobs),2)
        for job in jobs:
            command = job['command']
            self.assertIn(str(ROOT/'scripts/run_r16_with_device_identity.py'),command)
            self.assertIn(str(ROOT/'scripts/sm90x1_checkpoint.py'),command)
            self.assertNotIn('tilemega.serving.random_checkpoint',command)
            self.assertEqual(command[command.index('--decode')+1],'/frozen-trace.so')
            self.assertEqual(command[command.index('--prefill')+1],'/frozen-prefill.so')
        handle.close.assert_called_once()

    def test_correctness_runner_keeps_fresh_child_and_cache_provenance(self):
        import sm90r3_llm as runner
        binary = self.root/'decode.so';binary.write_bytes(b'fixture')
        arm = dict(decode=str(binary),prefill='prompt',decode_sha256=cache.sha(binary),prefill_sha256='prompt')
        spec = self.root/'spec.json'
        cache.write(spec,dict(arms={'W':arm},model='qwen3_30b_a3b',name='fixture',kind='corr',batch=1))
        out = self.root/'process'
        with patch.object(cache,'POLICY',self.policy_for_moe()), \
             patch.dict(os.environ,{'TILEMEGA_GPU_LOCK_HELD':'1'}), \
             patch.object(runner.subprocess,'call',return_value=0) as child:
            self.assertEqual(runner.measure(SimpleNamespace(spec=spec,arm='W',fresh=3,out=out)),0)
        command = child.call_args.args[0]
        self.assertIn(str(ROOT/'scripts/sm90x1_checkpoint.py'),command)
        self.assertEqual(command[command.index('--fresh')+1],'3')
        self.assertEqual(cache.read(out/'sm90r3-process.json')['fresh'],3)


if __name__ == '__main__':unittest.main()
