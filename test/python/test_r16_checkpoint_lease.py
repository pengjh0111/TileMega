"""Read-only FD leases preserve bytes after five fresh source owners exit."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
from lease_r16_checkpoint import CheckpointLease, start_ticks, terminal_plan


def source_fixture(path, fd, pid, payload):
    path.mkdir()
    (path/'config.json').write_bytes(b'{}')
    name = 'model.safetensors'
    (path/name).symlink_to(f'/proc/{pid}/fd/{fd}')
    manifest = dict(algorithm='test', config_sha256=hashlib.sha256(b'{}').hexdigest(),
        seed=0, aliases={}, tensors=[dict(name='weight', bytes=len(payload))],
        state='live', supervisor_pid=pid, total_bytes=len(payload), tensor_count=1,
        shards=[dict(file=name, bytes=len(payload), data_bytes=len(payload),
                     sha256=hashlib.sha256(payload).hexdigest())])
    identity = {key:manifest[key] for key in ('algorithm','config_sha256','seed','aliases','tensors')}
    fingerprint = hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    manifest['checkpoint_sha256'] = fingerprint
    (path/'random_checkpoint.json').write_text(json.dumps(manifest))
    (path/'model.safetensors.index.json').write_text(json.dumps(dict(weight_map={'weight':name})))
    return fingerprint


class LeaseTest(unittest.TestCase):
    def test_five_fresh_owners_exit_without_losing_leased_bytes(self):
        for index in range(5):
            with self.subTest(index=index), tempfile.TemporaryDirectory() as tmp:
                source, destination = Path(tmp)/'source', Path(tmp)/'lease'
                payload = bytes([index+1])*4096
                ready_r,ready_w = os.pipe()
                stop_r,stop_w = os.pipe()
                pid = os.fork()
                if pid == 0:
                    os.close(ready_r);os.close(stop_w)
                    fd = os.memfd_create('r16-lease-test',flags=os.MFD_CLOEXEC)
                    os.write(fd,payload)
                    fingerprint = source_fixture(source,fd,os.getpid(),payload)
                    os.write(ready_w,fingerprint.encode());os.close(ready_w)
                    os.read(stop_r,1);os.close(fd);os._exit(0)
                os.close(ready_w);os.close(stop_r)
                fingerprint = os.read(ready_r,128).decode()
                original = (source/'random_checkpoint.json').read_bytes()
                with CheckpointLease(source,destination,fingerprint,pid,start_ticks(pid)) as lease:
                    os.write(stop_w,b'x')
                    _,status = os.waitpid(pid,0)
                    self.assertEqual(os.waitstatus_to_exitcode(status),0)
                    self.assertFalse((source/'model.safetensors').exists())
                    self.assertEqual((destination/'model.safetensors').read_bytes(),payload)
                    self.assertEqual((destination/'random_checkpoint.json').read_bytes(),original)
                    self.assertTrue(lease.proof['shards'][0]['same_inode_verified'])
                    self.assertEqual(lease.proof['additional_weight_bytes_allocated'],0)
                    with self.assertRaises(OSError):
                        os.write(lease.fds[0],b'changed')
                self.assertFalse((destination/'model.safetensors').exists())
                os.close(ready_r);os.close(stop_w)

    def test_missing_or_unsealed_jobs_never_release(self):
        plan = dict(jobs=['native'],sealed=False,release_when_terminal=True)
        self.assertFalse(terminal_plan(plan,{'native':dict(status='done')}))
        plan['sealed'] = True
        self.assertFalse(terminal_plan(plan,{}))
        self.assertFalse(terminal_plan(plan,{'native':dict(status='running')}))
        self.assertTrue(terminal_plan(plan,{'native':dict(status='done')}))
        self.assertTrue(terminal_plan(plan,{'native':dict(status='failed')}))

    def test_empty_job_list_never_releases(self):
        self.assertFalse(terminal_plan(dict(jobs=[],sealed=True,release_when_terminal=True),{}))

    def test_wrong_fingerprint_or_reused_owner_pid_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'source'
            fd = os.memfd_create('r16-lease-identity',flags=os.MFD_CLOEXEC)
            try:
                payload = b'test'
                os.write(fd,payload)
                fingerprint = source_fixture(source,fd,os.getpid(),payload)
                for digest,ticks in (('0'*64,start_ticks(os.getpid())),(fingerprint,'0')):
                    with self.subTest(digest=digest,ticks=ticks), self.assertRaises(ValueError):
                        with CheckpointLease(source,Path(tmp)/'lease',digest,os.getpid(),ticks):
                            pass
            finally:
                os.close(fd)


if __name__ == '__main__':
    unittest.main()
