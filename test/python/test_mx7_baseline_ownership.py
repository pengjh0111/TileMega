"""Torchrun siblings can belong to distinct sessions inside one guarded job."""
from pathlib import Path
import tempfile
import unittest

from tilemega.serving.baseline_common import job_root


class OwnershipTests(unittest.TestCase):
    def test_distinct_rank_sessions_resolve_to_the_same_ancestor(self):
        with tempfile.TemporaryDirectory() as directory:
            proc = Path(directory)
            for pid, parent, command in ((10, 1, 'python\0-m\0tilemega.bench.gpu_guard\0'),
                                          (20, 10, 'python\0-m\0torch.distributed.run\0'),
                                          (30, 20, 'python\0rank0.py\0'),
                                          (40, 20, 'python\0rank1.py\0'),
                                          (99, 1, 'python\0someone_else.py\0')):
                folder = proc / str(pid)
                folder.mkdir()
                (folder / 'cmdline').write_bytes(command.encode())
                (folder / 'stat').write_text(f'{pid} (python worker) S {parent} {pid} {pid}\n')
            self.assertEqual(job_root(30, proc), 10)
            self.assertEqual(job_root(40, proc), 10)
            self.assertEqual(job_root(99, proc), 99)


if __name__ == '__main__':
    unittest.main()
