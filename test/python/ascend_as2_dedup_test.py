"""CPU-only immutable sample storage safety; no device is initialized."""
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from as2_dedup import file_stamp,link_identical


class DedupSafety(unittest.TestCase):
    def test_preserves_bytes_and_rejects_mismatch_and_escape(self):
        with tempfile.TemporaryDirectory(dir='/home/d50064638/junhuip/tilemega-as2/tmp') as directory:
            root=Path(directory);inside=root/'owned';inside.mkdir()
            source=inside/'a';target=inside/'b';outside=root/'outside'
            source.write_bytes(b'immutable sample');target.write_bytes(source.read_bytes());outside.write_bytes(source.read_bytes())
            sha=hashlib.sha256(source.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):link_identical(source,outside,[inside],file_stamp(source),file_stamp(outside),sha)
            target.write_bytes(b'wrong payload')
            with self.assertRaises(ValueError):link_identical(source,target,[inside],file_stamp(source),file_stamp(target),sha)
            self.assertEqual(target.read_bytes(),b'wrong payload')
            target.write_bytes(source.read_bytes());old=file_stamp(target)
            target.write_bytes(b'changed again')
            with self.assertRaises(ValueError):link_identical(source,target,[inside],file_stamp(source),old,sha)
            target.write_bytes(source.read_bytes())
            link_identical(source,target,[inside],file_stamp(source),file_stamp(target),sha)
            self.assertEqual(source.read_bytes(),b'immutable sample')
            self.assertEqual(target.read_bytes(),source.read_bytes())
            self.assertEqual(source.stat().st_ino,target.stat().st_ino)


if __name__=='__main__':unittest.main()
