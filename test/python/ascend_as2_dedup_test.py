"""CPU-only immutable sample storage safety; no device is initialized."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from as2_dedup import file_stamp,link_identical,completed_request_directories


class DedupSafety(unittest.TestCase):
    def test_storage_only_requires_completed_inactive_matching_request_producers(self):
        with tempfile.TemporaryDirectory(dir='/home/d50064638/junhuip/tilemega-as2/tmp') as directory:
            root=Path(directory);binding=root/'binding';binding.mkdir();out=root/'outputs';out.mkdir()
            for name in ('request_-1','request_0'):(out/name).mkdir()
            job=root/'job';job.mkdir();p=job/'job.json'
            receipt=dict(project='as2',status='passed',exit_code=0,device_locked=False,
                pid=999999999,start_ticks=0,executable_sha256='CPU-fixture',
                command=['CPU-fixture','request',str(binding),str(out),'67','1','1','0'])
            p.write_text(json.dumps(receipt))
            self.assertEqual(len(completed_request_directories([job],[root])),2)
            receipt['status']='failed';p.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):completed_request_directories([job],[root])
            receipt.update(status='passed',pid=os.getpid());receipt.pop('start_ticks');p.write_text(json.dumps(receipt))
            with self.assertRaises((ValueError,KeyError)):completed_request_directories([job],[root])
            fields=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()
            receipt['start_ticks']=int(fields[19]);p.write_text(json.dumps(receipt))
            with self.assertRaisesRegex(ValueError,'still alive'):completed_request_directories([job],[root])

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
