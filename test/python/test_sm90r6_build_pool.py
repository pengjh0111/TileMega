import fcntl
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[2]


class BuildSlots(unittest.TestCase):
    def test_five_fresh_builds_respect_pool_capacity_and_existing_holder(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);locks=[root/f'lock{i}' for i in range(2)]
            config=root/'pool.json';config.write_text(json.dumps({str(locks[0]):list(map(str,locks))}))
            events=root/'events.jsonl'
            child='''import fcntl,json,os,sys,time
from pathlib import Path
assert os.environ['CUDA_VISIBLE_DEVICES']==''
def event(kind):
 with Path(sys.argv[1]).open('a') as stream:
  fcntl.flock(stream,fcntl.LOCK_EX);stream.write(json.dumps([time.monotonic(),kind,os.getpid()])+'\\n');stream.flush()
event('start');time.sleep(.2);event('end')
'''
            def cohort(offset):
                processes=[]
                for index in range(offset,offset+5):
                    receipt=root/f'{index}.json'
                    command=[sys.executable,str(ROOT/'scripts/run_sm90r1_build.py'),
                        '--receipt',str(receipt),'--max-gib','1','--timeout','10','--lock',str(locks[0]),
                        '--lock-pools',str(config),'--',sys.executable,'-c',child,str(events)]
                    processes.append((subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE),receipt))
                for process,receipt in processes:
                    stdout,stderr=process.communicate(timeout=20)
                    self.assertEqual(process.returncode,0,stderr.decode())
                    self.assertEqual(json.loads(receipt.read_text())['status'],'completed')
                rows=[json.loads(line) for line in events.read_text().splitlines()]
                live=set();peak=0
                for stamp,kind,pid in sorted(rows):
                    if kind=='start':live.add(pid);peak=max(peak,len(live))
                    else:live.remove(pid)
                self.assertFalse(live);self.assertEqual(len(rows),10)
                events.unlink()
                return peak,[json.loads(p.read_text())['lock'] for _,p in processes]
            peak,used=cohort(0)
            self.assertEqual(peak,2);self.assertEqual(set(used),set(map(str,locks)))
            with locks[0].open('a') as foreign:
                fcntl.flock(foreign,fcntl.LOCK_EX)
                peak,used=cohort(5)
                self.assertEqual(peak,1);self.assertEqual(set(used),{str(locks[1])})


if __name__=='__main__':unittest.main()
