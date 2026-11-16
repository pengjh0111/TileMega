"""Synthetic SQLite fixture checks attribution, process scope and overlap."""
import csv
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import moe_profile_nsys


class NsysTest(unittest.TestCase):
    def test_same_named_kernels_keep_their_actual_nvtx_regions(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            database = folder / 'trace.sqlite'
            connection = sqlite3.connect(database)
            connection.executescript('''
                CREATE TABLE StringIds(id INTEGER,value TEXT);
                INSERT INTO StringIds VALUES(1,'same_generic_gemm');
                INSERT INTO StringIds VALUES(2,'cudaLaunchKernel');
                CREATE TABLE NVTX_EVENTS(start INTEGER,end INTEGER,globalTid INTEGER,text TEXT);
                CREATE TABLE CUPTI_ACTIVITY_KIND_RUNTIME(start INTEGER,end INTEGER,globalTid INTEGER,correlationId INTEGER,nameId INTEGER);
                CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL(start INTEGER,end INTEGER,deviceId INTEGER,globalPid INTEGER,
                    correlationId INTEGER,demangledName INTEGER,gridX INTEGER,gridY INTEGER,gridZ INTEGER,
                    blockX INTEGER,blockY INTEGER,blockZ INTEGER,registersPerThread INTEGER,
                    staticSharedMemory INTEGER,dynamicSharedMemory INTEGER);
            ''')
            pid, tid = 7 << 24, (7 << 24) + 11
            modules = []
            for i, stage in enumerate(('qkv_projection', 'output_projection')):
                label = f'moe_profile/{stage}/module{i}'
                modules.append(dict(label=label, stage=stage, source_evidence='CPU fixture only'))
                connection.execute('INSERT INTO NVTX_EVENTS VALUES(?,?,?,?)',
                                   (i*100, i*100+50, tid, label+'|inputs=[{"shape":[1,2048]}]'))
                connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_RUNTIME VALUES(?,?,?,?,?)',
                                   (i*100+10, i*100+20, tid, i+1, 2))
                connection.execute('INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)',
                                   (i*5+60, i*5+70, 0, pid, i+1, 1, 1, 1, 1, 128, 1, 1, 32, 0, 0))
            connection.commit()
            connection.close()
            proof = folder / 'engine.json'
            proof.write_text(json.dumps(dict(nvtx_attribution=[dict(modules=modules)])))
            output = folder / 'export'
            with patch.object(sys, 'argv', ['parser', '--sqlite', str(database), '--nvtx-proof', str(proof), '--out', str(output)]):
                moe_profile_nsys.main()
            with (output / 'kernel_metrics.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual({row['stage'] for row in rows}, {'qkv_projection', 'output_projection'})
            activity = json.loads((output / 'gpu_activity.json').read_text())['devices']['0']
            self.assertEqual(activity['kernel_sum_ns'], 20)
            self.assertEqual(activity['kernel_busy_union_ns'], 15)
            connection = sqlite3.connect(database)
            connection.executescript('''
                CREATE TABLE CUPTI_ACTIVITY_KIND_GRAPH_TRACE(start INTEGER,end INTEGER,deviceId INTEGER);
                INSERT INTO CUPTI_ACTIVITY_KIND_GRAPH_TRACE VALUES(50,100,0);
            ''')
            connection.close()
            with patch.object(sys, 'argv', ['parser', '--sqlite', str(database), '--out', str(output)]):
                moe_profile_nsys.main()
            activity = json.loads((output / 'gpu_activity.json').read_text())['devices']['0']
            self.assertEqual(activity['kernel_busy_union_ns'], 15)
            self.assertEqual(activity['activity_coverage_including_graph_spans_ns'], 50)
            self.assertEqual(activity['graph_span_median_ns'], 50)


if __name__ == '__main__':
    unittest.main()
