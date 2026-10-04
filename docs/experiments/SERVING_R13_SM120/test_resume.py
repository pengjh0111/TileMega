import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import resume


class RecoveryQueueTests(unittest.TestCase):
    def test_section_stamp_schema_matches_the_archived_native_target(self):
        target={'calibration_sections':{'base':{'stamp':'original'},'bf16':{'stamp':'native'}}}
        self.assertTrue(resume.section_stamps_match(target,{'base':'original','bf16':'native'}))
        self.assertFalse(resume.section_stamps_match(target,{'base':'changed','bf16':'native'}))

    def queue(self):
        with tempfile.TemporaryDirectory() as folder:
            location=Path(folder)
            (location/'queue').mkdir()
            (location/'launch_env_r3.json').write_text(json.dumps({'TILEMEGA_BIN':'/fixture/compiler'}))
            with patch.object(resume,'HERE',location),patch.object(resume,'WORK',location):
                resume.register()
            return json.loads((location/'queue/queue_resume_r4.json').read_text())

    def test_old_failures_are_not_reset(self):
        rows=self.queue()
        self.assertTrue(all(row['name'].endswith('_r4') for row in rows))
        self.assertEqual(len({row['name'] for row in rows}),len(rows))

    def test_micro_fault_is_explicit_and_does_not_skip_all_builds(self):
        rows={row['name']:row for row in self.queue()}
        self.assertIn('safe-shapes',rows['E2a_MB-1c_r4']['command'][0])
        self.assertIn('E2a_MB-1c_r4',rows['E3_exports_r4']['after_any'])
        self.assertNotIn('E2a_MB-1c_r3',rows['E3_exports_r4']['after'])

    def test_timing_runs_are_guarded_and_smoke_follows_builds(self):
        rows={row['name']:row for row in self.queue()}
        for label in ('E2a_MB-1c_r4','E3_R13F_llama_r4','E3_R13F_qwen3_r4','E2b_smoke_r4'):
            self.assertTrue(rows[label]['gpu'])
        self.assertIn('E3_fixed_r4',rows['E2b_smoke_r4']['after'])
        self.assertEqual(rows['E3_fixed_r4']['command'][0],'flock')


if __name__=='__main__':
    unittest.main()
