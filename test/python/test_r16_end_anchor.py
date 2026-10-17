# SPDX-License-Identifier: BSD-3-Clause
"""The end anchor must wait for valid final selections, not queued work."""
from pathlib import Path
import copy,sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from advance_r16_end_anchor import acceptance_ready

class EndAnchorTest(unittest.TestCase):
    def reports(self):
        def matrix(count,key):
            return dict(complete=True,completed_cells=count,expected_cells=count,
                **{key:[dict(model=str(i),system='s',batch=1,protocol_complete=True,fresh_processes=3)for i in range(count)]})
        return [matrix(30,'selected'),matrix(120,'cells'),dict(complete_cells=9,cells=[dict(system=str(i),batch=2,complete=True,fresh_processes=3)for i in range(9)]),
            dict(complete=True,completed_models=5,expected_models=5,cells=[dict(model=str(i),status='complete',selected=dict(id='valid winner'))for i in range(5)])]
    def test_incomplete_matrix_or_fresh_process_blocks_release(self):
        values=self.reports();self.assertTrue(acceptance_ready(*values))
        values[0]['selected'].pop();self.assertFalse(acceptance_ready(*values))
        values=self.reports();values[2]['cells'][0]['fresh_processes']=2;self.assertFalse(acceptance_ready(*values))
    def test_pending_dnn_is_not_a_final_selection(self):
        values=self.reports();values[3]['cells'][0]['status']='pending_candidates'
        self.assertFalse(acceptance_ready(*values))
    def test_historical_rejection_does_not_discard_a_valid_winner(self):
        values=self.reports();values[2]['terminal_failures']=[dict(old_attempt='failed')]
        values[3]['cells'][0]['rejected']=[dict(candidate='failed new combination')]
        self.assertTrue(acceptance_ready(*values))
    def test_duplicate_cells_cannot_fill_the_matrix(self):
        values=self.reports();values[1]['cells'][-1]=copy.deepcopy(values[1]['cells'][0])
        self.assertFalse(acceptance_ready(*values))

if __name__=='__main__':unittest.main()
