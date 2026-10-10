"""Prefill paging and executor selection without launching a GPU measurement."""
import json
import tempfile
from pathlib import Path
from unittest import TestCase, main

from tilemega.cli import Run, read_config
from tilemega.serving.execution import compiler_features, phase_pg_choices, prefill_combinations


class PrefillChoiceTest(TestCase):
    def test_legacy_defaults_and_explicit_prefill_candidates(self):
        for pg in ('off', 'l2', 'auto', 'pages', 'measure'):
            features=dict(pg=pg, prefill_pg='l2')
            expected=(pg,) if pg in ('off', 'l2') else ('l2',)
            self.assertEqual(phase_pg_choices(features,'prefill'),expected)
            features['prefill_pg']='measure'
            self.assertEqual(phase_pg_choices(features,'prefill'),('l2','pages'))
        self.assertEqual(prefill_combinations('pages','measure'),[('L1',0),('L2',0)])
        self.assertEqual(prefill_combinations('l2','L1'),[('L1',0)])
        self.assertNotIn('prefill_pg',compiler_features(dict(pg='pages',prefill_pg='measure')))

    def test_config_accepts_prefill_measure_and_rejects_unknown_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'run.json'
            value=dict(model=dict(path=directory),features=dict(prefill_pg='measure',prefill_executor='measure'))
            path.write_text(json.dumps(value))
            self.assertEqual(read_config(path)['features']['prefill_executor'],'measure')
            value['features']['prefill_pg']='auto';path.write_text(json.dumps(value))
            with self.assertRaisesRegex(ValueError,'prefill_pg'):read_config(path)

    def test_cached_winner_is_bound_to_library_contents(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);libraries={pg:root/(pg+'.so') for pg in ('l2','pages')}
            for path in libraries.values():path.write_bytes(path.name.encode())
            run=Run.__new__(Run);run.out=root;run.model=root
            run.config=dict(features=dict(prefill_executor='measure'),solver=dict(candidate_guard_wait_s=0))
            calls=[]
            def command(argv,label,**kwargs):
                calls.append(label)
                options=dict(zip(argv[3::2],argv[4::2]))
                self.assertTrue(kwargs['gpu'])
                self.assertEqual(options['--loop'],'0')
                self.assertEqual(options['--past-mid'],'0')
                out=Path(options['--out']);out.mkdir(exist_ok=True)
                mode=options['--mode'];page=options['--so'].endswith('pages.so')
                score=1 if page and mode=='L1' else 2
                (out/'measurements.json').write_text(json.dumps(dict(modes={mode:dict(mean_ms=score,decode_loop_used=False)})))
            run.command=command
            self.assertEqual(run.select_prefill(libraries,1),('pages','L1'))
            self.assertEqual(len(calls),12)
            calls.clear()
            self.assertEqual(run.select_prefill(libraries,1),('pages','L1'))
            self.assertEqual(calls,[])
            libraries['pages'].write_bytes(b'changed artifact')
            self.assertEqual(run.select_prefill(libraries,1),('pages','L1'))
            self.assertEqual(len(calls),12)

    def test_fixed_executor_and_library_need_no_selection_measurement(self):
        run=Run.__new__(Run)
        run.config=dict(features=dict(prefill_executor='L2'),solver={})
        self.assertEqual(run.select_prefill({'pages':Path('/plan.so')},1),('pages','L2'))


if __name__=='__main__':main()
