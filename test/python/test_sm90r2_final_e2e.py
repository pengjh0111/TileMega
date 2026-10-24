"""Synthetic CPU contract fixtures; these are not GPU or correctness evidence."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts'))
import run_sm90r2_final_e2e as runner


def fixture():
    coordinates = dict(schema=1, example='synthetic coordinates, not a real kernel')
    manifest = dict(schema=1, kind='sm90r2_final_artifacts', model='qwen3_0_6b', batch=1,
                    protocol=runner.PROTOCOL.copy(), source_head='a' * 40, source_digest='b' * 64,
                    compiler=dict(sha256='c' * 64), target=dict(sha256='d' * 64))
    source = dict(head=manifest['source_head'], worktree_diff_sha256=runner.EMPTY_DIFF,
                  source_digest=manifest['source_digest'], compiler_sha256=manifest['compiler']['sha256'])
    prefill = dict(phase='prefill', seq=64, batch_lo=1, batch_hi=1, capacity=1088,
                   pg='off', requested_residency=0, attention_impl='mma16', attention_kv_block=1088,
                   gemms=[dict(tile_m=64, tile_n=128, tile_k=64, stages=3, split_k=1, impl='wgmma')])
    decode = dict(phase='decode', seq=1, batch_lo=1, batch_hi=1, capacity=1088,
                  native_origin='solver', native_dense=coordinates, export_sha256='e' * 64)
    plans = dict(prefill=prefill, decode=decode, selected_candidate=copy.deepcopy(decode))
    solver = dict(selection='measure', converged=True, budget_exhausted=False, winner=0,
                  e3_validated=True, development_only=False, calibration=dict(fit_status='fitted'),
                  ranked=[dict(rank=0, coordinates=coordinates)], target_sha256='d' * 64,
                  export_sha256='e' * 64, measurements=[dict(rank=0, measured=True, status=0,
                      result=dict(accepted=True, binary_sha256='3' * 64))])
    decode['native_selection'] = copy.deepcopy(solver)
    identities = {role: dict(trace=False, defines={}, source=source.copy(), plan=plan,
                            target_sha256='d' * 64, binary_sha256=str(i) * 64,
                            generated_source_sha256='f' * 64)
                  for i, (role, plan) in enumerate(plans.items(), 1)}
    config = dict(model='qwen3_0_6b', batch=1, execution=dict(decode='L1', loop=1, prefill='L1'),
                  coordinates=dict(prefill=runner.prefill_signature(prefill)))
    rebuild = dict(reference_sha256='3' * 64, candidate_sha256='2' * 64,
                   generated_source_sha256=['f' * 64, 'f' * 64], diagnostic=False)
    for key in ('pass', 'same_coordinates', 'same_workload', 'same_source', 'same_instructions',
                'same_instruction_encoding', 'performance_eligible'):
        rebuild[key] = True
    return [manifest, config, identities, plans, solver, coordinates, rebuild, copy.deepcopy(prefill)]


class FinalE2EContract(unittest.TestCase):
    def test_consistent_relationships_only_validate_the_synthetic_contract(self):
        runner.validate_contract(*fixture())

    def reject(self, mutate, message):
        values = fixture()
        mutate(*values)
        with self.assertRaisesRegex(ValueError, message):
            runner.validate_contract(*values)

    def test_trace_and_boundary_macros_are_ineligible(self):
        self.reject(lambda m, c, ids, *rest: ids['prefill'].update(trace=True), 'trace')
        self.reject(lambda m, c, ids, *rest: ids['decode']['defines'].update(TILEMEGA_TRACE_NATIVE_BOUNDARY='1'), 'trace')

    def test_development_selection_cannot_masquerade_as_final(self):
        for key, value in [('e3_validated', False), ('development_only', True)]:
            self.reject(lambda m, c, ids, plans, solver, *rest: solver.update({key: value}), 'requires validated E-3')
        self.reject(lambda m, c, ids, plans, solver, *rest:
                    solver['calibration'].update(fit_status='initial_overlap_fitted'), 'requires validated E-3')

    def test_cpp_receipt_serialization_does_not_change_the_bound_target(self):
        target = dict(value=1.234567890123, table=[True, 9007199254740992, None])
        self.assertEqual(runner.cpp_json_roundtrip(target), dict(value=1.23456789, table=[True, 9007199254740992, None]))
        self.assertEqual(target['value'], 1.234567890123)

    def test_e3_thresholds_and_all_cells_remain_required(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); path = directory / 'synthetic-evidence.json'; path.write_text('{}')
            ref = dict(path=path.name, sha256=runner.sha(path))
            cells = [dict(cell=name, class_errors_pct={k: 0 for k in
                     ('qkv', 'attention', 'o', 'gate_up', 'down', 'lm_head')},
                     step_error_pct=0, kendall_tau=.8, global_solver_top3_contains_fastest=True,
                     distinct_programs=8, class_evidence=ref, step_evidence=ref, ranking_evidence=ref)
                     for name in sorted(runner.DEVELOPMENT_CELLS)]
            manifest = fixture()[0]
            evidence = dict(schema=1, kind='sm90r2_e3_validation', source_head=manifest['source_head'],
                            target_sha256=manifest['target']['sha256'], fit_status='fitted',
                            independent_validation=True, solver_gate_pass=True, cells=cells)
            runner.validate_e3(evidence, manifest, directory)
            for key, value in [('class_errors_pct', dict(cells[0]['class_errors_pct'], attention=15.01)),
                               ('step_error_pct', -10.01), ('kendall_tau', .699),
                               ('global_solver_top3_contains_fastest', False), ('distinct_programs', 7)]:
                bad = copy.deepcopy(evidence); bad['cells'][0][key] = value
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, 'gate failed'):
                    runner.validate_e3(bad, manifest, directory)
            evidence['cells'] = cells[:-1]
            with self.assertRaisesRegex(ValueError, 'seven'):
                runner.validate_e3(evidence, manifest, directory)

    def test_source_head_dirty_snapshot_and_compiler_must_match(self):
        for key, value in [('head', '0' * 40), ('worktree_diff_sha256', '0' * 64),
                           ('source_digest', '0' * 64), ('compiler_sha256', '0' * 64)]:
            with self.subTest(key=key):
                self.reject(lambda m, c, ids, *rest: ids['prefill']['source'].update({key: value}), 'source/compiler')

    def test_fixed_decode_cannot_replace_solver_output(self):
        self.reject(lambda m, c, ids, plans, *rest: plans['decode'].update(native_origin='fixed'), 'solver output')
        self.reject(lambda m, c, ids, plans, *rest: plans['selected_candidate'].update(native_origin='fixed'), 'solver output')

    def test_prefill_must_keep_reference_coordinates(self):
        self.reject(lambda m, c, ids, plans, *rest: plans['prefill']['gemms'][0].update(tile_n=64), 'prefill coordinates')
        self.reject(lambda m, c, ids, plans, *rest: plans['prefill'].update(seq=1), 'ISL64')

    def test_incomplete_search_and_invalid_winner_are_rejected(self):
        for key, value in [('winner', -1), ('winner', True), ('winner', 1),
                           ('selection', 'predicted'), ('converged', False), ('budget_exhausted', True)]:
            with self.subTest(key=key, value=value):
                self.reject(lambda m, c, ids, plans, solver, *rest: solver.update({key: value}), 'measured winner')

    def test_callback_cannot_measure_another_binary(self):
        def mutate(m, c, ids, plans, solver, *rest):
            solver['measurements'][0]['result']['binary_sha256'] = '0' * 64
            plans['decode']['native_selection'] = copy.deepcopy(solver)
        self.reject(mutate, 'measured-selection')

    def test_rebuild_receipt_is_bound_to_final_binary_and_source(self):
        for key, value in [('pass', False), ('diagnostic', True), ('same_instruction_encoding', False),
                           ('candidate_sha256', '0' * 64), ('generated_source_sha256', ['0' * 64] * 2)]:
            with self.subTest(key=key):
                self.reject(lambda m, c, ids, plans, solver, coord, rebuild, ref: rebuild.update({key: value}), 'rebuild')

    def test_gpus_five_six_and_active_timing_limits(self):
        for gpu in (5, 6):
            with self.assertRaisesRegex(ValueError, 'functional-only'):
                runner.validate_gpu(gpu, {})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'active.json'
            path.write_text(json.dumps(dict(allowed_timing_gpu_indices=[7])))
            with self.assertRaisesRegex(ValueError, 'active timing'):
                runner.validate_gpu(1, dict(gpu_allowlist_file=str(path)))
            self.assertEqual(runner.validate_gpu(7, dict(gpu_allowlist_file=str(path))), 1)

    def test_bound_files_reject_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'prompt.json'; path.write_text('[[1]]')
            record = dict(path=path.name, sha256=runner.sha(path))
            self.assertEqual(runner.checked_file(record, path.parent), path)
            path.write_text('[[2]]')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                runner.checked_file(record, path.parent)

    def test_explicit_prefill_and_measurement_only_command(self):
        bound = dict(manifest=dict(batch=4), files=dict(model_config=Path('/model/config.json'),
                     prompt=Path('/frozen/prompt.json')), artifacts={role: dict(binary=Path('/final') / (role + '.so'))
                     for role in ('prefill', 'decode')})
        command = runner.command_for(bound, Path('/attempt/job'), 2)
        self.assertEqual(command[command.index('--prefill') + 1], '/final/prefill.so')
        self.assertEqual(command[command.index('--round-index') + 1], '2')
        self.assertEqual(command[command.index('--repeats') + 1], '3')
        self.assertFalse({'--hf-check', '--check-step-loop', '--check-executors'} & set(command))
        self.assertEqual(runner.AMENDMENT['c1'], 'not_performed')
        self.assertEqual(runner.AMENDMENT['c2'], 'not_performed')

    def test_queue_builder_writes_inactive_single_system_jobs(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); policy = directory / 'policy.json'
            policy.write_text(json.dumps(dict(idle_power_mapping_provenance=dict(uuid_by_gpu={'1': 'GPU-test'}))))
            bound = dict(files=dict(guard_policy=policy), manifest=dict(model='llama3_1_8b', batch=16),
                         path=directory / 'artifacts.json', manifest_sha256='a' * 64, runner_sha256='b' * 64)
            out = directory / 'prepared'; runner.prepare_queue(bound, 1, out)
            jobs = runner.read(out / 'queue/queue_final_e2e.json')
            self.assertEqual(len(jobs), 5)
            self.assertTrue(all(j['gpu'] and j['timing'] for j in jobs))
            self.assertTrue(all('numactl' == j['command'][0] and '--expected-uuid' in j['command'] for j in jobs))
            self.assertEqual(runner.read(out / 'preparation.json')['gpu_jobs_started'], 0)
            self.assertFalse(runner.read(out / 'preparation.json')['active_queue_publication'])

    def test_raw_timing_and_token_completeness_are_checked_without_correctness_claims(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); output = directory / 'measurements/B1'; output.mkdir(parents=True)
            manifest = dict(batch=1, prompt=dict(sha256='a'), model_config=dict(sha256='b'),
                            prefill=dict(binary=dict(sha256='c')), decode=dict(binary=dict(sha256='d')))
            rows = []
            for count in (1, 1024):
                for run in range(4):
                    name = f'tokens_N{count}_run{run}.json'
                    runner.write(output / name, [[0] * count])
                    rows.append(dict(N=count, run=run, warmup=run == 0,
                                     wall_seconds=.1 if count == 1 else 1.1, tokens_file=name))
            value = dict(system='TileMega', batch=1, prompt_length=64, max_tokens=1024, round_index=0,
                         prompt_ids_sha256='a', model_config_sha256='b', pid=123,
                         configuration=dict(prefill_sha256='c', decode_sha256='d',
                                            mode='L1', decode_loop=1, prefill_mode='L1'),
                         generation_runs=rows, **runner.summarize(rows, 1, 1024))
            path = output / 'measurements.json'; runner.write(path, value)
            result = runner.metrics_from_job(dict(manifest=manifest), directory, 0)
            self.assertAlmostEqual(result['metrics']['tpot_seconds'], 1 / 1023)
            self.assertEqual(len(result['tokens']), 8)
            self.assertNotIn('c1_pass', result)
            value['e2e_seconds'] = 2.2; runner.write(path, value)
            with self.assertRaisesRegex(ValueError, 'raw runs'):
                runner.metrics_from_job(dict(manifest=manifest), directory, 0)
            value['e2e_seconds'] = 1.1; runner.write(path, value)
            runner.write(output / 'tokens_N1024_run3.json', [[0] * 1023])
            with self.assertRaisesRegex(ValueError, 'token table'):
                runner.metrics_from_job(dict(manifest=manifest), directory, 0)

    def test_outer_guard_failure_prevents_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory); prep = directory / 'prep'; attempt = directory / 'attempt'
            (prep / 'queue').mkdir(parents=True); (attempt / 'job').mkdir(parents=True)
            runner.write(prep / 'queue/queue_final_e2e.json', [])
            names = ['f0', 'f1', 'f2']
            bound = dict(manifest=dict(model='qwen3_0_6b'), manifest_sha256='a', runner_sha256='b')
            runner.write(prep / 'preparation.json', dict(manifest_sha256='a', runner_sha256='b', gpu=1,
                         jobs=names, queue_sha256=runner.sha(prep / 'queue/queue_final_e2e.json')))
            runner.write(directory / 'state.json', {n: dict(status='done', exit_code=0, out=str(attempt)) for n in names})
            runner.write(attempt / 'guard_result.json', dict(code=75))
            runner.write(attempt / 'job/final-e2e-process.json', {})
            runner.write(attempt / 'job/physical-device-identity.json', {})
            runner.write(attempt / 'job/physical-device-command.json', dict(actual_returncode=0))
            with self.assertRaisesRegex(ValueError, 'guard/physical'):
                runner.collect(bound, prep, directory / 'state.json', directory / 'result.json')
            self.assertFalse((directory / 'result.json').exists())


if __name__ == '__main__':
    unittest.main()
