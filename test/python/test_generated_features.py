"""Check actual source generation; this suite never invokes nvcc or a GPU."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import unittest

from tilemega.build.feature_contract import (
    INTEGER_FEATURES, build_defines, implementations, validate,
)


ROOT = Path(__file__).resolve().parents[2]
COMPILER = os.environ.get('TILEMEGA_BIN')


def one_layer_export(source):
    """Retain one complete decoder layer and reconnect the final norm."""
    source = copy.deepcopy(source)
    exported_inputs = source['signature']['inputs']

    def layer(node):
        paths = [entry['path'] for entry in node.get('nn_module_stack', [])]
        layers = [int(match[1]) for path in paths
                  if (match := re.search(r'(?:^|\.)layers\.(\d+)(?:\.|$)', path))]
        return layers[0] if layers else None

    layers = [node for node in source['nodes'] if layer(node) is not None]
    first_last = [node for node in layers if layer(node) == 0][-1]['name']
    last_last = layers[-1]['name']
    excluded = {item['name'] for item in source['signature']['inputs']
                if re.search(r'(?:layers[._]|past_[kv])(\d+)', item.get('target') or item['name'])
                and int(re.search(r'(?:layers[._]|past_[kv])(\d+)',
                                  item.get('target') or item['name'])[1]) > 0}
    kept = [node for node in source['nodes']
            if node['name'] not in excluded and layer(node) in (None, 0)]

    def reconnect(value):
        if isinstance(value, str):
            # Export may source a shared symbolic past extent from any layer.
            if value in excluded and re.fullmatch(r'past_[kv]\d+', value):
                return re.sub(r'\d+$', '0', value)
            return first_last if value == last_last else value
        if isinstance(value, list):
            return [reconnect(item) for item in value]
        if isinstance(value, dict):
            return {key: reconnect(item) for key, item in value.items()}
        return value

    source['nodes'] = reconnect(kept)
    outputs = source['signature']['outputs'][:3]
    source['signature']['outputs'] = outputs
    for node in source['nodes']:
        if node['op'] == 'output':
            node['inputs'] = node['inputs'][:3]
    source.pop('graph_signature', None)
    nodes = {node['name']: node for node in source['nodes']}
    reachable = set()

    def visit(name):
        if name not in nodes:
            raise ValueError('one-layer fixture has a missing dependency: ' + name)
        if name in reachable:
            return
        reachable.add(name)
        for operand in nodes[name]['inputs']:
            visit(operand)

    visit(next(node['name'] for node in source['nodes'] if node['op'] == 'output'))
    source['nodes'] = [node for node in source['nodes'] if node['name'] in reachable]
    source['signature']['inputs'] = [item for item in source['signature']['inputs']
                                     if item['name'] in reachable]
    old_users = [item['name'] for item in exported_inputs
                 if item['kind'] == 'USER_INPUT']
    new_users = [item['name'] for item in source['signature']['inputs']
                 if item['kind'] == 'USER_INPUT']
    # Shape guards index user inputs, not graph placeholders. Project the
    # removed layers' identical KV extents onto layer zero before reindexing.
    def guard_input(match):
        name = reconnect(old_users[int(match[1])])
        return "L['flat_args'][" + str(new_users.index(name)) + ']'
    source['guards'] = [re.sub(r"L\['flat_args'\]\[(\d+)\]", guard_input, guard)
                        for guard in source.get('guards', [])]
    return source


class FeatureFixtureTests(unittest.TestCase):
    def test_single_layer_preserves_valid_shape_guards(self):
        exported = json.loads((ROOT / 'test/fixtures/serving_rejection_export.json').read_text())
        fixture = one_layer_export(exported)
        users = [item for item in fixture['signature']['inputs'] if item['kind'] == 'USER_INPUT']
        self.assertEqual([item['name'] for item in users],
                         ['input_ids', 'past_k0', 'past_v0', 'rope_cos', 'rope_sin'])
        self.assertEqual(len(fixture['guards']), len(exported['guards']))
        nodes = {item['name']: item for item in fixture['nodes']}
        for guard in fixture['guards']:
            for index, dimension in re.findall(r"flat_args'\]\[(\d+)\]\.size\(\)\[(\d+)\]", guard):
                self.assertLess(int(index), len(users))
                self.assertLess(int(dimension), len(nodes[users[int(index)]['name']]['shape']))


@unittest.skipUnless(COMPILER, 'set TILEMEGA_BIN to the compiler under test')
class GeneratedFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base = Path(os.environ.get('TILEMEGA_FEATURE_RESULTS',
                                   ROOT / 'runs/r15/generated_feature_contract'))
        base.mkdir(parents=True, exist_ok=True)
        cls.out = Path(tempfile.mkdtemp(prefix='check_', dir=base))
        exported = json.loads((ROOT / 'test/fixtures/serving_rejection_export.json').read_text())
        cls.llm = cls.out / 'decoder.json'
        cls.llm.write_text(json.dumps(one_layer_export(exported)) + '\n')
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        (cls.out / 'checks.json').write_text(json.dumps(cls.results, indent=2) + '\n')

    def generate(self, family, label, flags, rejection=None):
        folder = self.out / family / label
        folder.mkdir(parents=True)
        cuda = folder / 'plan.cu'
        fixture = self.llm if family == 'llm' else ROOT / 'test/fixtures/dnn/dwpw_pair.json'
        command = [COMPILER, 'compile', str(fixture), str(cuda), '--emit', 'serving',
            '--serving', 'decode' if family == 'llm' else 'forward',
            '--frontend', 'decoder' if family == 'llm' else 'dnn', '--pg', 'l2',
            '--pdl', 'off', '--runtime-target', str(ROOT / 'test/fixtures/serving_rejection_target.json'),
            '--capacity', '192', '--past-range', '64:128', '--batch', '1',
            '--serve-kv-block', '64', '--selection', 'predicted', *flags]
        (folder / 'command.json').write_text(json.dumps(command) + '\n')
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, timeout=180)
        (folder / 'compiler.log').write_text(result.stdout)
        receipt = dict(family=family, label=label, returncode=result.returncode,
                       rejection=rejection)
        self.results.append(receipt)
        if rejection is not None:
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(rejection, result.stdout)
            self.assertFalse(cuda.exists())
            receipt['pass'] = True
            return None
        self.assertEqual(result.returncode, 0, f'{family}/{label}: {folder / "compiler.log"}')
        manifest = json.loads(Path(str(cuda) + '.plan.json').read_text())
        source = cuda.read_text()
        options = shlex.split(Path(str(cuda) + '.runtime_flags.txt').read_text())
        defines = build_defines(source, options)
        validate(manifest, defines)
        self.assertEqual(defines.get('TILEMEGA_DM_SUPPORT', '0'), '1' if family == 'dm' else '0')
        fields = implementations(manifest, defines)
        receipt.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       defines=defines, implementations=fields)
        return manifest, defines, fields, receipt

    # A dense decoder's decode plan on xcore1000 is the MX-4 static program
    # (MX4 prompt §3.9), which takes none of the interpreted executor's
    # options: those are checked on the DNN frontend, which still runs it.
    def test_integer_features_reach_the_interpreted_executor(self):
        for family in ('dm',):
            for feature, (macro, default) in INTEGER_FEATURES.items():
                values = (1, 2) if feature == 'attention_buffers' else (
                    (0, 200) if feature == 'v3_poll_ns' else (0, 31) if feature == 'kphase_mask' else (0, 1))
                for value in values:
                    with self.subTest(family=family, feature=feature, value=value):
                        result = self.generate(family, feature + str(value),
                            ['--' + feature.replace('_', '-'), str(value)])
                        manifest, defines, fields, receipt = result
                        self.assertEqual(int(defines.get(macro, str(default))), value)
                        self.assertEqual(int(manifest[feature]), value)
                        if feature == 'attention_buffers':
                            self.assertEqual(fields['attention']['buffers'], value)
                        if feature == 'attention_noinline':
                            self.assertEqual(fields['attention']['noinline'], bool(value))
                        if feature in ('mma_reg_pipe', 'ep_direct'):
                            self.assertTrue(fields['gemms'])
                            self.assertTrue(all(g[feature] == value for g in fields['gemms']))
                        receipt['pass'] = True

    def test_attention_and_handoff_switches(self):
        for family in ('dm',):
            for attention in ('mma16', 'simt'):
                with self.subTest(family=family, attention=attention):
                    manifest, defines, fields, receipt = self.generate(
                        family, attention, ['--attention-impl', attention])
                    self.assertEqual(manifest['attention_impl'], attention)
                    self.assertEqual(fields['attention']['impl'], attention)
                    receipt['pass'] = True
            for value in (0, 1):
                with self.subTest(family=family, nonpaged_la=value):
                    manifest, defines, fields, receipt = self.generate(
                        family, 'nonpaged_la' + str(value), ['--nonpaged-la', str(value)])
                    self.assertEqual(bool(manifest['nonpaged_la']), bool(value))
                    receipt['pass'] = True

    def test_invalid_feature_values_are_rejected(self):
        cases = [('attention-impl', 'invalid', '--attention-impl must be'),
                 ('attention-buffers', '0', '--attention-buffers must be'),
                 ('mma-reg-pipe', '2', '--mma-reg-pipe must be'),
                 ('attention-frontier', '2', '--attention-frontier must be'),
                 ('ep-direct', '2', '--ep-direct must be'),
                 ('nonpaged-la', '2', '--nonpaged-la must be'),
                 ('watchdog', '2', '--watchdog must be'),
                 ('kphase-mask', '32', 'invalid K-phase'),
                 ('v3-poll-ns', '-1', 'invalid K-phase')]
        for family in ('dm',):
            for flag, value, message in cases:
                with self.subTest(family=family, flag=flag):
                    self.generate(family, 'reject_' + flag, ['--' + flag, value], message)

    def test_dense_decode_rejects_interpreted_options(self):
        message = 'belongs to the interpreted decode executor'
        for flags in (['--attention-impl', 'mma16'], ['--nonpaged-la', '1'], []):  # [] still carries --pg
            with self.subTest(flags=flags):
                self.generate('llm', 'mx4_' + '_'.join(flags).strip('-') or 'mx4_pg', flags, message)


if __name__ == '__main__':
    unittest.main()
