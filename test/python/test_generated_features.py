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
    INTEGER_FEATURES, build_defines, implementations, validate, validate_gemm_dispatch,
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

    def generate(self, family, label, flags, rejection=None, *, fixture=None, target=None):
        folder = self.out / family / label
        folder.mkdir(parents=True)
        cuda = folder / 'plan.cu'
        fixture = fixture or (self.llm if family == 'llm' else ROOT / 'test/fixtures/dnn/dwpw_pair.json')
        decoder = family in ('llm', 'moe')
        command = [COMPILER, 'compile', str(fixture), str(cuda), '--emit', 'serving',
            '--serving', 'decode' if decoder else 'forward',
            '--frontend', 'decoder' if decoder else 'dnn', '--pg', 'l2',
            '--pdl', 'off', '--runtime-target', str(target or ROOT / 'test/fixtures/serving_rejection_target.json'),
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
        dispatch=validate_gemm_dispatch(manifest,defines,source)
        receipt['gemm_dispatch']=dispatch
        self.assertEqual(defines.get('TILEMEGA_DM_SUPPORT', '0'), '1' if family in ('dm','moe') else '0')
        fields = implementations(manifest, defines)
        receipt.update(source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                       defines=defines, implementations=fields)
        return manifest, defines, fields, receipt

    def test_integer_features_reach_both_frontends(self):
        for family in ('llm', 'dm'):
            for feature, (macro, default) in INTEGER_FEATURES.items():
                if feature == 'decode_tile_m':
                    continue  # Dedicated legal 8/16 Hopper decoder matrix below.
                values = (1, 2) if feature == 'attention_buffers' else (
                    (0, 200) if feature == 'v3_poll_ns' else (0, 31) if feature == 'kphase_mask' else (0, 1))
                for value in values:
                    with self.subTest(family=family, feature=feature, value=value):
                        if feature == 'attention_tma' and value == 1:
                            self.generate(family,feature+str(value),['--attention-tma','1'],
                                'attention TMA requires prefill WGMMA and qkv-epilogue 1')
                            continue
                        if feature == 'activation_tma' and value == 1:
                            message = ('activation TMA requires a native Hopper target' if family == 'llm'
                                       else 'activation TMA currently requires native nonpaged Hopper decode')
                            self.generate(family,feature+str(value),['--activation-tma','1'],message)
                            continue
                        if feature == 'cluster_dim' and value == 0:
                            self.generate(family,feature+str(value),['--cluster-dim','0'],
                                '--cluster-dim must be 1, 2 or 4')
                            continue
                        if feature == 'dependency_intern' and value == 1:
                            self.generate(family,feature+str(value),['--dependency-intern','1'],
                                'currently require sm_90')
                            continue
                        if feature == 'requested_residency' and value == 1:
                            self.generate(family,feature+str(value),['--residency','1'],
                                'currently require sm_90')
                            continue
                        if value == 1 and feature == 'moe_weight_bulk':
                            self.generate(family,feature+str(value),['--moe-weight-bulk','1'],
                                '--moe-weight-bulk requires')
                            continue
                        if value == 1 and feature in ('qkv_epilogue','moe_activation_bulk'):
                            # These require Hopper decode structure; the generic
                            # sm89 decoder/DNN fixtures deliberately lack it.
                            rejection = 'qkv epilogue requires' if feature == 'qkv_epilogue' else (
                                '--moe-activation-bulk requires' if family == 'dm' else 'MoE activation bulk requires')
                            self.generate(family,feature+str(value),
                                ['--'+feature.replace('_','-'),str(value)],rejection)
                            continue
                        flags = ['--' + feature.replace('_', '-'), str(value)]
                        if feature == 'requested_residency':
                            # Zero records an unspecified request; the public
                            # explicit --residency coordinate only admits 1/2.
                            flags = [] if value == 0 else ['--residency',str(value)]
                        result = self.generate(family, feature + str(value), flags)
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
        for family in ('llm', 'dm'):
            for attention in ('mma16', 'pvswap'):
                with self.subTest(family=family, attention=attention):
                    manifest, defines, fields, receipt = self.generate(
                        family, attention, ['--attention-impl', attention])
                    self.assertEqual(manifest['attention_impl'], attention)
                    # Decode-only arithmetic switches do not select encoder or
                    # prefill attention for a DNN forward plan.
                    self.assertEqual(fields['attention']['impl'], attention if family == 'llm' else 'mma16')
                    receipt['pass'] = True
            for value in (0, 1):
                with self.subTest(family=family, nonpaged_la=value):
                    manifest, defines, fields, receipt = self.generate(
                        family, 'nonpaged_la' + str(value), ['--nonpaged-la', str(value)])
                    self.assertEqual(bool(manifest['nonpaged_la']), bool(value))
                    receipt['pass'] = True

    def test_hopper_qkv_epilogue_binds_a_legal_decoder_projection(self):
        manifest,defines,fields,receipt=self.generate('llm','hopper_qkv_epilogue',
            ['--qkv-epilogue','1','--attention-impl','wgmma','--attention-page','uniform',
             '--deferred-norm','0','--resource-probes','0'],target=ROOT/'configs/targets/sm_90_h800.json')
        self.assertEqual(manifest['qkv_epilogue'],1)
        self.assertEqual(defines['TILEMEGA_QKV_EPILOGUE'],'1')
        self.assertEqual(defines['TILEMEGA_ARCH_ID'],'900')
        self.assertTrue(fields['qkv_epilogue'])
        self.assertEqual(fields['attention']['impl'],'wgmma')
        receipt['pass']=True

    def test_dependency_intern_binds_only_native_dm_models(self):
        manifest,defines,fields,receipt=self.generate('dm','hopper_dependency_intern',
            ['--dependency-intern','1'],target=ROOT/'configs/targets/sm_90_h800.json')
        self.assertEqual(manifest['dependency_intern'],1)
        self.assertEqual(defines['TILEMEGA_DEPENDENCY_INTERN'],'1')
        self.assertEqual(defines['TILEMEGA_ARCH_ID'],'900')
        self.assertEqual(defines['TILEMEGA_DM_SUPPORT'],'1')
        self.generate('llm','reject_dependency_intern',['--dependency-intern','1'],
            'dependency interning requires a DM model plan',target=ROOT/'configs/targets/sm_90_h800.json')
        receipt['pass']=True

    def test_prefill_tma_rejects_missing_preconditions(self):
        if 'attention_tma' not in INTEGER_FEATURES:
            self.skipTest('compiler snapshot predates prefill TMA')
        target=ROOT/'configs/targets/sm_90_h800.json'
        flags=['--serving','prefill','--pg','off','--seed-arch','native',
               '--prefill-attention-impl','wgmma','--qkv-epilogue','1',
               '--attention-tma','1','--deferred-norm','0',
               '--nonpaged-weight-layout','tiled','--resource-probes','0']
        self.generate('llm','reject_prefill_tma_qkv',flags+['--qkv-epilogue','0'],
            'attention TMA requires prefill WGMMA and qkv-epilogue 1',target=target)
        self.generate('llm','reject_prefill_tma_mma',flags+['--prefill-attention-impl','mma16'],
            'attention TMA requires prefill WGMMA and qkv-epilogue 1',target=target)
        self.generate('llm','reject_prefill_tma_value',flags+['--attention-tma','2'],
            '--attention-tma must be 0 or 1',target=target)

    def test_explicit_native_residency_uses_the_public_coordinate(self):
        if 'requested_residency' not in INTEGER_FEATURES:
            self.skipTest('compiler snapshot predates the residency request coordinate')
        manifest,defines,fields,receipt=self.generate('llm','hopper_residency1',
            ['--residency','1'],target=ROOT/'configs/targets/sm_90_h800.json')
        self.assertEqual(manifest['requested_residency'],1)
        self.assertEqual(defines['TILEMEGA_SERVING_RESIDENCY_REQUEST'],'1')
        self.assertEqual(defines['TILEMEGA_ARCH_ID'],'900')
        self.generate('llm','reject_residency0',['--residency','0'],
            '--residency must be 1 or 2',target=ROOT/'configs/targets/sm_90_h800.json')
        receipt['pass']=True

    def test_dense_mixed_implementations_reach_actual_runtime_classes(self):
        flags=['--deferred-norm','0','--nonpaged-weight-layout','tiled','--resource-probes','0',
               '--gemm-impl-lm-head','mma16']
        for family in ('qkv','o','gate-up','down'):flags+=['--gemm-impl-'+family,'wgmma']
        for executor in ('off','l2'):
            manifest,defines,fields,receipt=self.generate('llm','mixed_wgmma_mma_head_'+executor,
                flags+['--pg',executor],target=ROOT/'configs/targets/sm_90_h800.json')
            dispatch=receipt['gemm_dispatch']
            self.assertEqual(len(dispatch),5)
            self.assertTrue(all(g['impl']=='wgmma'for g in dispatch[:-1]))
            self.assertEqual(dispatch[-1]['impl'],'mma16')
            self.assertNotEqual(dispatch[0]['compiled_variant'],dispatch[-1]['compiled_variant'])
            self.assertEqual(int(defines['TILEMEGA_GEMM_VARIANT_COUNT']),2)
            receipt['pass']=True

    def test_cluster_combine_seed_imports_tile_owned_reducers(self):
        target=ROOT/'configs/targets/sm_90_h800.json'
        flags=['--nonpaged-la','1','--deferred-norm','0',
               '--nonpaged-weight-layout','tiled','--resource-probes','0']
        for family in ('qkv','o','gate-up','down','lm-head'):
            flags+=['--gemm-impl-'+family,'wgmma']
        for dimension in (2,4):
            cg=self.out/f'cluster{dimension}.mlir'
            manifest,defines,fields,receipt=self.generate('llm',f'cluster{dimension}',
                flags+['--combine','cluster','--cluster-dim',str(dimension),
                       '--dump-cg',str(cg)],target=target)
            self.assertEqual(manifest['combine'],'cluster')
            self.assertEqual(defines['TILEMEGA_CLUSTER_COMBINE'],'1')
            self.assertTrue(any(g['split_k']==dimension for g in manifest['gemms']))
            source=Path(str(cg.parent/'llm'/f'cluster{dimension}'/'plan.cu')).read_text()
            variant=re.search(r'\{kRuntimeGemms0, kDependencies0, \d+u, '
                r'kDependencyOffsets0, kSchedule0, \d+u, \d+u, \d+u, \d+u, (\d+)u',source)
            self.assertIsNotNone(variant)
            self.assertTrue(int(variant[1]) & 8)
            graph=cg.read_text()
            self.assertIn('tilemega.combiner_tile_per_block = true',graph)
            combines=[line for line in graph.splitlines()
                      if 'tmcg.tile_space' in line and '.combine' in line]
            self.assertTrue(combines)
            self.assertTrue(all('ownership = "tile_per_block"'in line for line in combines))
            # A CG imported with element ownership cannot gain a proof by
            # selecting the runtime cluster coordinate after import.
            invalid=cg.with_name(f'cluster{dimension}-invalid.mlir')
            invalid.write_text(graph.replace('tilemega.combiner_tile_per_block = true',
                                             'tilemega.combiner_tile_per_block = false'))
            self.generate('llm',f'reject_cluster{dimension}_unproved',
                ['--nonpaged-la','1','--combine','cluster','--cluster-dim',str(dimension)],
                'cluster combine requires tile-owned importer ownership and dependencies',
                fixture=invalid,target=target)
            receipt['pass']=True

    def test_decode_m8_requires_small_native_wgmma_decode(self):
        if 'decode_tile_m' not in INTEGER_FEATURES:
            self.skipTest('compiler snapshot predates the M8 coordinate')
        flags=['--decode-tile-m','8','--deferred-norm','0',
               '--nonpaged-weight-layout','tiled','--resource-probes','0']
        for family in ('qkv','o','gate-up','down','lm-head'):
            flags+=['--gemm-impl-'+family,'wgmma']
        target=ROOT/'configs/targets/sm_90_h800.json'
        for batch in (1,8):
            manifest,defines,fields,receipt=self.generate('llm',f'hopper_m8_b{batch}',
                flags+['--batch',str(batch)],target=target)
            self.assertEqual(manifest['decode_tile_m'],8)
            self.assertEqual(defines['TILEMEGA_DECODE_TILE_M'],'8')
            self.assertEqual(fields['decode_tile_m'],8)
            self.assertTrue(manifest['gemms'])
            self.assertTrue(all(g['impl']=='wgmma' and g['tile_m']==8 for g in manifest['gemms']))
            receipt['pass']=True
        self.generate('llm','reject_m8_b16',flags+['--batch','16'],
            'M8 requires native Hopper WGMMA decode B<=8',target=target)
        self.generate('llm','reject_m8_mma',['--decode-tile-m','8'],
            'M8 requires native Hopper WGMMA decode B<=8',target=target)
        self.generate('llm','reject_m8_forced_sm80',flags+['--arch-paths','sm80'],
            'WGMMA requires native Hopper decode',target=target)
        self.generate('llm','reject_m8_expert_bulk',flags+['--moe-activation-bulk','1'],
            'M8 requires native Hopper WGMMA decode B<=8',target=target)

    def test_activation_tma_requires_actual_native_dense_wgmma(self):
        if 'activation_tma' not in INTEGER_FEATURES:
            self.skipTest('compiler snapshot predates the activation TMA coordinate')
        flags=['--activation-tma','1','--deferred-norm','0','--nonpaged-weight-layout','tiled','--resource-probes','0']
        for family in ('qkv','o','gate-up','down','lm-head'):flags+=['--gemm-impl-'+family,'wgmma']
        manifest,defines,fields,receipt=self.generate('llm','hopper_activation_tma',flags,
            target=ROOT/'configs/targets/sm_90_h800.json')
        self.assertEqual(manifest['activation_tma'],1)
        self.assertEqual(defines['TILEMEGA_ACTIVATION_TMA'],'1')
        self.assertEqual(defines['TILEMEGA_ARCH_ID'],'900')
        self.assertTrue(all(g['impl']=='wgmma' for g in fields['gemms']))
        self.generate('llm','reject_mma_activation_tma',['--activation-tma','1','--deferred-norm','0'],
            'activation TMA requires an active dense M8/M16 WGMMA GEMM',target=ROOT/'configs/targets/sm_90_h800.json')
        self.generate('llm','reject_invalid_activation_tma',['--activation-tma','2'],
            '--activation-tma must be 0 or 1',target=ROOT/'configs/targets/sm_90_h800.json')
        receipt['pass']=True

    def test_fused_depthwise_pointwise_wgmma_requires_complete_channel_blocks(self):
        if 'dwpw_impl' not in implementations({'gemms': []}, {}):
            self.skipTest('compiler snapshot predates the fused DW/PW coordinate')
        import torch
        from tilemega.export_bridge import serialize
        class Pair(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.dw=torch.nn.Conv2d(64,64,3,stride=2,padding=1,groups=64,device='meta')
                self.pw=torch.nn.Conv2d(64,96,1,device='meta')
            def forward(self,x):
                return self.pw(torch.nn.functional.hardtanh(self.dw(x),0,6)).relu()
        program=torch.export.export(Pair().to(torch.bfloat16).eval(),
            (torch.empty((2,64,7,9),device='meta',dtype=torch.bfloat16),),
            dynamic_shapes={'x':{0:torch.export.Dim('batch',min=1,max=64)}})
        fixture=self.out/'dwpw_k64_fixture.json'
        fixture.write_text(json.dumps(serialize(program))+'\n')
        target=ROOT/'configs/targets/sm_90_h800.json'
        flags=['--dwpw-impl','wgmma','--dwpw-fuse','auto','--resource-probes','0']
        manifest,defines,fields,receipt=self.generate('dm','hopper_fused_dwpw',flags,
                                                    fixture=fixture,target=target)
        self.assertEqual(manifest['dwpw_impl'],'wgmma')
        self.assertEqual(fields['dwpw_impl'],'wgmma')
        self.assertEqual(defines['TILEMEGA_DWPW_WGMMA'],'1')
        self.assertEqual(defines['TILEMEGA_ARCH_ID'],'900')
        # Invocation stages live in generated device tables, rather than the
        # compact identity manifest. This fixture has exactly one PW GEMM.
        source=(self.out/'dm/hopper_fused_dwpw/plan.cu').read_text()
        self.assertIn('{TaskKind::kDwPwFused,',source)
        self.assertIn('DispatchDmFused',source)
        self.assertIn('#define TILEMEGA_DM_VARIANT_BINDINGS 1',source)
        self.assertRegex(source,r'DmVariantBinding<0> \{ static constexpr unsigned roles = 4u;')
        self.assertEqual(len(fields['gemms']),1)
        self.assertEqual(fields['gemms'][0]['impl'],'wgmma')
        self.assertEqual(manifest['gemms'][0]['tile_m'],64)
        self.assertEqual(manifest['gemms'][0]['tile_k'],64)
        self.generate('dm','reject_dwpw_incomplete_k64',flags,
            'DW/PW WGMMA requires a fused complete K64 channel block',target=target)
        self.generate('dm','reject_dwpw_fusion_disabled',['--dwpw-impl','wgmma','--dwpw-fuse','0'],
            'DW/PW WGMMA requires native Hopper DNN forward, fusion enabled',fixture=fixture,target=target)
        self.generate('dm','reject_dwpw_sm89',flags,'requested Hopper implementation requires native sm_90 target',fixture=fixture)
        receipt['pass']=True

    def test_hopper_bulk_gather_requires_real_row_gather_expert_gemms(self):
        import torch
        from tilemega.serving.export import export
        from tilemega.export_bridge import serialize
        directory=self.out/'moe_decoder_fixture'
        config=dict(architectures=['Qwen3MoeForCausalLM'],model_type='qwen3_moe',
            hidden_size=128,intermediate_size=128,moe_intermediate_size=64,num_hidden_layers=1,
            num_experts=4,num_experts_per_tok=2,num_attention_heads=2,num_key_value_heads=1,
            head_dim=64,vocab_size=128,rms_norm_eps=1e-6,rope_theta=1e6,norm_topk_prob=True,
            tie_word_embeddings=False,hidden_act='silu')
        export(config,'decode',192,directory,meta_weights=True)
        fixture=directory/'export.json'
        fixture.write_text(json.dumps(serialize(torch.export.load(directory/'exported_program.pt2')))+'\n')
        flags=['--moe-activation-bulk','1','--moe-binding','group','--moe-bm','16',
               '--deferred-norm','0','--nonpaged-weight-layout','tiled','--resource-probes','0']
        for family in ('qkv','o','gate-up','down','lm-head'):flags+=['--gemm-impl-'+family,'wgmma']
        manifest,defines,fields,receipt=self.generate('moe','hopper_bulk_gather',flags,
            fixture=fixture,target=ROOT/'configs/targets/sm_90_h800.json')
        self.assertEqual(manifest['moe_activation_bulk'],1)
        self.assertEqual(defines['TILEMEGA_MOE_ACTIVATION_BULK'],'1')
        self.assertEqual(defines['TILEMEGA_ARCH_ID'],'900')
        self.assertEqual(defines['TILEMEGA_DM_SUPPORT'],'1')
        self.assertTrue(fields['gemms'])
        self.assertTrue(all(g['impl']=='wgmma' for g in fields['gemms']))
        # The same complete MoE structure with MMA cannot claim bulk WGMMA.
        self.generate('moe','reject_mma_bulk_gather',['--moe-activation-bulk','1','--deferred-norm','0'],
            'MoE activation bulk requires a WGMMA row-gather expert GEMM',
            fixture=fixture,target=ROOT/'configs/targets/sm_90_h800.json')
        receipt['pass']=True

    def test_invalid_feature_values_are_rejected(self):
        cases = [('attention-impl', 'invalid', '--attention-impl must be'),
                 ('attention-buffers', '0', '--attention-buffers must be'),
                 ('mma-reg-pipe', '2', '--mma-reg-pipe must be'),
                 ('attention-frontier', '2', '--attention-frontier must be'),
                 ('ep-direct', '2', '--ep-direct must be'),
                 ('nonpaged-la', '2', '--nonpaged-la must be'),
                 ('watchdog', '2', '--watchdog must be'),
                 ('qkv-epilogue', '2', '--qkv-epilogue must be'),
                 ('moe-activation-bulk','2','--moe-activation-bulk requires'),
                 ('moe-weight-bulk','2','--moe-weight-bulk requires'),
                 ('kphase-mask', '32', 'invalid K-phase'),
                 ('v3-poll-ns', '-1', 'invalid K-phase')]
        for family in ('llm', 'dm'):
            for flag, value, message in cases:
                with self.subTest(family=family, flag=flag):
                    self.generate(family, 'reject_' + flag, ['--' + flag, value], message)


if __name__ == '__main__':
    unittest.main()
