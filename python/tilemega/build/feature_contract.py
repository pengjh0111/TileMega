"""Bind manifest implementation claims to the compiler's macro inputs."""
import re


INTEGER_FEATURES = {
    'dependency_intern': ('TILEMEGA_DEPENDENCY_INTERN', 0),
    'cluster_dim': ('TILEMEGA_SERVING_CLUSTER_DIM', 1),
    'qkv_epilogue': ('TILEMEGA_QKV_EPILOGUE', 0),
    'moe_activation_bulk': ('TILEMEGA_MOE_ACTIVATION_BULK', 0),
    'watchdog': ('TILEMEGA_WATCHDOG', 1),
    'attention_noinline': ('TILEMEGA_ATTENTION_NOINLINE', 0),
    'attention_buffers': ('TILEMEGA_ATTENTION_BUFFERS', 2),
    'parallel_argmax': ('TILEMEGA_EP_PARALLEL_ARGMAX', 0),
    'ep_direct': ('TILEMEGA_EP_DIRECT', 0),
    'attention_frontier': ('TILEMEGA_ATTENTION_FRONTIER', 0),
    'mma_reg_pipe': ('TILEMEGA_MMA_REG_PIPE', 0),
    'kphase_mask': ('TILEMEGA_KPHASE_CLASS_MASK', 31),
    'v3_poll_ns': ('TILEMEGA_V3_POLL_NS', 0),
    'l2_slim': ('TILEMEGA_L2_SLIM', 0),
    'page_loop_split': ('TILEMEGA_PAGE_LOOP_SPLIT', 0),
    'evict_first': ('TILEMEGA_EVICT_FIRST', 0),
    'evict_last': ('TILEMEGA_EVICT_LAST', 1),
}


def build_defines(generated, options):
    defines = {}
    for arg in options:
        if arg.startswith('-D'):
            key, _, value = arg[2:].partition('=')
            value = value or '1'
            if key in defines and defines[key] != value:
                raise ValueError('conflicting compiler macro: ' + key)
            defines[key] = value
    # Codegen emits overridable defaults in this exact three-line form.
    # Their inactive branch must not be mistaken for a source override.
    generated = re.sub(r'^#ifndef (\w+)\n(#define \1 [^\n]+\n)#endif\n',
                       lambda match: '' if match[1] in defines else match[2],
                       generated, flags=re.M)
    for key, value in re.findall(r'^#define\s+(\w+)\s+([^\n]+)', generated, re.M):
        value = value.strip()
        if key in defines and defines[key] != value:
            # A generated #define takes precedence in C++; accepting the
            # command-line value would give the artifact a false identity.
            raise ValueError('generated source overrides compiler macro: ' + key)
        defines[key] = value
    return defines


def validate(manifest, defines):
    seed_arch = manifest.get('seed_arch', 'legacy')
    if seed_arch not in ('native', 'legacy'):
        raise ValueError('manifest seed_arch must be native or legacy')
    if seed_arch == 'native':
        ids = {'sm_80':800, 'sm_89':890, 'sm_90':900, 'sm_100':1000, 'sm_120':1200}
        target = manifest.get('seed_target_arch')
        if target not in ids or int(defines.get('TILEMEGA_ARCH_ID', '800')) != ids[target]:
            raise ValueError('manifest native seed target differs from TILEMEGA_ARCH_ID')
    for feature, (macro, default) in INTEGER_FEATURES.items():
        if feature in manifest:
            actual = int(defines.get(macro, str(default)))
            if int(manifest[feature]) != actual:
                raise ValueError(f'manifest {feature} differs from {macro}')
    if 'attention_impl' in manifest:
        if 'TILEMEGA_ATTENTION_IMPL' in defines:
            index=int(defines['TILEMEGA_ATTENTION_IMPL'])
            if not 0<=index<=4:raise ValueError('invalid attention implementation macro')
            actual=('mma16','pvswap','pvswap2','simt','wgmma')[index]
        else:
            actual = 'pvswap' if int(defines.get('TILEMEGA_ATTENTION_PVSWAP', '0')) else 'mma16'
        if manifest['attention_impl'] != actual:
            raise ValueError('manifest attention_impl differs from compiler macros')
    if 'encoder_attention_impl' in manifest:
        actual = 'wgmma' if int(defines.get('TILEMEGA_ENCODER_ATTENTION_WGMMA', '0')) else 'mma16'
        if manifest['encoder_attention_impl'] != actual:
            raise ValueError('manifest encoder_attention_impl differs from compiler macros')
    if 'prefill_attention_impl' in manifest:
        actual = 'wgmma' if int(defines.get('TILEMEGA_PREFILL_ATTENTION_WGMMA', '0')) else 'mma16'
        if manifest['prefill_attention_impl'] != actual:
            raise ValueError('manifest prefill_attention_impl differs from compiler macros')
    if 'attention_page' in manifest:
        wgmma = int(defines.get('TILEMEGA_ATTENTION_IMPL', '0')) == 4
        uniform = bool(int(defines.get('TILEMEGA_ATTENTION_PAGE_UNIFORM', str(int(wgmma)))))
        if uniform != wgmma or manifest['attention_page'] != ('uniform' if uniform else 'chunk16'):
            raise ValueError('manifest attention_page differs from its legal compiler layout')
    if 'gemv_load' in manifest:
        actual = 'nc' if int(defines.get('TILEMEGA_GEMV_LOAD_NC', '0')) else 'ldg'
        if manifest['gemv_load'] != actual:
            raise ValueError('manifest gemv_load differs from compiler macros')
    if 'conv_impl' in manifest:
        actual = 'tma_im2col' if int(defines.get('TILEMEGA_CONV_TMA_IM2COL', '0')) else 'cpasync'
        if manifest['conv_impl'] != actual:
            raise ValueError('manifest conv_impl differs from compiler macros')
    if 'nonpaged_la' in manifest:
        actual = manifest.get('pg') != 'pages' and bool(int(defines.get('TILEMEGA_NONPAGED_LA', '0')))
        if bool(manifest['nonpaged_la']) != actual:
            raise ValueError('manifest nonpaged_la differs from compiler macros')


def implementations(manifest, defines):
    """The same implementation fields serve source checks and binary identity."""
    paged = manifest.get('pg') == 'pages'
    decode = manifest.get('phase') == 'decode'
    return {'seed_arch': manifest.get('seed_arch', 'legacy'),
            'execution_arch_id': int(defines.get('TILEMEGA_ARCH_ID', '800')),
            'qkv_epilogue': bool(manifest.get('qkv_epilogue', False)),
            'gemv_load': manifest.get('gemv_load', 'ldg'),
            'conv_impl': manifest.get('conv_impl', 'cpasync'),
            'encoder_attention': {'impl': manifest.get('encoder_attention_impl','mma16'),
                'task_body': 'Sm90EncoderAttentionTaskBody' if manifest.get('encoder_attention_impl') == 'wgmma' else 'EncoderAttentionTaskBody'},
            'dn_vector_sums': bool(manifest.get('dn_vector_sums', False)),
            'swiglu_interleave_u': int(manifest.get('swiglu_interleave_u', 16)),
            'gemms': [{'index': g['index'], 'impl': g.get('impl', 'mma16'),
                'task_body': 'PagedGemmTaskBody' if paged else (
                    'ServingGemvTaskBody' if g.get('impl') == 'gemv' else 'ServingGemmTaskBody'),
                'mma_reg_pipe': int(defines.get('TILEMEGA_MMA_REG_PIPE', '0')),
                'ep_direct': int(defines.get('TILEMEGA_EP_DIRECT', '0'))}
                for g in manifest['gemms']],
            'attention': {'impl': manifest.get('attention_impl', 'mma16') if decode else manifest.get('prefill_attention_impl', 'mma16'),
                'page_layout': manifest.get('attention_page',
                    'uniform' if manifest.get('attention_impl') == 'wgmma' else 'chunk16'),
                'noinline': bool(manifest.get('attention_noinline', False)),
                'buffers': int(defines.get('TILEMEGA_ATTENTION_BUFFERS', '2')),
                'task_body': ('PagedAttentionTaskBody' if paged else
                    'IndependentAttentionTaskBody') if decode else ('Sm90PrefillAttentionTaskBody'
                    if manifest.get('prefill_attention_impl') == 'wgmma' else 'FusedAttentionTaskBody'),
                'page_policy': ('WarpGroup' if manifest.get('attention_impl') == 'wgmma'
                    else 'Packed' if paged else 'WarpPrivate') if decode else None,
                'load': manifest.get('paged_attention_load', 'loader'),
                'frontier': bool(int(defines.get('TILEMEGA_ATTENTION_FRONTIER', '0')))}}

