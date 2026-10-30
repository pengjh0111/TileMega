"""Bind manifest implementation claims to the compiler's macro inputs."""
import re


INTEGER_FEATURES = {
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
    if manifest.get('program_kind') == 'moe' and manifest.get('program_version',2) == 3:
        if int(defines.get('TILEMEGA_MX_STATIC_MOE_V3','0')) != 1 or manifest.get('threads') != 512:
            raise ValueError('MoE V3 manifest requires its compiler macro and 512 threads')
    for feature, (macro, default) in INTEGER_FEATURES.items():
        if feature in manifest:
            actual = int(defines.get(macro, str(default)))
            if int(manifest[feature]) != actual:
                raise ValueError(f'manifest {feature} differs from {macro}')
    if 'attention_impl' in manifest:
        if 'TILEMEGA_ATTENTION_IMPL' in defines:
            index=int(defines['TILEMEGA_ATTENTION_IMPL'])
            if not 0<=index<=3:raise ValueError('invalid attention implementation macro')
            actual=('mma16','pvswap','pvswap2','simt')[index]
        else:
            actual = 'pvswap' if int(defines.get('TILEMEGA_ATTENTION_PVSWAP', '0')) else 'mma16'
        if manifest['attention_impl'] != actual:
            raise ValueError('manifest attention_impl differs from compiler macros')
    if 'kv_layout' in manifest and manifest.get('executor') not in ('mx_static', 'mx_static_prompt'):
        vt = bool(int(defines.get('TILEMEGA_KV_LAYOUT_VT', '0')))
        if (manifest['kv_layout'] == 'vt') != vt:
            raise ValueError('manifest kv_layout differs from TILEMEGA_KV_LAYOUT_VT')
        if 'kv_storage' in manifest:
            fragments = bool(int(defines.get('TILEMEGA_MX_KV_FRAGMENTS_V2', '0')))
            if (manifest['kv_storage'] == 'block_fragments') != fragments:
                raise ValueError('manifest kv_storage differs from TILEMEGA_MX_KV_FRAGMENTS_V2')
    if 'nonpaged_la' in manifest:
        actual = manifest.get('pg') != 'pages' and bool(int(defines.get('TILEMEGA_NONPAGED_LA', '0')))
        if bool(manifest['nonpaged_la']) != actual:
            raise ValueError('manifest nonpaged_la differs from compiler macros')


def implementations(manifest, defines):
    """The same implementation fields serve source checks and binary identity."""
    paged = manifest.get('pg') == 'pages'
    decode = manifest.get('phase') == 'decode'
    if manifest.get('executor') in ('mx_static', 'mx_static_prompt'):
        if manifest.get('version', 4) >= 5:
            if manifest.get('program_kind') == 'moe' and manifest.get('program_version',2) == 3:
                return {'executor':manifest['executor'],'version':manifest['version'],'program_version':3,
                    'kv_layout':manifest['kv_layout'],'kv_storage':manifest['kv_storage'],
                    'classes':manifest['classes'],'threads':manifest['threads'],
                    'b_min':manifest['b_min'],'attn_prefetch':manifest['attn_prefetch'],
                    'split':manifest['split'],'route_table_offset':manifest['route_table_offset'],
                    'decisions':manifest['decisions'],
                    'attention':{'impl':'MxAttentionV2','task_body':'MoeProgramV3'}}
            return {'executor': manifest['executor'], 'version': manifest['version'],
                    'kv_layout': manifest['kv_layout'], 'kv_storage': manifest['kv_storage'],
                    'classes': manifest['classes'], 'threads': manifest['threads'],
                    'b_min': manifest['b_min'], 'attn_prefetch': manifest['attn_prefetch'],
                    'decisions': manifest['decisions'],
                    'attention': {'impl': 'MxAttentionV2', 'task_body':
                        'MoeProgramV2' if manifest.get('program_kind') == 'moe' else 'DecodeProgramV2'}}
        return {'executor': 'mx_static', 'kv_layout': manifest.get('kv_layout'),
                'classes': manifest['classes'], 'attention_tmin': manifest['attention_tmin'],
                'barrier_prefetch': int(manifest.get('barrier_prefetch', 1)),
                'attention': {'impl': 'MxDecodeAttention', 'task_body': 'DecodeProgram'}}
    return {'dn_vector_sums': bool(manifest.get('dn_vector_sums', False)),
            'swiglu_interleave_u': int(manifest.get('swiglu_interleave_u', 16)),
            'gemms': [{'index': g['index'], 'impl': g.get('impl', 'mma16'),
                'task_body': 'PagedGemmTaskBody' if paged else (
                    'ServingGemvTaskBody' if g.get('impl') == 'gemv' else 'ServingGemmTaskBody'),
                'mma_reg_pipe': int(defines.get('TILEMEGA_MMA_REG_PIPE', '0')),
                'ep_direct': int(defines.get('TILEMEGA_EP_DIRECT', '0'))}
                for g in manifest['gemms']],
            'attention': {'impl': manifest.get('attention_impl', 'mma16'),
                'noinline': bool(manifest.get('attention_noinline', False)),
                'buffers': int(defines.get('TILEMEGA_ATTENTION_BUFFERS', '2')),
                'task_body': ('PagedAttentionTaskBody' if paged else
                    'IndependentAttentionTaskBody') if decode else 'FusedAttentionTaskBody',
                'page_policy': ('Packed' if paged else 'WarpPrivate') if decode else None,
                'load': manifest.get('paged_attention_load', 'loader'),
                'frontier': bool(int(defines.get('TILEMEGA_ATTENTION_FRONTIER', '0'))),
                **({'kv_storage': manifest['kv_storage']} if 'kv_storage' in manifest else {})}}

