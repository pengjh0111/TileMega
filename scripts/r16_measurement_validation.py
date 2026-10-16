"""Read-only validation of completed baseline receipts; standard library only."""
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


NATIVE_LLM_SYSTEMS = ('TileMega', 'TM-R15', 'TM-R15-IR', 'TM-RF1-on', 'TM-RF1-off', 'TM-seed',
    'TM-H1', 'TM-H1H4', 'TM-sm80', 'TM-noWGMMA',
    'TM-H3-stage', 'TM-H3-cluster2', 'TM-H3-cluster4',
    'TM-paged-mma', 'TM-paged-h1h4',
    'TM-H4d-stage', 'TM-H4d-cluster2', 'TM-H4d-cluster4')
R15_HEAD = '243fe9d06f5f8c55297c0c175295812f0843f329'
RF1_PATCH_SHA256 = '9b233022186721f305abeeb57928e7a64f4b9e5b252b0d75dedec7172c7fa0c6'
_HASH_CACHE = {}


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def file_sha(path):
    path = Path(path).resolve()
    stat = path.stat()
    key = str(path), stat.st_size, stat.st_mtime_ns
    if key not in _HASH_CACHE:
        result = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(8 << 20), b''):
                result.update(chunk)
        _HASH_CACHE[key] = result.hexdigest()
    return _HASH_CACHE[key]


def prefill_prefix_dependency_issues(identity, generated):
    """Identify the observed current-M-tile-only prefill QKV dependency.

    Attention also reads earlier tokens' K/V. A single producer M tile that
    covers a whole aligned sample is outside this specific failure scope.
    Inspect generated dependency tables rather than trusting a source version.
    """
    plan = identity.get('plan', {})
    seq = int(plan.get('seq', 1))
    if plan.get('phase') != 'prefill' or seq <= 1:
        return []
    stages = re.search(r'constexpr\s+StageDesc\s+kStages\[\]\s*=\s*\{(.*?)\n\};', generated, re.S)
    if stages is None:
        return []
    kinds = re.findall(r'\{TaskKind::(\w+),\s*(\d+)u?', stages[1])
    operands = []
    for row in re.findall(r'^\s*\{TaskKind::.*$', stages[1], re.M):
        values = re.search(r'\{([^{}]*)\}', row)
        first = values[1].split(',')[0].strip().removesuffix('u') if values else ''
        operands.append(int(first) if first.isdecimal() else None)
    descriptors = re.search(r'constexpr\s+GemmDesc\s+kGemms\[\]\s*=\s*\{(.*?)\n\};', generated, re.S)
    destinations = re.findall(r'\{\s*\d+,\s*\d+,\s*\d+u?,\s*\d+u?,\s*\d+u?,\s*(\d+)u?', descriptors[1]) if descriptors else []
    issues = []
    for deps in re.finditer(r'constexpr\s+StageDependency\s+kDependencies(\d+)\[\]\s*=\s*\{(.*?)\n\};', generated, re.S):
        variant = deps[1]
        table = re.search(r'constexpr\s+GemmRuntimeDesc\s+kRuntimeGemms'+variant+r'\[\]\s*=\s*\{(.*?)\n\};', generated, re.S)
        if table is None:
            continue
        gemms = re.findall(r'\{\s*\d+u?,\s*\d+u?,\s*(\d+)u?,\s*\d+u?,\s*\d+u?,\s*\d+u?\s*\}', table[1])
        rows = re.findall(r'\{(\d+)u?,\s*(\d+)u?,\s*StageDependency::Map::(\w+),\s*(\d+)u?,\s*(-?\d+),\s*(-?\d+),\s*(\d+)u?,\s*\d+u?\s*\}', deps[2])
        for producer, consumer, mapping, divisor, scale, offset, count in rows:
            producer, consumer = int(producer), int(consumer)
            if max(producer, consumer) >= len(kinds):
                raise ValueError('native dependency stage index outside stage table')
            kind, gemm = kinds[producer]
            if kinds[consumer][0] != 'kFusedAttention' or kind not in ('kGemm', 'kGemmCombine'):
                continue
            if int(gemm) >= len(destinations) or consumer >= len(operands):
                raise ValueError('native prefill QKV operand metadata is incomplete')
            if int(destinations[int(gemm)]) != operands[consumer]:
                continue
            if int(gemm) >= len(gemms):
                raise ValueError('native dependency GEMM index outside runtime table')
            tile_m = int(gemms[int(gemm)])
            narrow = mapping == 'kIdentity' or (mapping == 'kWindow' and int(divisor) == 1
                and int(offset) == 0 and int(scale) == int(count))
            if tile_m < seq and narrow:
                issues.append(dict(runtime_variant=int(variant), producer=producer,
                    consumer=consumer, tile_m=tile_m, seq=seq, mapping=mapping,
                    reason='current M tile omits earlier-token prefill K/V dependencies'))
    return issues


def native_identity(binary, recorded=None):
    binary = Path(binary)
    identity = json.loads(Path(str(binary)+'.identity.json').read_text())
    if identity.get('schema') != 1 or identity.get('artifact_id') != json_digest({
            key: value for key, value in identity.items() if key != 'artifact_id'}):
        raise ValueError('native artifact identity digest differs')
    if recorded is not None and recorded != identity:
        raise ValueError('recorded native identity differs from its sidecar')
    if identity.get('trace') is not False:
        raise ValueError('instrumented native artifact is ineligible for performance')
    for suffix, key in (('', 'binary_sha256'), ('.cu', 'generated_source_sha256'),
                        ('.plan.json', 'manifest_sha256'), ('.ptxas.log', 'resources_sha256')):
        if file_sha(str(binary)+suffix) != identity.get(key):
            raise ValueError('native artifact sidecar or binary changed: '+suffix)
    if 'build_command_sha256' in identity and file_sha(str(binary)+'.build_command.txt') != identity['build_command_sha256']:
        raise ValueError('native compiler command changed')
    macros = identity.get('complete_macros')
    if macros is not None:
        if macros['compiler_command'] != identity['options']:
            raise ValueError('native macro capture command differs')
        for unit in macros['translation_units']:
            if file_sha(unit['macros_file']) != unit['macros_sha256']:
                raise ValueError('native macro capture changed')
    generated = Path(str(binary)+'.cu').read_text()
    issues = prefill_prefix_dependency_issues(identity, generated)
    if issues:
        raise ValueError('known invalid prefill K/V prefix dependency; regenerate and retest: '+json.dumps(issues))
    if 'gemms' in identity['plan']:
        # An independent, stdlib-only check prevents matching sidecar hashes
        # from certifying a manifest that disagrees with actual kernel dispatch.
        table = re.search(r'constexpr\s+GemmRuntimeDesc\s+kRuntimeGemms0\[\]\s*=\s*\{(.*?)\n\};',
                          generated, re.S)
        if table is None:
            raise ValueError('native generated GEMM dispatch table is absent')
        rows = re.findall(r'\{\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?,\s*(\d+)u?\s*\}', table[1])
        if len(rows) != len(identity['plan']['gemms']):
            raise ValueError('native generated GEMM count differs from manifest')
        defines = identity['defines']
        for index, (gemm, row) in enumerate(zip(identity['plan']['gemms'], rows)):
            variant, split, m, n, k, stages = map(int, row)
            if gemm['index'] != index or not 0 <= variant < int(defines.get('TILEMEGA_GEMM_VARIANT_COUNT', 1)):
                raise ValueError('native GEMM class or index is invalid')
            geometry = dict(split_k=split, tile_m=m, tile_n=n, tile_k=k, stages=stages)
            if any(int(gemm[field]) != value for field, value in geometry.items()):
                raise ValueError('native runtime GEMM geometry differs from manifest')
            prefix = 'TILEMEGA_GEMM_' if variant == 0 else f'TILEMEGA_GEMM_V{variant}_'
            if any(int(defines[prefix+field]) != value for field, value in
                    (('TILE_M',m), ('TILE_N',n), ('TILE_K',k), ('STAGES',stages))):
                raise ValueError('native compiled GEMM geometry differs from runtime table')
            impl = int(defines.get(f'TILEMEGA_GEMM_IMPL_{variant}', 0))
            if impl not in (0,1,2) or gemm.get('impl', 'mma16') != ('mma16','gemv','wgmma')[impl]:
                raise ValueError('native actual GEMM implementation differs from manifest')
    return identity


def canonical_model(root, config_sha, fallback=None):
    candidates = [directory.name for directory in (Path(root)/'runs/r16/models').glob('*')
                  if (directory/'config.json').is_file() and file_sha(directory/'config.json') == config_sha]
    if len(candidates) == 1:
        return candidates[0]
    if not candidates and fallback is not None:
        return fallback
    raise ValueError('model config SHA does not identify one canonical R16 model')


def pristine_r15(identity, root):
    source = identity['source']
    compiler = Path(root)/'runs/r16/build-r15/tools/tilemega'
    if (source.get('head') != R15_HEAD or
            source.get('worktree_diff_sha256') != hashlib.sha256(b'').hexdigest() or
            Path(source.get('root', '')).resolve() != (Path(root)/'runs/r16/tm-r15').resolve() or
            Path(source.get('compiler', '')).resolve() != compiler.resolve() or
            source.get('compiler_sha256') != file_sha(compiler)):
        raise ValueError('TM-R15 requires the pristine baseline HEAD, source and compiler')


def rf1_ablation(identity, root):
    source = identity['source']
    directory = Path(root)/'runs/r16/rf1/r15-no-l1-spread'
    compiler = directory/'build/tools/tilemega'
    if (source.get('head') != R15_HEAD or
            source.get('worktree_diff_sha256') != RF1_PATCH_SHA256 or
            Path(source.get('root', '')).resolve() != (directory/'source').resolve() or
            Path(source.get('compiler', '')).resolve() != compiler.resolve() or
            source.get('compiler_sha256') != file_sha(compiler)):
        raise ValueError('TM-RF1-off requires the recorded single-function R15 ablation')


def native_llm(value, root):
    if root is None:
        raise ValueError('native receipt requires the canonical R16 model registry')
    model = canonical_model(root, value['model_config_sha256'])
    config = value['configuration']
    identities = []
    for index, phase in enumerate(('prefill', 'decode')):
        identity = native_identity(config[phase+'_binary'], config['identities'][phase])
        provenance = identity.get('source', {}).get('frontend_ir_provenance')
        if provenance is not None and value['system'] != 'TM-R15-IR':
            raise ValueError('repaired frontend IR requires the explicit TM-R15-IR label')
        if value['system'] in ('TM-R15', 'TM-R15-IR', 'TM-RF1-on'):
            pristine_r15(identity, root)
            if value['system'] == 'TM-R15-IR':
                if phase == 'prefill':
                    # Lazy import keeps existing frozen, non-IR harnesses standalone.
                    from r16_r15_ir_provenance import validate_frontend_ir
                    validate_frontend_ir(identity, root)
                elif provenance is not None:
                    raise ValueError('TM-R15-IR decode must retain the original R15 frontend')
        elif value['system'] == 'TM-RF1-off':
            rf1_ablation(identity, root)
        if identity['binary_sha256'] != config[phase+'_sha256'] or identity['plan'] != config['plan_manifests'][index]:
            raise ValueError('native phase manifest or binary SHA differs')
        identities.append(identity['artifact_id'])
    path = Path(config['checkpoint_manifest'])
    if path.resolve() != (Path(value['model'])/'random_checkpoint.json').resolve():
        raise ValueError('shared checkpoint manifest belongs to a different model directory')
    checkpoint = json.loads(path.read_text())
    payload = {key: checkpoint[key] for key in ('algorithm', 'config_sha256', 'seed', 'aliases', 'tensors')}
    checkpoint_sha = json_digest(payload)
    if (checkpoint['state'] not in ('live', 'closed') or checkpoint['config_sha256'] != value['model_config_sha256'] or
            checkpoint.get('checkpoint_sha256') != checkpoint_sha or config.get('checkpoint_sha256') != checkpoint_sha):
        raise ValueError('shared checkpoint identity differs or supervisor failed')
    return dict(canonical_model=model, native_artifact_ids=','.join(identities),
                source_checkpoint_sha256=checkpoint_sha, checkpoint_identity_shared=True,
                source_checkpoint_label='shared synthetic TileMega/HF checkpoint; pretrained C-1 unavailable')


def native_dnn(value):
    identity = native_identity(value['library_path'], value['identity'])
    if value.get('artifact_id') != identity['artifact_id']:
        raise ValueError('native DNN artifact ID differs')
    execution = value['execution_identity']
    if (execution.get('artifact_id') != identity['artifact_id'] or execution.get('trace') is not False or
            execution.get('execution_id') != json_digest({key: item for key, item in execution.items() if key != 'execution_id'})):
        raise ValueError('native DNN execution identity differs')
    if file_sha(value['numerical_gate_path']) != value['numerical_gate_sha256'] or file_sha(value['input_path']) != value['input_sha256']:
        raise ValueError('native DNN numerical gate or input changed')
    gate = json.loads(Path(value['numerical_gate_path']).read_text())
    if (gate.get('passed') is not True or gate.get('synthetic_weights') or gate.get('artifact_id') != identity['artifact_id'] or
            gate.get('input_tensors_sha256') != value['input_sha256']):
        raise ValueError('native DNN gate does not certify the timed artifact and input')
    if gate.get('archive_sha256') and value.get('exported_program_sha256') != gate['archive_sha256']:
        raise ValueError('native DNN export archive differs from its numerical gate')
    if value.get('source_checkpoint_sha256'):
        checkpoint=Path(value['export_path'])/'checkpoint/model.safetensors'
        if (gate.get('source_checkpoint_sha256') != value['source_checkpoint_sha256'] or
                file_sha(checkpoint) != value['source_checkpoint_sha256']):
            raise ValueError('native DNN source checkpoint differs from its numerical gate')
    return dict(native_artifact_ids=identity['artifact_id'], source_checkpoint_label='stored export checkpoint certified by native input gate')


def positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def equal(actual, expected, name):
    positive(actual, name)
    if not math.isclose(actual, expected, rel_tol=1e-8, abs_tol=1e-12):
        raise ValueError(f"{name} differs from raw samples/formula")


def ownership(guard, samples):
    acceptance = guard.get('acceptance')
    if acceptance not in ('strict', 'provisional_container_pid_namespace'):
        raise ValueError('unknown guard ownership acceptance')
    unresolved = sorted({pid for sample in samples for pid in sample.get('unresolved_host_pids', [])})
    verified = guard.get('ownership_verified') is True and all(
        sample.get('ownership_verified') is True for sample in samples)
    if acceptance == 'strict' and (not verified or unresolved):
        raise ValueError('strict ownership is unsupported by guard evidence')
    return dict(acceptance=acceptance, ownership_verified=verified, unresolved_host_pids=unresolved)


def telemetry(path):
    samples = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not samples:
        raise ValueError('missing telemetry')
    for sample in samples:
        if sample.get('accepted') is not True:
            raise ValueError('measurement telemetry was rejected')
        positive(sample.get('time'), 'telemetry timestamp')
        data = sample.get('telemetry', {})
        try:
            if int(data['clocks_throttle_reasons.active'], 0) & 0xE8:
                raise ValueError('thermal slowdown or power brake')
        except (KeyError, TypeError):
            raise ValueError('throttle evidence unavailable')
        for name in ('power.draw', 'clocks.sm', 'clocks.mem', 'temperature.gpu'):
            positive(float(data[name]), name)
    return samples


def completed_guard(directory, job):
    invalidation = Path(directory)/'guard_invalidation.json'
    if invalidation.exists():
        reason = json.loads(invalidation.read_text()).get('reason', 'guard invalidated')
        raise ValueError('completed guard was invalidated; retest required: '+reason)
    if job.get('status') != 'done' or job.get('exit_code') != 0:
        raise ValueError('job did not complete successfully')
    guard = json.loads((directory/'guard_result.json').read_text())
    if guard.get('code') != 0:
        raise ValueError('guard did not accept completed run')
    return guard


def validate_native_repetitions(path, value=None):
    """Check native greedy repeats for short correctness and full timing runs."""
    path = Path(path)
    value = json.loads(path.read_text()) if value is None else value
    if value.get('system') not in NATIVE_LLM_SYSTEMS:
        return
    reference = {}
    for row in value.get('generation_runs', []):
        count = row['N']
        token_path = path.parent/row['tokens_file']
        if token_path.resolve().parent != path.parent.resolve():
            raise ValueError('token file escapes its receipt directory')
        tokens = json.loads(token_path.read_text())
        if count in reference and tokens != reference[count]:
            raise ValueError(f'native greedy tokens differ between repetitions for N={count}')
        reference[count] = tokens


def validate_llm(path, guard, root=None):
    path = Path(path)
    value = json.loads(path.read_text())
    if (path.parent/'invalid_measurements.json').exists():
        raise ValueError('invalid measurement receipt coexists with summary')
    if value.get('max_tokens') != 1024 or value.get('prompt_length') != 64:
        raise ValueError('not the final 64/1024 protocol')
    if value.get('valid') is False or value.get('evidence') != 'verified':
        raise ValueError('receipt does not report verified execution')
    package = {'vllm':'vllm', 'sglang':'sglang', 'tensorrt-llm':'tensorrt-llm', 'torch.compile':'transformers'}.get(value.get('system'))
    native = value.get('system') in NATIVE_LLM_SYSTEMS
    if not value.get('versions', {}).get('torch') or (not native and (not package or not value['versions'].get(package))):
        raise ValueError('framework or torch version was not recorded')
    batch = value['batch']
    if type(batch) is not int or batch not in (1, 2, 4, 8, 16):
        raise ValueError('invalid static batch')
    rows, medians = value.get('generation_runs', []), {}
    for count in (1, 1024):
        group = [row for row in rows if row.get('N') == count]
        if (len(group) != 4 or sum(row.get('warmup') is True for row in group) != 1 or
                {row.get('run') for row in group} != {0, 1, 2, 3}):
            raise ValueError('expected one warmup and three timed repetitions')
        timed = []
        for row in group:
            if row.get('warmup') is not (row.get('run') == 0):
                raise ValueError('warmup must precede all timed repetitions')
            elapsed = positive(row.get('wall_seconds'), 'request duration')
            token_path = path.parent/row['tokens_file']
            if token_path.resolve().parent != path.parent.resolve():
                raise ValueError('token file escapes its receipt directory')
            tokens = json.loads(token_path.read_text())
            if (len(tokens) != batch or any(not isinstance(token_row, list) or len(token_row) != count or
                    any(type(token) is not int or token < 0 for token in token_row) for token_row in tokens)):
                raise ValueError('token file has the wrong shape or invalid IDs')
            if row.get('warmup') is False:
                timed.append(elapsed)
            elif row.get('warmup') is not True:
                raise ValueError('warmup flag must be boolean')
        medians[count] = statistics.median(timed)
    if len(rows) != 8:
        raise ValueError('unexpected request measurements')
    # Executor gates use the last repetition; short numeric gates need this
    # same check before they can authorize later candidate selection.
    validate_native_repetitions(path, value)
    equal(value['e2e_seconds'], medians[1024], 'e2e_seconds')
    equal(value['ttft_seconds'], medians[1], 'ttft_seconds')
    equal(value['tpot_seconds'], (medians[1024]-medians[1])/1023, 'tpot_seconds')
    equal(value['output_tokens_per_second'], batch*1024/medians[1024], 'throughput')
    config_file = Path(value['model'])/'config.json'
    if hashlib.sha256(config_file.read_bytes()).hexdigest() != value['model_config_sha256']:
        raise ValueError('model config changed since measurement')
    log = path.parent/'guard.jsonl'
    if Path(value['guard_log']).resolve() != log.resolve():
        raise ValueError('guard log belongs to another measurement')
    samples = telemetry(log)
    labels = {sample['label']: sample for sample in samples}
    if len(labels) != len(samples):
        raise ValueError('duplicate telemetry labels')
    for row in rows:
        for suffix in ('before', 'after'):
            if f"N{row['N']}-run{row['run']}-{suffix}" not in labels:
                raise ValueError('missing request telemetry boundary')
    own = ownership(guard, samples)
    if value.get('ownership_verified') != own['ownership_verified'] or value.get('acceptance') != own['acceptance']:
        raise ValueError('summary ownership differs from its guard evidence')
    if native:
        value['_r16_native'] = native_llm(value, root)
    return value, samples, own


def request_intervals(value, samples):
    labels = {sample['label']: sample for sample in samples}
    result = []
    for row in value['generation_runs']:
        if row['warmup'] or row['N'] != 1024:
            continue
        prefix = f"N1024-run{row['run']}"
        before, after = labels[prefix+'-before']['time'], labels[prefix+'-after']['time']
        duration = row['wall_seconds']
        if after-before < duration-1e-6:
            raise ValueError('wall clock boundaries do not enclose request duration')
        # The actual request lies somewhere within [before,after] and lasts d.
        # Its guaranteed inner interval is [after-d,before+d]; outer intervals
        # alone could overlap only because nvidia-smi took time between requests.
        inner_start, inner_end = after-duration, before+duration
        result.append(dict(run=row['run'], duration=duration, outer_start=before, outer_end=after,
                           inner_start=inner_start, inner_end=max(inner_start, inner_end)))
    return result


def validate_dnn(path, guard):
    path = Path(path)
    value = json.loads(path.read_text())
    if value.get('valid') is False or value.get('evidence') != 'verified' or (path.parent/'invalid_measurements.json').exists():
        raise ValueError('DNN receipt is marked invalid or unverified')
    protocol = value['measurement']
    if protocol.get('warmup') != 50 or protocol.get('iterations') != 200:
        raise ValueError('not the final DNN warmup50/iterations200 protocol')
    values = value.get('samples_ms', [])
    if len(values) != 200:
        raise ValueError('DNN sample count differs')
    for elapsed in values:
        positive(elapsed, 'DNN CUDA event duration')
    equal(value['median_ms'], statistics.median(values), 'DNN median')
    equal(value['p90_ms'], sorted(values)[math.ceil(.9*len(values))-1], 'DNN p90')
    log = path.parent/'guard.jsonl'
    if Path(value['guard_log']).resolve() != log.resolve():
        raise ValueError('DNN guard log belongs to another measurement')
    samples = telemetry(log)
    if not {'before-measurement', 'after-measurement'} <= {sample['label'] for sample in samples}:
        raise ValueError('DNN measurement telemetry missing')
    own = ownership(guard, samples)
    if value.get('system') in ('tilemega', 'TileMega'):
        value['_r16_native'] = native_dnn(value)
    return value, own
