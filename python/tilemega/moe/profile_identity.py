"""Verify a routing profile before including it in plan and build identities."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _sha(value):
    if not isinstance(value, str) or len(value) != 64 or any(
            c not in '0123456789abcdef' for c in value):
        raise ValueError('invalid routing SHA256')
    return value


def _integer(value, minimum=0):
    if type(value) is not int or value < minimum or value > 2**53-1:
        raise ValueError('routing count must be an exact nonnegative integer')
    return value


def _input(identity, name):
    matches = [value for path, value in identity['inputs'].items()
               if Path(path).name == name]
    if len(matches) != 1:
        raise ValueError('missing or ambiguous routing input identity: ' + name)
    return _sha(matches[0])


def verify_profile(value, *, layers, experts, top_k, tokens=tuple(1 << i for i in range(13)),
                   block_rows=(16, 32, 64, 128)):
    """Return the content identity; histogram validation is independent of CUDA.

    The digest detects altered profile contents. It is not an authenticity proof
    for a checkpoint; the collector's per-tensor and artifact identities remain
    embedded in the result for provenance.
    """
    if value.get('schema') != 'tilemega.dm1.routing.profile.v1' or value.get('evidence') != 'verified':
        raise ValueError('unsupported or unverified routing profile')
    content = {key: item for key, item in value.items() if key != 'profile_id'}
    identity = _sha(value.get('profile_id'))
    if _digest(content) != identity:
        raise ValueError('routing profile content differs from its identity')
    layers, experts, top_k = (_integer(x, 1) for x in (layers, experts, top_k))
    if top_k > experts or not tokens or len(set(tokens)) != len(tokens):
        raise ValueError('invalid routing model/token coordinates')
    for token in tokens:
        _integer(token, 1)
    if not block_rows or len(set(block_rows)) != len(block_rows):
        raise ValueError('invalid routing binding block sizes')
    for b in block_rows:
        _integer(b, 1)
    sampling = value['sampling']
    if any(_integer(sampling[key], 1) != expected for key, expected in
           (('layers', layers), ('experts', experts), ('top_k', top_k))):
        raise ValueError('routing profile differs from model configuration')
    sampling_sha = hashlib.sha256((json.dumps(sampling, indent=2, sort_keys=True,
                                               allow_nan=False)+'\n').encode()).hexdigest()
    embedding = value['embedding_identity']
    if _input(embedding, 'sampling.json') != sampling_sha or \
            _input(embedding, 'tokens.json') != _sha(sampling['tokens_sha256']):
        raise ValueError('routing sampling or token identity differs')
    checkpoint = embedding['checkpoint']
    if _sha(checkpoint['config_sha256']) != _sha(sampling['config_sha256']):
        raise ValueError('routing checkpoint differs from sampling configuration')
    index_sha = _sha(checkpoint['index_sha256'])
    implementation = embedding['implementation']
    if implementation['attention'] != 'sdpa' or implementation['experts'] != 'grouped_mm':
        raise ValueError('routing reference implementation differs')
    for key in ('script_sha256', 'hf_model_source_sha256', 'hf_experts_source_sha256'):
        _sha(implementation[key])
    previous = _sha(embedding['output_sha256'])
    records = value['layers']
    if len(records) != layers:
        raise ValueError('incomplete routing layer profile')
    for i, record in enumerate(records):
        layer_identity = record['identity']
        if _integer(record['layer']) != i or _integer(layer_identity['layer']) != i:
            raise ValueError('routing layers are not in decoder order')
        if _input(layer_identity, f'hidden_{i:02d}.safetensors') != previous or \
                _input(layer_identity, 'sampling.json') != sampling_sha:
            raise ValueError('broken routing hidden-state identity chain')
        if layer_identity['implementation'] != implementation or \
                _sha(layer_identity['checkpoint']['index_sha256']) != index_sha or \
                _sha(layer_identity['checkpoint']['config_sha256']) != sampling['config_sha256']:
            raise ValueError('routing checkpoint or implementation changed between layers')
        _sha(layer_identity['routing_sha256'])
        if 'region_sha256' in layer_identity:
            _sha(layer_identity['region_sha256'])
        previous = _sha(layer_identity['output_sha256'])
        coordinates = record['coordinates']
        if set(coordinates) != {str(t) for t in tokens}:
            raise ValueError('routing token coordinates differ from requested domain')
        for t in tokens:
            point = coordinates[str(t)]
            windows = _integer(point['windows'], 1)
            if _integer(point['tokens'], 1) != t or \
                    _integer(point['assignments_per_window'], 1) != t*top_k:
                raise ValueError('invalid routing coordinate extent')
            def histogram(counts, minimum, maximum):
                parsed = {}
                for key, count in counts.items():
                    if not isinstance(key, str) or not key.isascii() or not key.isdecimal() or str(int(key)) != key:
                        raise ValueError('noncanonical routing histogram bin')
                    n, count = int(key), _integer(count, 1)
                    if not minimum <= n <= maximum:
                        raise ValueError('routing histogram bin outside extent')
                    parsed[n] = count
                if sum(parsed.values()) != windows:
                    raise ValueError('routing histogram omits windows')
                return parsed
            distinct = histogram(point['distinct_experts_histogram'], top_k, min(experts, t*top_k))
            distinct_total = sum(n*count for n, count in distinct.items())
            mean = point['expected_distinct_experts']
            if type(mean) not in (int, float) or not math.isfinite(mean) or abs(mean-distinct_total/windows) > 1e-10:
                raise ValueError('routing distinct-expert mean differs from histogram')
            if len(point['tokens_per_expert_histograms']) != experts:
                raise ValueError('routing profile omits experts')
            assignments, active, expert_histograms = 0, 0, []
            for h in point['tokens_per_expert_histograms']:
                counts = histogram(h, 0, t)
                expert_histograms.append(counts)
                assignments += sum(n*count for n, count in counts.items())
                active += sum(count for n, count in counts.items() if n)
            if assignments != windows*t*top_k or active != distinct_total:
                raise ValueError('routing histograms violate assignment conservation')
            groups = point.get('group_blocks_histograms', {})
            if set(groups) != {str(b) for b in block_rows}:
                raise ValueError('routing binding block coordinates differ')
            for b in block_rows:
                slots = t*top_k
                capacity = (slots+b-1)//b + min(experts, slots)
                blocks = histogram(groups[str(b)], (slots+b-1)//b, min(slots, capacity))
                expected_total = sum(((n+b-1)//b)*count for h in expert_histograms
                                     for n, count in h.items())
                if sum(n*count for n, count in blocks.items()) != expected_total:
                    raise ValueError('routing joint block mean differs from expert marginals')
    return identity


def read_profile(path, **model):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate routing JSON field: ' + key)
            result[key] = value
        return result
    raw = Path(path).read_bytes()
    value = json.loads(raw, object_pairs_hook=unique)
    identity = verify_profile(value, **model)
    return value, dict(profile_id=identity, file_sha256=hashlib.sha256(raw).hexdigest())
