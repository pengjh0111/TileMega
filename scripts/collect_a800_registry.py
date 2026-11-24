#!/usr/bin/env python3
"""Merge compatible A800 calibration targets without replacing observations."""
import argparse
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path


CONSTANTS = 'sm80_native_constants'
DECISIONS = 'sm80_native_decisions'
DEVICE = ('name', 'memory_mib', 'capability', 'mig', 'driver')
SUPPLEMENTS = ('dense_gemm_service', 'fused_expert_service', 'attention_fresh_service',
               'compact_attention_service', 'fused_trace_header_equivalence',
               'residual_copy_header_equivalence', 'default_body_equivalence')
MERGED = ('mma_mainloop', 'dense_gate_service', 'registry', *SUPPLEMENTS)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mma_shape(key, sms):
    grid, tn, tk, wn, stages = map(int, key.split(':'))
    if (grid not in (sms, 2*sms) or wn not in (1, 2, 4) or tn <= 0
            or tn % (8*wn) or tk <= 0 or tk % 64 or stages not in (1, 2, 4)
            or stages > tk//(16*(4//wn))):
        raise ValueError(f'invalid SM80 MMA key {key}')
    return dict(grid=grid, tn=tn, tk=tk, warps_n=wn, warps_k=4//wn,
                register_stages=stages)


def service_profiles(section):
    if section is None:
        return []
    if section['schema'] == 1:
        return [section]
    if section['schema'] == 2:
        return section['profiles']
    raise ValueError('unknown dense gate service schema')


def append_service(constants, additions):
    profiles = service_profiles(constants.get('dense_gate_service'))
    for item in additions:
        if item not in profiles:
            profiles.append(item)
    if profiles:
        constants['dense_gate_service'] = dict(schema=2, profiles=profiles)


def append_supplements(constants, section, path):
    for name in SUPPLEMENTS:
        supplement = section.get(name)
        if supplement is None:
            continue
        if name not in constants:
            constants[name] = deepcopy(supplement)
            continue
        existing = constants[name]
        if existing == supplement:
            continue
        if name == 'default_body_equivalence':
            for header, proof in supplement.items():
                if existing.setdefault(header, deepcopy(proof)) != proof:
                    raise ValueError(f'conflicting default body proof for {header} in {path}')
            continue
        if name == 'compact_attention_service':
            profiles = {}
            for profile in service_profiles(existing) + service_profiles(supplement):
                sample = profile['measurement']
                geometry = sample.get('dim', 128), sample.get('queries', 4), sample.get('norm', 0)
                if profiles.setdefault(geometry, profile) != profile:
                    raise ValueError(f'conflicting compact attention {geometry} service in {path}')
            constants[name] = dict(schema=2, profiles=list(map(deepcopy, profiles.values())))
            continue
        if ('profiles' not in supplement or 'profiles' not in existing
                or {k:v for k,v in supplement.items() if k != 'profiles'}
                != {k:v for k,v in existing.items() if k != 'profiles'}):
            raise ValueError(f'incompatible {name} reference in {path}')
        for profile in supplement['profiles']:
            if profile in existing['profiles']:
                continue
            if name == 'fused_expert_service':
                observation = {k:v for k,v in profile.items() if k != 'production_phases'}
                previous = next((row for row in existing['profiles']
                                 if {k:v for k,v in row.items() if k != 'production_phases'} == observation), None)
                if previous is not None:
                    phases = profile.get('production_phases')
                    if phases is not None:
                        if previous.setdefault('production_phases', deepcopy(phases)) != phases:
                            raise ValueError(f'conflicting fused phase observations in {path}')
                    continue
            existing['profiles'].append(deepcopy(profile))


def merge(paths):
    paths = [Path(p) for p in paths]
    base, *extra = [json.loads(p.read_text()) for p in paths]
    result = deepcopy(base)
    constants = result[CONSTANTS]
    constants.pop('dense_gate_service', None)
    table = constants['mma_mainloop']
    table['median_ns'] = {}
    table['p99_ns'] = {}
    table['supplements'] = []
    sms = base['resources']['num_sms']
    versions = {}
    sources = {}
    for path, target in zip(paths, (base, *extra)):
        source = digest(path)
        section = target[CONSTANTS]
        stable = {k:v for k,v in section.items() if k not in MERGED}
        if (base['arch_tag'] != 'sm_80' or target['arch_tag'] != 'sm_80'
                or 'A800' not in base['device_provenance']['name']
                or base['resources'] != target['resources']
                or any(base['device_provenance'][k] != target['device_provenance'][k] for k in DEVICE)
                or stable != {k:v for k,v in base[CONSTANTS].items() if k not in MERGED}
                or {k:v for k,v in base[DECISIONS].items() if k!='reuse_payload'}
                   != {k:v for k,v in target[DECISIONS].items() if k!='reuse_payload'}):
            raise ValueError(f'incompatible target or base constants in {path}')
        sources[source] = path.name
        if target[DECISIONS].get('reuse_payload'):
            result[DECISIONS]['reuse_payload'] = True
        append_service(constants, service_profiles(section.get('dense_gate_service')))
        append_supplements(constants, section, path)
        mma = section['mma_mainloop']
        keys = mma['median_ns'].keys()
        if keys != mma['p99_ns'].keys():
            raise ValueError(f'MMA median and p99 keys differ in {path}')
        supplemental = {}
        for supplement in mma.get('supplements', []):
            supplement_keys = set(supplement.get('shape_keys', supplement.get('measurements', {})))
            if not supplement_keys or not supplement_keys <= keys or supplemental.keys() & supplement_keys:
                raise ValueError(f'invalid MMA supplement keys in {path}')
            if supplement not in table['supplements']:
                table['supplements'].append(supplement)
            supplemental.update(dict.fromkeys(supplement_keys, supplement))
        for key in keys:
            shape = mma_shape(key, sms)
            median, p99 = (mma[stat][key] for stat in ('median_ns', 'p99_ns'))
            if not 0 < median <= p99 < math.inf:
                raise ValueError(f'invalid MMA measurement for {key}')
            origin = supplemental.get(key, mma)
            identity = {'gpu_uuid': origin['gpu_uuid'], 'evidence': origin['evidence']}
            if 'binary_sha256' in origin:
                identity['binary_sha256'] = origin['binary_sha256']
            else:
                # A guarded cohort can bind several linked binaries in one plan.
                for field in ('plan_sha256', 'admission_sha256'):
                    evidence_hash = origin[field]
                    if len(evidence_hash) != 64:
                        raise ValueError(f'invalid MMA cohort {field} in {path}')
                    identity[field] = evidence_hash
            row = dict(key=shape, unit='ns_per_k16_per_split', input_norm=True,
                       residency=shape['grid']//sms, **identity)
            previous = versions.setdefault(key, row)
            if (previous != row
                    or key in table['median_ns'] and
                    (table['median_ns'][key], table['p99_ns'][key]) != (median, p99)):
                raise ValueError(f'MMA conflict at {key}: {previous} vs {row} from {path}')
            table['median_ns'][key], table['p99_ns'][key] = median, p99
    constants['registry'] = dict(schema=1, sources=sources, mma=versions)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs='+', type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = merge(args.target)
    with args.out.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write('\n')
