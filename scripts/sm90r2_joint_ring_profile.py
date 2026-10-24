#!/usr/bin/env python3
"""Exact native joint-ring profile reader and development price equation.

No calibration import, fitting, interpolation, or steady-to-finite extrapolation.
Synthetic fixtures remain explicitly labelled and are never written to targets.
"""
import copy
import math

DIMENSIONS = ('batch', 'tile_n', 'tile_k', 'slots', 'warps_n', 'warps_k',
              'register_stages', 'grid', 'active_ctas', 'compiled_residency', 'input_k', 'tasks_per_cta',
              'prefix_bytes', 'prefetch_l2_bytes', 'dynamic_shared_bytes', 'workspace_offset')
ENDPOINT_MODEL = 'producer_call_to_finalize_observed_v1'
CACHE_STATE = 'four_shape_replay_no_scrub'


class MissingJointRingProfile(ValueError):
    def __init__(self, dimensions, cache_state, mma):
        super().__init__('missing exact native joint-ring profile; calibration pending')
        self.measurement = dict(metric='native_joint_ring_profile',
                                dimensions=copy.deepcopy(dimensions), labels=dict(cache_state=cache_state, mma=mma))


def number(value, name, zero=False):
    n = value[name]
    if isinstance(n, bool) or not isinstance(n, (float, int)) or not math.isfinite(n) or (n < 0 if zero else n <= 0):
        raise ValueError('invalid native joint-ring interval: ' + name)
    return n


class JointRingProfiles:
    def __init__(self, value, *, expected_producer_header_sha256):
        if value['schema'] != 1 or isinstance(value['schema'], bool) or value['endpoint_model'] != ENDPOINT_MODEL:
            raise ValueError('unsupported native joint-ring profile schema/endpoints')
        binding = value['source_binding']
        for name in ('source_digest', 'producer_header_sha256', 'probe_sha256', 'binary_sha256', 'collector_sha256'):
            digest = binding[name]
            if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
                raise ValueError('invalid native joint-ring source binding')
        if binding['producer_header_sha256'] != expected_producer_header_sha256:
            raise ValueError('native joint-ring producer header differs from calibration')
        self.entries, ids, coordinates = [], set(), set()
        for raw in value['profiles']:
            row = copy.deepcopy(raw)
            profile_id = row['profile_id']
            if not isinstance(profile_id, str) or not profile_id or profile_id in ids:
                raise ValueError('duplicate/empty native joint-ring profile id')
            ids.add(profile_id)
            d = {name: row[name] for name in DIMENSIONS}
            for name, n in d.items():
                if (isinstance(n, bool) or not isinstance(n, (int, float)) or not math.isfinite(n) or
                    n != math.floor(n) or not 0 <= n <= 2147483647 or
                    (n == 0 and name not in ('prefix_bytes', 'prefetch_l2_bytes', 'workspace_offset'))):
                    raise ValueError('invalid native joint-ring dimension: ' + name)
            processes = number(row, 'processes')
            if (row['cache_state'] != CACHE_STATE or row['mma'] != 'mma_sync' or
                row['aggregation'] != 'per_cta_sample_median' or
                row['evidence'] not in ('verified', 'synthetic_cpu_fixture') or
                processes < 5 or processes != math.floor(processes) or row['continuous_task_drain'] is not False or
                d['tasks_per_cta'] != 1 or d['compiled_residency'] != 1 or d['prefix_bytes'] or d['prefetch_l2_bytes'] or
                d['active_ctas'] > d['grid'] or d['warps_n'] * d['warps_k'] != 4 or
                d['input_k'] < d['tile_k'] or d['input_k'] % d['tile_k'] or not 2 <= d['slots'] <= 8 or
                d['batch'] not in (1, 2, 4, 8, 16) or d['warps_n'] not in (1, 2, 4) or
                d['tile_n'] % (8*d['warps_n']) or d['tile_k'] % 64 or
                d['register_stages'] not in (1, 2, 4) or d['register_stages'] > d['tile_k']/(16*d['warps_k']) or
                d['workspace_offset'] % 1024 or d['dynamic_shared_bytes'] < d['workspace_offset'] + 1024 +
                2*d['slots']*d['tile_k']*(16+d['tile_n']) + 64*d['warps_k']*d['tile_n'] + 128):
                raise ValueError('unsupported native joint-ring profile scope/evidence')
            key = tuple(d.values()) + (row['cache_state'],)
            if key in coordinates:
                raise ValueError('duplicate native joint-ring profile coordinates')
            coordinates.add(key)
            total = number(row, 'issue_to_finalize_ns')
            first = number(row, 'first_full_ns')
            if any(number(row, name, True) > total for name in ('main_span_ns', 'finalize_tail_ns')) or first > total:
                raise ValueError('invalid native joint-ring diagnostic span')
            count = min(int(d['slots']) + 1, int(d['input_k'] // d['tile_k']))
            prefix = row['finite_prefix']
            if len(prefix) != count:
                raise ValueError('incomplete native joint-ring cold-fill prefix')
            prior = 0
            for q, point in enumerate(prefix):
                full, release = number(point, 'full_ns'), number(point, 'release_ns')
                if (isinstance(point['sequence'], bool) or point['sequence'] != q or full < prior or
                    release < full or release > total or (q == 0 and full != first)):
                    raise ValueError('nonmonotone native joint-ring cold-fill prefix')
                prior = release
            row['source_binding'] = copy.deepcopy(binding)
            self.entries.append((d, row))

    def find(self, dimensions, cache_state=CACHE_STATE, mma='mma_sync'):
        for d, row in self.entries:
            if d == dimensions and row['cache_state'] == cache_state and mma == 'mma_sync':
                return copy.deepcopy(row)
        raise MissingJointRingProfile(dimensions, cache_state, mma)


def quote_stage(profile, *, barrier_ns, initialization_ns, epilogue_ns, issue_ns=0, receive_ns=0):
    """Mirror the new C++ branch; the old pooled residual awaits refitting."""
    components = dict(barrier_ns=barrier_ns, initialization_ns=initialization_ns,
                      epilogue_ns=epilogue_ns, issue_ns=issue_ns, receive_ns=receive_ns)
    for name in components:
        number(components, name, True)
    flow = number(profile, 'issue_to_finalize_ns')
    receipt = copy.deepcopy(profile)
    receipt['residual_model'] = 'none_pending_independent_refit'
    receipt['diagnostic_only'] = True
    receipt['fit_status'] = 'joint_ring_diagnostic_pending_refit'
    return dict(stage_ns=barrier_ns + initialization_ns + flow + epilogue_ns + issue_ns + receive_ns,
                startup_ns=initialization_ns, flow_ns=flow, compute_ns=0, activation_ns=0,
                read_startup_ns=0, joint_ring_profile=receipt, **components)
