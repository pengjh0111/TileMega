"""Resume the frozen primary audit after the recorded GPU outage."""

import collections
import concurrent.futures
import csv
import fcntl
import os
from pathlib import Path
import sys
import time

import guarded as g
import measure_population as measure
import run_primary


CASE = measure.CASE
STATE = CASE / 'phase4_failover'
AMENDMENT = CASE / 'gpu_failover.json'
V1 = Path('/root/TileMega-search-quality/runs/sm90_search_quality/qwen3_1_7b-B1-20261123')


def bind_cards(frozen, guard):
    original = {card['uuid']: card for card in frozen['gpu_cards']}
    current = g.inventory()
    v1_inventory = {}
    with (V1 / 'gpu-inventory.csv').open() as handle:
        for index, uuid, bus in csv.reader(handle, skipinitialspace=True):
            v1_inventory[uuid] = (index, bus[-12:].lower())
    v1_power = g.read(V1 / 'guard_policy.json')['idle_power_w_by_gpu']
    cards = []
    power = {}
    for card in current:
        idx, uuid = card['index'], card['uuid']
        if idx not in (0, 1) and uuid not in original:
            continue
        reference = original.get(uuid, frozen['gpu_cards'][0])
        fields = ('name', 'driver_version', 'memory.total', 'clocks.max.sm')
        if uuid in original:
            fields += ('pci.bus_id', 'numa')
        assert all(card[key] == reference[key] for key in fields), card
        guard.require_allowed_gpu({'gpu_allowlist_file': g.read(
            frozen['files']['guard_policy']['path'])['gpu_allowlist_file']}, idx, timing=True)
        old_idx, old_bus = v1_inventory[uuid]
        assert old_bus == card['pci.bus_id']
        power[str(idx)] = v1_power[old_idx]
        cards.append(card)
    assert cards
    return cards, power


def retry_allowlist(frozen, incident):
    known = {}
    for item in incident['round0_infrastructure_failure_evidence']:
        q, path = item['id'], Path(item['attempt'])
        assert path.parent == measure.MEAS / 'round0' / q
        assert g.sha(path / 'run.log') == item['log_sha256']
        assert g.read(path / 'guard/guard_result.json')['code'] == 1
        assert not any((path / f'job/measurements/{key}.json').exists() for key in ('c1', 'c2'))
        assert item['uuid'] == frozen['round_gpu_uuid'][0][q]
        known[path] = q
    return known


def main():
    import tilemega.bench.gpu_guard as guard

    frozen = measure.verify_freeze()
    frozen_hash = g.sha(CASE / 'frozen_search.json')
    incident_path = CASE / 'phase4_gpu_incident.json'
    incident = g.read(incident_path)
    assert incident['frozen_search_sha256'] == frozen_hash
    known = retry_allowlist(frozen, incident)
    cards, power = bind_cards(frozen, guard)
    policy_path = STATE / 'guard_policy.json'
    allowed = [c['index'] for c in cards]
    identities = ('index', 'uuid', 'pci.bus_id', 'driver_version', 'numa',
                  'name', 'memory.total', 'clocks.max.sm')
    physical = [{key: c[key] for key in identities} for c in cards]
    policy = dict(g.read(frozen['files']['guard_policy']['path']),
                  allowed_gpu_indices=allowed, allowed_timing_gpu_indices=allowed,
                  idle_power_w_by_gpu=power,
                  sq2_live_uuid_by_index={str(c['index']): c['uuid'] for c in cards})
    if AMENDMENT.exists():
        amendment = g.read(AMENDMENT)
        assert g.read(policy_path) == policy
        assert amendment['cards'] == physical
        assert amendment['frozen_search_sha256'] == frozen_hash
        assert amendment['guard_policy_sha256'] == g.sha(policy_path)
        assert amendment['adapter_sha256'] == g.sha(__file__)
        assert frozen['frozen_at'] < amendment['prepared_at'] < time.time()
    else:
        g.write(policy_path, policy)
        accepted_before = collections.Counter()
        for path in measure.MEAS.glob('*/*/result.json'):
            result = g.read(path)
            if result['status'] == 'valid':
                accepted_before[result['gpu_uuid']] += 1
        g.write(AMENDMENT, {
            'prepared_at': time.time(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
            'frozen_search_sha256': frozen_hash, 'incident_sha256': g.sha(incident_path),
            'adapter_sha256': g.sha(__file__), 'cards': physical,
            'guard_policy_sha256': g.sha(policy_path),
            'v1_power_policy_sha256': g.sha(V1 / 'guard_policy.json'),
            'v1_gpu_inventory_sha256': g.sha(V1 / 'gpu-inventory.csv'),
            'accepted_before_by_uuid': dict(accepted_before),
            'known_retry_attempts': {q: str(path) for path, q in known.items()},
            'reason': 'One original H800 UUID disappeared and the two survivors stayed occupied. '
                      'The user authorized devices 0 and 1 when the original cards are occupied.',
            'dispatch': 'Use the frozen shuffle and reverse as shared queues, omitting completed jobs. '
                        'Offer the next job to the first idle authorized H800 in ascending index order. '
                        'A guard rejection returns only that job to the queue after a 30-second cooldown '
                        'for that device. Placement uses availability and never reads TPOT.',
            'confirmed_change': 'Rounds and confirmation can span different H800 UUIDs. '
                                'The frozen model, set membership, binaries, timing procedure and '
                                'guard thresholds remain unchanged.',
        })

    def check_infrastructure(folder, code):
        if code != 1:
            return
        with (folder / 'run.log').open('rb') as log:
            log.seek(max(0, (folder / 'run.log').stat().st_size - 8192))
            tail = log.read()
        if (b'CUDA driver identity query failed: 999' in tail or
                b'AcceleratorError: CUDA error: unknown error' in tail):
            raise RuntimeError(f'infrastructure CUDA fault recurred, retained {folder}')

    def prior(folder):
        for path in sorted(folder.glob('attempt*')):
            record = path / 'guard/guard_result.json'
            if path not in known and record.is_file():
                result = g.read(record)
                if result['code'] != 75:
                    check_infrastructure(path, result['code'])
                    return path, result
        return None

    def attempt(command, folder, _policy, card, **kwargs):
        result = g.gpu_attempt(command, folder, policy_path, card, **kwargs)
        check_infrastructure(folder, result['code'])
        return result

    def dispatch_round(phase, index, order, _lanes, actual, spec, checks=False):
        pending = collections.deque()
        for q in order:
            folder = measure.MEAS / phase / q
            result_path = folder / 'result.json'
            if result_path.exists():
                result = g.read(result_path)
                if Path(result['attempt']) not in known:
                    assert result['round_index'] == index
                    assert result['binary_sha256'] == actual[q]['binary_sha256']
                    continue
                assert result['status'] == 'runtime_failed'
                archive = folder / 'result.pre_gpu_recovery.json'
                if archive.exists():
                    assert g.read(archive) == result
                    result_path.unlink()
                else:
                    result_path.rename(archive)
            previous = prior(folder)
            if previous:
                card = g.read(previous[0] / 'gpu_before.json')
                assert measure.attempt_process(phase, index, q, card, actual, spec, checks)
            else:
                pending.append(q)

        print(f'{phase}: {len(pending)} pending on cards {allowed}', flush=True)
        active = {}
        cooldown = {}
        last_wait = None
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(cards)) as pool:
            while pending or active:
                for idx, (future, q) in list(active.items()):
                    if future.done():
                        del active[idx]
                        if not future.result():
                            pending.append(q)
                            cooldown[idx] = time.monotonic() + 30
                wait = []
                for card in cards:
                    idx = card['index']
                    if not pending or idx in active or time.monotonic() < cooldown.get(idx, 0):
                        continue
                    try:
                        row = guard.gpu(idx)
                        ready = guard.idle(row, policy, 32768)
                    except (ValueError, OSError, RuntimeError) as exc:
                        wait.append(f'{idx} telemetry unavailable: {exc}')
                        continue
                    if not ready:
                        wait.append(f'{idx}: {row["used_mib"]:.0f} MiB used, {row["utilization_pct"]:.0f}% util')
                        continue
                    q = pending.popleft()
                    print(f'{phase} dispatch {q} to GPU {idx} ({card["uuid"]})', flush=True)
                    active[idx] = (pool.submit(measure.attempt_process, phase, index, q, card,
                                               actual, spec, checks), q)
                if pending and not active:
                    minute = int(time.monotonic() // 60)
                    if last_wait != minute:
                        print(f'{phase} waiting: {len(pending)} jobs. {"; ".join(wait)}', flush=True)
                        last_wait = minute
                if active or pending:
                    time.sleep(5)

    write_index = measure.write_attempt_index

    def attempt_index(spec):
        write_index(spec)
        path = CASE / 'timings.csv'
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            if Path(row['attempt']) in known:
                assert row['status'] == 'unfinished' and row['guard_code'] == '1'
                row['status'] = 'infrastructure_failed'
        measure.csv_dump(path, rows)

    measure.prior_attempt = prior
    measure.gpu_attempt = attempt
    measure.run_round = dispatch_round
    measure.write_attempt_index = attempt_index
    if not (CASE / 'measurement_complete.json').exists():
        measure.main()


if __name__ == '__main__':
    sys.path.insert(0, str(g.ROOT / 'python'))
    with (CASE / 'campaign.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        g.write(STATE / 'pid.json', {'pid': os.getpid(), 'started': time.time(),
                                     'script_sha256': g.sha(__file__)})
        main()
    run_primary.main()
