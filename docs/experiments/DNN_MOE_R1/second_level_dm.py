#!/usr/bin/env python3
"""Forward selection protocol; measurements are supplied by guarded queue jobs."""
import math
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from identity_dm import sha, verify
from gpu_guard import stop


class BudgetExhausted(RuntimeError):
    pass


def _identity(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def family_shortlist(candidates, top_m):
    if not isinstance(top_m, int) or isinstance(top_m, bool) or top_m < 1:
        raise ValueError('top-M must be a positive integer per structural family')
    families, seen = {}, set()
    for row in candidates:
        key = row['identity_id']
        if not _identity(key) or key in seen:
            raise ValueError('candidate identities must be unique SHA256 values')
        seen.add(key)
        if row.get('role') != 'solver_candidate':
            raise ValueError('comparison arms cannot enter solver selection')
        if row.get('correctness_passed') is not True or row.get('correctness_identity') != key:
            raise ValueError('candidate needs correctness evidence for its actual binary')
        score = row['model_ns']
        if isinstance(score, bool) or not math.isfinite(score) or score <= 0:
            raise ValueError('candidate model score must be positive and finite')
        family = row['family']
        if not isinstance(family, str) or not family:
            raise ValueError('candidate needs a structural family')
        families.setdefault(family, []).append(dict(row))
    if not families:
        raise ValueError('no legal solver candidates')
    result = []
    for family in sorted(families):
        result.extend(sorted(families[family], key=lambda r: (r['model_ns'], r['identity_id']))[:top_m])
    return result


class ForwardRace:
    """Half elimination followed by three alternating fresh-process rounds.

    The orchestration caller starts the monotonic budget before builds and
    supplies that start here. Expiry never produces a partial-data winner.
    Each accepted record joins the guarded job to its exact binary identity.
    """
    def __init__(self, candidates, top_m, budget_s, *, started_at=None, now=time.monotonic):
        if isinstance(budget_s, bool) or not math.isfinite(budget_s) or budget_s <= 0:
            raise ValueError('selection budget must be positive and finite')
        self.now = now
        self.start = now() if started_at is None else started_at
        if not math.isfinite(self.start) or self.start > now():
            raise ValueError('selection start must precede the current monotonic time')
        self.deadline = self.start + budget_s
        self.candidates = {r['identity_id']: r for r in family_shortlist(candidates, top_m)}
        self.active = sorted(self.candidates)
        self.initial_ids = list(self.active)
        self.phase = 'elimination' if len(self.active) > 3 else 'final'
        self.round = 0
        self.records, self.eliminations, self.processes = [], [], set()
        self.status, self.winner = 'running', None
        self._order()

    def _order(self):
        offset = self.round % len(self.active)
        self.pending = self.active[offset:] + self.active[:offset]

    def _budget(self):
        if self.status == 'budget_exhausted' or self.now() >= self.deadline:
            self.status, self.winner = 'budget_exhausted', None
            raise BudgetExhausted('build/measure budget expired before complete selection')

    def next_request(self):
        if self.status == 'complete':
            return None
        self._budget()
        return dict(phase=self.phase, round=self.round, identity_id=self.pending[0],
                    candidate=self.candidates[self.pending[0]])

    def record(self, row):
        request = self.next_request()
        if request is None:
            raise ValueError('selection already complete')
        if any(row.get(key) != request[key] for key in ('phase', 'round', 'identity_id')):
            raise ValueError('measurement does not match the next alternating request')
        if row.get('guarded') is not True or row.get('correctness_passed') is not True:
            raise ValueError('measurement requires guarded execution and correctness')
        process = row.get('process_id')
        if not isinstance(process, str) or not process or process in self.processes:
            raise ValueError('each measurement requires a distinct process identity')
        value = row['median_ns']
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('measurement median must be positive and finite')
        self.processes.add(process)
        self.records.append(dict(row))
        self.pending.pop(0)
        if self.pending:
            return
        records = [r for r in self.records if r['phase'] == self.phase and r['round'] == self.round]
        if self.phase == 'elimination':
            scores = {r['identity_id']: r['median_ns'] for r in records}
            ranked = sorted(self.active, key=lambda key: (scores[key], key))
            keep = max(3, (len(ranked)+1)//2)
            eliminated = ranked[keep:]
            self.eliminations.append(dict(round=self.round, eliminated=eliminated, scores=scores))
            self.active = sorted(ranked[:keep])
            if len(self.active) <= 3:
                self.phase, self.round = 'final', 0
            else:
                self.round += 1
        elif self.round == 2:
            scores = {key: statistics.median(r['median_ns'] for r in self.records
                if r['phase'] == 'final' and r['identity_id'] == key) for key in self.active}
            self.winner = min(self.active, key=lambda key: (scores[key], key))
            self.status = 'complete'
            return
        else:
            self.round += 1
        self._order()

    def result(self):
        if self.status != 'complete':
            raise ValueError('incomplete selection cannot seal a winner')
        return dict(source='solver_second_level', phase='forward', winner=self.winner,
                    candidates=[self.candidates[key] for key in self.initial_ids],
                    measurements=list(self.records), eliminations=list(self.eliminations),
                    finalists=list(self.active), budget_elapsed_s=self.now()-self.start)


def run_forward_race(job):
    """Consume built plans and run each adaptive request through the scheduler.

    The build caller supplies the original monotonic start, so waiting and
    building consume the same budget as measurements. This entry point never
    accepts external timing records or invokes a timer outside gpu_guard.
    """
    if job.get('phase') != 'forward':
        raise ValueError('this selector requires a forward objective')
    here = Path(__file__).resolve().parent
    root, out = Path(job['root']), Path(job['out'])
    out.mkdir(parents=True, exist_ok=False)
    candidates = []
    for candidate in job['candidates']:
        row = dict(candidate)
        so = Path(row['so'])
        identity = verify(so)
        key = identity['artifact_id']
        gate_path = Path(row['correctness_result'])
        gate = json.loads(gate_path.read_text())
        if gate.get('passed') is not True or gate.get('identity_id') != key:
            raise ValueError('forward candidate lacks its exact-binary numerical gate')
        if row['identity_id'] != key or identity['execution']['phase'] != 'forward':
            raise ValueError('forward candidate identity/phase changed')
        row.update(correctness_passed=True, correctness_identity=key,
                   correctness_sha256=sha(gate_path), spill=identity['spill'])
        candidates.append(row)
    race = ForwardRace(candidates, job['top_m'], job['time_budget_s'],
                       started_at=job['started_at'])
    while request := race.next_request():
        key, phase, number = request['identity_id'], request['phase'], request['round']
        candidate = request['candidate']
        so = Path(candidate['so'])
        if verify(so)['artifact_id'] != key:
            raise ValueError('candidate binary changed between selection rounds')
        step = out/f'{phase}-r{number}-{key}'
        queue, events, guarded = step/'queue', step/'events', step/'guarded'
        queue.mkdir(parents=True, exist_ok=False)
        sample = step/'measurement.json'
        replacements = {'{so}': str(so), '{out}': str(sample), '{identity}': key,
                        '{phase}': phase, '{round}': str(number)}
        command = []
        for argument in candidate['measure_command']:
            for pattern, value in replacements.items():
                argument = argument.replace(pattern, value)
            command.append(argument)
        if not command or not any('{out}' in a for a in candidate['measure_command']):
            raise ValueError('candidate measure command needs a fresh output path')
        remaining = math.floor(race.deadline-time.monotonic())
        if remaining <= 0:
            race._budget()
            raise BudgetExhausted('less than one second remains in the selection budget')
        task = dict(name='candidate_measurement', command=command, cwd=str(root),
                    gpu=True, out=str(guarded), timeout_s=remaining,
                    needs_free_mib=candidate.get('needs_free_mib', 12288))
        (queue/'queue_measure.json').write_text(json.dumps([task], indent=2)+'\n')
        with (step/'observer.log').open('w') as log:
            observer = subprocess.Popen([sys.executable, str(here/'run_queue.py'),
                '--queue-dir', str(queue), '--out', str(events),
                '--policy', str(here/'guard_policy.json')],
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                code = observer.wait(timeout=max(0, race.deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                # The owned process tree includes scheduler, guard and timer;
                # PID/start checks protect other sessions during cleanup.
                stop(observer)
                race.status, race.winner = 'budget_exhausted', None
                (out/'race.json').write_text(json.dumps(dict(status=race.status,
                    records=race.records, eliminations=race.eliminations), indent=2)+'\n')
                raise BudgetExhausted('selection budget expired while awaiting its guarded job')
            if code:
                raise RuntimeError('guarded candidate queue failed; see '+str(step/'observer.log'))
        completion = json.loads((events/'completion.json').read_text())
        guard_path = guarded/'guard_result.json'
        guard = json.loads(guard_path.read_text())
        if completion.get('passed') is not True or guard != dict(code=0, reason='child exit 0'):
            raise ValueError('candidate measurement lacks a successful guard receipt')
        measured = json.loads(sample.read_text())
        if measured.get('identity_id') != key or verify(so)['artifact_id'] != key:
            raise ValueError('measured candidate identity changed')
        record = dict(phase=phase, round=number, identity_id=key,
                      median_ns=measured['median_ns'], process_id=measured['process_id'],
                      guarded=True, correctness_passed=True, sample_sha256=sha(sample),
                      guard_sha256=sha(guard_path), queue_sha256=sha(queue/'queue_measure.json'))
        race.record(record)
        (out/'race.json').write_text(json.dumps(dict(status=race.status, records=race.records,
            eliminations=race.eliminations, active=race.active), indent=2)+'\n')
    result = race.result()
    (out/'plans.json').write_text(json.dumps(result, indent=2)+'\n')
    return result
