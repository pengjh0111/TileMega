#!/usr/bin/env python3
"""Forward selection protocol; measurements are supplied by guarded queue jobs."""
import math
import statistics
import time


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
