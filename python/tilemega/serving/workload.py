"""Validated request shapes and the past trajectory used for selection."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Workload:
    isl: int = 64
    osl: int = 1024
    batches: tuple[int, ...] = (1, 16)
    capacity: int | None = None

    def __post_init__(self):
        if any(type(value) is not int or value < 1 for value in (self.isl, self.osl)):
            raise ValueError('ISL and OSL must be positive integers')
        if (not self.batches or len(set(self.batches)) != len(self.batches) or
                any(type(value) is not int or not 1 <= value <= 16 for value in self.batches)):
            raise ValueError('batches must be distinct integers in [1,16]')
        capacity = self.isl+self.osl if self.capacity is None else self.capacity
        if type(capacity) is not int or capacity < self.isl+self.osl:
            raise ValueError('capacity must cover ISL + OSL')
        object.__setattr__(self, 'capacity', capacity)
        object.__setattr__(self, 'batches', tuple(self.batches))

    @property
    def pasts(self) -> tuple[int, ...]:
        # Preserve the historical experiment exactly. Other requests sample
        # the start, midpoint and the token just before the final endpoint.
        if (self.isl, self.osl) == (64, 1024):
            return (64, 575, 1000)
        return tuple(sorted({self.isl, self.isl+(self.osl-1)//2,
                             self.isl+max(0, self.osl-2)}))

    @property
    def objective(self) -> str:
        return f'uniform_{self.isl}_{self.isl+self.osl-1}'


def workload_points(section: dict) -> tuple[Workload, ...]:
    points = section.get('points', [[section.get('prompt_len', 64),
                                    section.get('max_new_tokens', 1024)]])
    if not isinstance(points, list) or not points:
        raise ValueError('workload.points must be a nonempty list of [ISL, OSL]')
    batches = section.get('batch', [1, 16])
    if not isinstance(batches, (list, tuple)):
        raise ValueError('workload.batch must be a list')
    result = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError('each workload point must contain ISL and OSL')
        result.append(Workload(*point, tuple(batches), section.get('capacity')))
    if len({(item.isl, item.osl) for item in result}) != len(result):
        raise ValueError('duplicate workload points')
    return tuple(result)


def validate_prompt_ids(ids, batch=None, prompt_len=None) -> int:
    if not isinstance(ids, list) or not ids or not isinstance(ids[0], list) or not ids[0]:
        raise ValueError('prompts must be a nonempty rectangular token table')
    length = len(ids[0])
    if (any(not isinstance(row, list) or len(row) != length or
            any(type(token) is not int or token < 0 for token in row) for row in ids) or
            (batch is not None and (batch < 1 or len(ids) < batch)) or
            (prompt_len is not None and length != prompt_len)):
        raise ValueError('prompt table does not cover the requested batch and ISL')
    return length
