#!/usr/bin/env python3
"""Independent host oracle for the bounded four-direction payload litmus."""
from array import array
import argparse
import hashlib
import json
from pathlib import Path
import sys

CELL, ELEMENTS, STRIDE = 128, 4096, 4224
EPOCHS, PHASES, GLOBAL_FIELDS, PHASE_FIELDS = 8, 12, 6, 8
GUARD = 0xA5A5A5A5


def read_words(path, count):
    if sys.byteorder != 'little' or path.stat().st_size != count * 4:
        raise ValueError('Wrong endian or evidence extent: ' + str(path))
    value = array('I')
    with path.open('rb') as stream:
        value.fromfile(stream, count)
    return value


def expected(worker, epoch, size):
    return array('f', (64 * (worker + 1) + 8 * epoch + i % 7 for i in range(size)))


def inputs(direction, groups, worker):
    aic = worker < groups
    group = worker if aic else (worker - groups) // 2
    previous = (group - 1) % groups
    if direction == 0:
        return [previous] if aic else []
    if direction == 1:
        return [] if aic else [previous]
    if direction == 2:
        return [groups + 2 * previous, groups + 2 * previous + 1] if aic else []
    return [] if aic else [groups + 2 * previous + 1 - (worker - groups) % 2]


def audit(directory):
    result = json.loads((directory / 'result.json').read_text())
    groups, workers = result['groups'], result['workers']
    if groups not in (8, 14) or workers != 3 * groups:
        raise ValueError('Invalid bounded topology')
    fields = GLOBAL_FIELDS + PHASES * PHASE_FIELDS
    controls = read_words(directory / 'controls.bin', fields * workers * CELL + CELL)
    payload = read_words(directory / 'payload.bin', PHASES * workers * STRIDE)
    consumed = read_words(directory / 'consumer.bin', PHASES * workers * 2 * STRIDE)
    field = lambda f, w: controls[(f * workers + w) * CELL]
    for f in range(fields):
        for w in range(workers):
            at = (f * workers + w) * CELL
            if any(controls[at + 1:at + CELL]):
                raise ValueError('Control cell guard was overwritten')
    if any(x != GUARD for x in controls[-CELL:]):
        raise ValueError('Control tail was overwritten')
    fault = result['injected_abort_worker']
    if fault is not None and not 0 <= fault < workers:
        raise ValueError('Invalid injected actor')
    for w in range(workers):
        if field(5, w) != w:
            raise ValueError('Missing/aliased worker identity')
        if fault is None:
            if field(1, w) or field(0, w) != 2 * PHASES:
                raise ValueError('Missing final rendezvous or actor abort')
        elif field(1, w) != (4 if w == fault else 2):
            raise ValueError('Injected abort did not propagate through each actor')
    checked_elements = 0
    for phase in range(PHASES):
        size = (32, 1056, ELEMENTS)[phase // 4]
        direction = phase % 4
        for w in range(workers):
            aic = w < groups
            producer = (direction in (0, 1) and aic) or (direction in (2, 3) and not aic)
            predecessors = inputs(direction, groups, w)
            at = (phase * workers + w) * STRIDE
            for storage, start in ((payload, at), (consumed, at * 2), (consumed, (at // STRIDE * 2 + 1) * STRIDE)):
                if any(x != GUARD for x in storage[start + size:start + STRIDE]):
                    raise ValueError('Payload/consumer tail guard was overwritten')
            if fault is not None:
                continue
            control = lambda f: field(GLOBAL_FIELDS + phase * PHASE_FIELDS + f, w)
            want = (EPOCHS if producer else 0, EPOCHS if predecessors else 0,
                    EPOCHS if predecessors else 0, EPOCHS, 0, 0, 0)
            if tuple(control(f) for f in range(7)) != want:
                raise ValueError(f'Wrong phase protocol counters: phase={phase} worker={w}')
            old_sum = sum(sum(expected(p, EPOCHS - 1, size)) for p in predecessors)
            if control(7) != old_sum:
                raise ValueError('Old-payload pre-read was not observed before final overwrite')
            want_payload = array('I')
            if producer:
                want_payload.frombytes(expected(w, EPOCHS, size).tobytes())
            else:
                want_payload.extend([0] * size)
            if payload[at:at + size] != want_payload:
                raise ValueError(f'Final producer payload differs: phase={phase} worker={w}')
            checked_elements += size
            for j in range(2):
                start = ((phase * workers + w) * 2 + j) * STRIDE
                want_consumer = array('I')
                if j < len(predecessors):
                    want_consumer.frombytes(expected(predecessors[j], EPOCHS, size).tobytes())
                else:
                    want_consumer.extend([GUARD] * size)
                if consumed[start:start + size] != want_consumer:
                    raise ValueError(f'Consumer DMA output differs: phase={phase} worker={w} input={j}')
                checked_elements += size
    error_fields = ('self_timeouts', 'pre_read_errors', 'scalar_errors', 'dma_errors',
                    'guard_errors', 'mapping_errors')
    if any(result[k] for k in error_fields):
        raise ValueError('Device reports an error')
    if result['status'] != ('expected_abort' if fault is not None else 'pass'):
        raise ValueError('Device status did not match test mode')
    if fault is None and (result['aborted_actors'] or result['unfinished']):
        raise ValueError('Device did not complete all phases')
    return dict(status='pass', groups=groups, workers=workers, injected_abort_worker=fault,
                checked_final_payload_elements=checked_elements,
                all_payload_and_control_guards_pass=True,
                files={name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                       for name in ('result.json', 'controls.bin', 'payload.bin', 'consumer.bin')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(args.directory), indent=2))
