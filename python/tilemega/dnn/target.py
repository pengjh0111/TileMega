"""Resolve explicitly configured archived solver rates without new measurements."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from tilemega.cache import atomic_json, file_sha
from tilemega.fingerprint import ROOT


def resolve_target(config, directory):
    def locate(value):
        path = Path(value).expanduser()
        return (path if path.is_absolute() else ROOT / path).resolve()

    source = locate(config['target'])
    reference = config.get('solver', {}).get('calibration_reference')
    if not reference:
        return source
    reference = locate(reference)
    target = json.loads(source.read_text())
    archived = json.loads(reference.read_text())
    for field in ('arch_tag', 'sm_major', 'sm_minor'):
        if target.get(field) != archived.get(field):
            raise ValueError('calibration reference architecture differs: ' + field)
    borrowed = []

    def missing(destination, origin, key, label):
        if key not in destination and key in origin:
            destination[key] = copy.deepcopy(origin[key])
            borrowed.append(label + key)

    missing(target, archived, 'serving_hop', '')
    for dtype in ('bf16', 'f32'):
        events = target.setdefault('event_calibration_by_dtype', {}).setdefault(dtype, {})
        old_events = archived.get('event_calibration_by_dtype', {}).get(dtype, {})
        # The per-task pair and its raw-data provenance form one contract.
        fields = ('task_publication', 'task_wait', 'task_source', 'task_source_sha256')
        present = [field in events for field in fields]
        if any(present) and not all(present):
            raise ValueError('incomplete target per-task event calibration: ' + dtype)
        if not any(present):
            for field in fields:
                missing(events, old_events, field, 'event_calibration_by_dtype.' + dtype + '.')
        calibration = target.setdefault('calibration_by_dtype', {}).setdefault(dtype, {})
        old_calibration = archived.get('calibration_by_dtype', {}).get(dtype, {})
        for field in ('task_body', 'inflight_curve', 'cta_stream_curve'):
            missing(calibration, old_calibration, field, 'calibration_by_dtype.' + dtype + '.')
    if not borrowed:
        return source
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    resolved = directory / 'resolved_target.json'
    receipt = directory / 'resolved_target.provenance.json'
    if resolved.exists() or receipt.exists():
        raise FileExistsError('refuse to overwrite resolved target provenance')
    provenance = dict(scope='inferred transfer of archived rates; no new device calibration',
        target=dict(path=str(source), sha256=file_sha(source)),
        reference=dict(path=str(reference), sha256=file_sha(reference)), fields=borrowed)
    target['dm_calibration_reference'] = provenance
    atomic_json(resolved, target)
    atomic_json(receipt, dict(**provenance, resolved_sha256=file_sha(resolved)))
    return resolved
