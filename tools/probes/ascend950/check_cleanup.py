#!/usr/bin/env python3
"""Exercise deletion boundaries using only a tiny, retained project fixture."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

repo = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location('as1_cleanup', repo / 'scripts/as1_cleanup.py')
cleanup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)
out = Path('/home/d50064638/junhuip/tilemega-as1/tmp') / ('cleanup-check-'+str(time.time_ns()))
out.mkdir()
allowed = out / 'allowed'
allowed.mkdir()
entries = []


def artifact(name, **overrides):
    p = allowed / name
    p.write_text('fixture\n')
    e = dict(path=str(p), category='candidate', keep=False, regenerable=True,
             completed=True, active_refs=[], stamp=cleanup.file_stamp(p))
    e.update(overrides)
    entries.append(e)
    return p


removable = artifact('remove.o')
retained = artifact('retained.o', keep=True)
failure = artifact('failed.cpp', category='failure')
active = artifact('active.o', active_refs=[{'pid': os.getpid()}])
unknown = artifact('unknown.o')
del entries[-1]['active_refs']
changed = artifact('changed.o')
changed.write_text('changed content\n')
first = artifact('build1.o', category='build', build_id='b1', created_at='2026-01-01')
second = artifact('build2.o', category='build', build_id='b2', created_at='2026-01-02')
outside = out / 'outside.txt'
outside.write_text('outside\n')
entries.append(dict(entries[0], path=str(outside), stamp=cleanup.file_stamp(outside)))
link = allowed / 'escape.o'
link.symlink_to(outside)
entries.append(dict(entries[0], path=str(link)))
manifest = out / 'manifest_fixture.json'
manifest.write_text(json.dumps(entries, indent=2)+'\n')
before = {p: p.read_bytes() for p in allowed.iterdir() if p.is_file()}
items = cleanup.plan(entries, [allowed], 2)
if before != {p: p.read_bytes() for p in allowed.iterdir() if p.is_file()}:
    raise AssertionError('Dry run changed files')
eligible = [i for i in items if i['action'] == 'remove']
if [i['path'] for i in eligible] != [str(removable)]:
    raise AssertionError('Unexpected deletion set: '+repr(eligible))
cleanup.apply_item(eligible[0], [allowed])
if removable.exists() or not all(p.exists() for p in
        [retained, failure, active, unknown, changed, first, second, outside, link]):
    raise AssertionError('Incorrect apply boundary')
result = {'test': 'T-15', 'status': 'passed', 'evidence': 'verified',
          'registered_cases': len(entries), 'deleted_files': 1,
          'protected_survivors': len(entries)-1, 'fixture': str(out)}
(out / 'result.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps(result))
