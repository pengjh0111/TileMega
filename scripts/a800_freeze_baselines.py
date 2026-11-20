"""Publish a content-addressed snapshot once all A800 baseline cells resolve."""
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
from pathlib import Path
import tempfile

from a100_baselines import BATCHES, ENGINES, MODELS

TERMINAL = {'measured', 'N/A', 'OOM'}


def freeze(table, directory):
    # One published directory is the campaign's frozen reference. A later
    # collector invocation can never replace it.
    if directory.exists():
        meta = json.loads((directory/'manifest.json').read_text())
        if hashlib.sha256((directory/table.name).read_bytes()).hexdigest() != meta['table_sha256']:
            raise ValueError('frozen baseline file changed')
        print(json.dumps(dict(state='frozen', table_sha256=meta['table_sha256'])))
        return
    payload = table.read_bytes()
    rows = list(csv.DictReader(io.StringIO(payload.decode())))
    expected = {(m, engine, str(b)) for m in MODELS
                for engine in (*ENGINES, 'mpk') for b in BATCHES}
    keys = {(r['model'], r['framework'], r['batch']) for r in rows}
    if keys != expected or len(rows) != len(expected):
        raise ValueError('baseline table must contain every matrix cell exactly once')
    counts = Counter(r['status'] for r in rows)
    if counts.keys() - TERMINAL:
        print(json.dumps(dict(state='collecting', counts=counts)))
        return
    if any((r['status'] == 'measured' and (not r['receipt_sha256'] or int(r['processes']) < 1))
           or (r['status'] != 'measured' and not r['evidence']) for r in rows):
        raise ValueError('resolved baseline cell lacks evidence')
    sha = hashlib.sha256(payload).hexdigest()
    with tempfile.TemporaryDirectory(dir=directory.parent) as scratch:
        snapshot = Path(scratch)/'snapshot'
        snapshot.mkdir()
        (snapshot/table.name).write_bytes(payload)
        (snapshot/'manifest.json').write_text(json.dumps(
            dict(protocol_version='a800-single-v2', table_sha256=sha, counts=counts), indent=2)+'\n')
        snapshot.rename(directory)
    print(json.dumps(dict(state='frozen', table_sha256=sha)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('table', type=Path)
    parser.add_argument('directory', type=Path)
    args = parser.parse_args()
    freeze(args.table, args.directory)
