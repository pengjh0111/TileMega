"""Read the latest formal baseline, including explicit unavailable outcomes."""
import csv
import hashlib
import json
from pathlib import Path


def checked(value):
    path = Path(value['path'])
    if hashlib.sha256(path.read_bytes()).hexdigest() != value['sha256']:
        raise ValueError('baseline evidence changed: ' + str(path))
    return path


def latest_baselines(path):
    result = {}
    for row in csv.DictReader(Path(path).open()):
        if row['record_type'] == 'final_two_fresh_processes':
            if row['status'] not in ('pass', 'N/A'):
                raise ValueError('baseline outcome is neither qualified nor explicitly unavailable')
            result[int(row['tp']), int(row['batch']), row['system']] = row
    return result


def check_group(row, expected):
    samples = [json.loads(checked(x['metrics']).read_text()) for x in json.loads(row['measurements_json'])]
    failures = [json.loads(checked(x['guard']).read_text()) for x in json.loads(row['errors_json'])]
    if not samples and not failures:
        raise ValueError('baseline has no measured or unavailable provenance')
    if any(list(map(int, x['gpu_indices'])) != expected for x in samples + failures):
        raise ValueError('TP4 final and baseline used different physical GPU groups')
