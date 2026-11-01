"""Rebuild all eight fixed plans and compare their CU and MIR references."""
import argparse
import concurrent.futures
import csv
import hashlib
import json
from pathlib import Path
import subprocess


parser = argparse.ArgumentParser()
parser.add_argument('--out', type=Path, required=True)
parser.add_argument('--record-type', required=True)
parser.add_argument('--resume', action='store_true')
args = parser.parse_args()
root = Path.cwd()
csv_path = root / 'docs/results/MX7_frozen_gate_c550.csv'
with csv_path.open() as stream:
    reader = csv.DictReader(stream)
    fields = reader.fieldnames
    refs = [row for row in reader if row['record_type'] == 'reference']
assert len(refs) == 8
head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def one(ref):
    name = f"{ref['model']}_B{ref['batch']}_{ref['phase']}"
    out = args.out.resolve() / name
    out.mkdir(parents=True, exist_ok=args.resume)
    so = out / 'plan.so'
    reference_dir = (root / ref['artifact']).parent
    command = json.loads((reference_dir / 'command.json').read_text())
    command[3] = str(so)
    (out / 'command.json').write_text(json.dumps(command) + '\n')
    if args.resume and so.exists():
        from tilemega.build.identity import verify
        verify(so)
        built = subprocess.CompletedProcess(command, 0)
    else:
        with (out / 'compile.log').open('w') as log:
            built = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    if built.returncode:
        return dict(ref, record_type=args.record_type, source_head=head, status='compile_failed',
                    artifact=str(so.relative_to(root)), generated_cu_sha256='', artifact_sha256='',
                    function_instructions_json='', kernel_instructions_json='', instruction_report='')
    with (out / 'mir_report.txt').open('w') as log:
        subprocess.run(['python', 'scripts/mx_mir_report.py', str(so) + '.ptxas.log',
                        '--csv', str(out / 'mir_loops.csv')], stdout=log, check=True)
    functions = [line for line in (out / 'mir_report.txt').read_text().splitlines()
                 if not line.startswith('   ') and 'instrs=' in line]
    resources = json.loads(Path(str(so) + '.resources.json').read_text())
    cu_sha = sha(str(so) + '.cu')
    status = ('cu_mismatch' if cu_sha != ref['generated_cu_sha256'] else
              'instruction_mismatch' if functions != json.loads(ref['function_instructions_json']) else 'pass')
    print(name, status, flush=True)
    return dict(ref, record_type=args.record_type, source_head=head, status=status,
                artifact=str(so.relative_to(root)), generated_cu_sha256=cu_sha,
                artifact_sha256=sha(so), function_instructions_json=json.dumps(functions),
                kernel_instructions_json=json.dumps(resources['kernel_instructions']),
                instruction_report=str((out / 'mir_report.txt').relative_to(root)))


with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
    rows = list(pool.map(one, refs))
with csv_path.open('a', newline='') as stream:
    csv.DictWriter(stream, fieldnames=fields, lineterminator='\n').writerows(rows)
raise SystemExit(0 if all(row['status'] == 'pass' for row in rows) else 2)
