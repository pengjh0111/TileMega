"""Scope immutable native MX-5 compile observations; never runs a GPU/query."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import shlex
from .mx5_select import check_file, record
from .mx5_plan_resources import OriginCertificate, PlanLedger, program_epoch

REQUIRED = ('pin', 'resolved_pin', 'generated_source', 'command', 'source_snapshot',
            'source_patch', 'compiler', 'producer', 'log')
# A future owner-side actual runtime query must return this binary-bound object.
# Declared residency and successful compilation are not occupancy observations.
OCCUPANCY_QUERY_CONTRACT = dict(kind='mx5_plan_occupancy', fields=(
    'binary_sha256', 'threads', 'query_status', 'resident_ctas'), execution='actual_runtime_query_only')


def load_native(path):
    native_record = record(path)
    value = json.loads(check_file(native_record).read_text())
    if value.get('schema_version') != 1 or value.get('kind') != 'mx5_native_compile_receipt':
        raise ValueError('expected native compile receipt schema1')
    if value.get('changed_inputs') != []:
        raise ValueError('native compiler inputs changed during compilation')
    code, status = value.get('returncode'), value.get('wait_status')
    if type(code) is not int or not 0 <= code <= 255 or type(status) is not int or status < 0:
        raise ValueError('native receipt lacks normalized actual compiler status')
    # waitpid result must agree with the explicit normalization used by native.
    import os
    expected = os.waitstatus_to_exitcode(status)
    expected = expected if expected >= 0 else 128 - expected
    if expected != code:
        raise ValueError('native wait status differs from device compiler returncode')
    files = value.get('files', {})
    for name in REQUIRED:
        if name not in files:
            raise ValueError('native receipt lacks required file: ' + name)
        check_file(files[name])
        if value.get(name + '_sha256') != files[name]['sha256']:
            raise ValueError('native receipt hash mismatch: ' + name)
    if files.get('binary') is not None:
        check_file(files['binary'])
    binary_sha = files['binary']['sha256'] if files.get('binary') else None
    if value.get('binary_sha256') != binary_sha:
        raise ValueError('native binary hash mismatch')
    if value.get('same_compile_mir') != files['log']['sha256']:
        raise ValueError('native receipt lacks same-compile MIR binding')
    argv = shlex.split(check_file(files['command']).read_text())
    if value.get('argv') != argv or not argv or Path(argv[0]).resolve() != check_file(files['compiler']):
        raise ValueError('native actual argv differs from recorded command/compiler')
    if '-o' not in argv:
        raise ValueError('native actual command lacks output')
    output = Path(argv[argv.index('-o') + 1]).resolve()
    if files.get('binary') and check_file(files['binary']) != output:
        raise ValueError('native binary differs from actual command output')
    if str(check_file(files['generated_source'])) not in argv:
        raise ValueError('native command does not bind generated source')
    return native_record, value


def assemble(native_path, ledger, *, stage_sources, resources=None, occupancy=None):
    """Bind one real command to the ledger's next ordinal without authorizing it."""
    ledger.origin.verify()
    if ledger.terminal_error:
        raise ValueError('plan ledger is blocked: ' + ledger.terminal_error)
    if len(ledger.attempts) != len(ledger.authorizations):
        raise ValueError('another retry authorization is required before assembling a further attempt')
    native_record, native = load_native(native_path)
    files = native['files']
    pin = json.loads(check_file(files['pin']).read_text())
    if pin != ledger.pin.data:
        raise ValueError('native requested pin differs from current authorized effective pin')
    if files['producer']['sha256'] != ledger.origin.data['preview']['producer']['sha256']:
        raise ValueError('native producer differs from immutable origin; regenerate origin')
    if not stage_sources:
        raise ValueError('missing source-qualified stage archive inventory')
    for source in stage_sources:
        check_file(source['archive'])
    snapshot = json.loads(check_file(files['source_snapshot']).read_text())
    if files['source_patch']['sha256'] != snapshot.get('worktree_diff_sha256'):
        raise ValueError('native snapshot patch differs from compiler source patch')
    if snapshot.get('compiler_sha256') != files['producer']['sha256']:
        raise ValueError('native snapshot producer mismatch')
    result = dict(schema_version=1, kind='mx5_plan_attempt', origin_sha256=ledger.origin.sha256,
                  candidate_id=ledger.candidate_id, ordinal=len(ledger.attempts),
                  **{name: files[name] for name in REQUIRED}, binary=files.get('binary'),
                  compile_receipt=native_record, resources=record(resources) if resources else None,
                  occupancy=record(occupancy) if occupancy else None, stage_sources=stage_sources)
    if native['returncode'] == 0 and (result['binary'] is None or result['resources'] is None):
        raise ValueError('successful device compile lacks binary/resource metadata; resource report failure stops')
    # append() independently re-inspects ELF/MIR, source archives and all plan pins.
    # With missing actual occupancy, a fitting plan remains explicitly blocked.
    result['program_source_epoch'] = program_epoch(result)
    return result


def write_exclusive(path, value):
    with Path(path).open('x') as out:
        json.dump(value, out, indent=2, allow_nan=False); out.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--native', required=True)
    parser.add_argument('--preview', required=True)
    parser.add_argument('--candidate', required=True, type=int)
    parser.add_argument('--ledger', help='saved existing ledger; omit only for ordinal0')
    parser.add_argument('--stage-sources', required=True)
    parser.add_argument('--resources')
    parser.add_argument('--occupancy', help='actual query receipt only; never synthesized here')
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    origin = OriginCertificate.capture(args.preview)
    ledger = PlanLedger.restore(args.ledger) if args.ledger else PlanLedger.start(origin, args.candidate)
    if ledger.origin.sha256 != origin.sha256:
        raise ValueError('saved ledger differs from requested immutable origin')
    if ledger.candidate_id != args.candidate:
        raise ValueError('candidate differs from saved ledger')
    value = assemble(args.native, ledger, stage_sources=json.loads(Path(args.stage_sources).read_text()),
                     resources=args.resources, occupancy=args.occupancy)
    write_exclusive(args.out, value)


if __name__ == '__main__':
    main()
