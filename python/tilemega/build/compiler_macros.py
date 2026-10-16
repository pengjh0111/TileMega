# SPDX-License-Identifier: BSD-3-Clause
"""Capture complete macro tables from nvcc's actual host/device preprocessors."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preprocessor_invocations(text):
    rows = []
    for line in text.splitlines():
        if not line.startswith('#$ '):
            continue
        command = shlex.split(line[3:])
        if '-E' not in command:
            continue
        compiler = Path(command[0]).name
        if not re.fullmatch(r'(?:[\w.]+-)*(?:gcc|g\+\+|clang|clang\+\+|cc|c\+\+)', compiler):
            raise ValueError('unknown nvcc preprocessor executable')
        if any(token in (';', '&&', '||', '|', '>', '>>', '<', '&') or
               '$' in token or '`' in token for token in command):
            raise ValueError('nvcc preprocessor command contains unresolved shell syntax')
        if command.count('-o') != 1 or command.index('-o') + 1 >= len(command):
            raise ValueError('nvcc preprocessor lacks one explicit output')
        source = [token for token in command[1:]
                  if Path(token).suffix in ('.cu', '.c', '.cc', '.cpp', '.cxx')]
        if len(source) != 1:
            raise ValueError('nvcc preprocessor must name one translation unit')
        definitions = []
        for index, token in enumerate(command):
            if token == '-D':
                if index + 1 == len(command):
                    raise ValueError('incomplete preprocessor definition')
                definitions.append(command[index + 1])
            elif token.startswith('-D'):
                definitions.append(token[2:])
        arch = [value.partition('=')[2] for value in definitions
                if value.startswith('__CUDA_ARCH__=')]
        if len(arch) > 1 or (arch and not arch[0].isdigit()):
            raise ValueError('ambiguous device preprocessor architecture')
        phase = 'device_' + arch[0] if arch else 'host'
        rows.append(dict(source=source[0], phase=phase, command=command))
    if not rows or len({(row['source'], row['phase']) for row in rows}) != len(rows):
        raise ValueError('missing or duplicate nvcc preprocessing phases')
    for source in {row['source'] for row in rows}:
        phases = {row['phase'] for row in rows if row['source'] == source}
        if 'host' not in phases or not any(phase.startswith('device_') for phase in phases):
            raise ValueError('translation unit lacks host or device preprocessing')
    return rows


def macro_definitions(text):
    definitions = {}
    pattern = re.compile(r'^#define\s+([A-Za-z_]\w*(?:\([^)]*\))?)(?:\s+(.*))?$')
    for line in text.splitlines():
        match = pattern.fullmatch(line)
        if not match or match[1] in definitions:
            raise ValueError('malformed or duplicate preprocessor macro definition')
        definitions[match[1]] = match[2] or ''
    if not definitions:
        raise ValueError('empty complete macro table')
    return definitions


def capture_cucc(command, out):
    """MACA cu-bridge: cucc has no nvcc --dryrun and rewrites its arguments
    into one `mxcc -x maca ...` invocation (its own include paths, -imacros,
    --include cuda_runtime.h, offload arch and device __CUDA_ARCH__). With
    WCUDA_DEBUG=1 it prints that invocation; it does not depend on the source
    text, so it is taken from compiling an empty unit with the build's own
    flags. The real preprocessor then runs twice on the generated unit with
    -E -dM and mxcc's -maca-{host,device}-only."""
    import os
    sources = [token for token in command[1:] if Path(token).suffix == '.cu']
    if len(sources) != 1:
        raise ValueError('cucc capture expects one generated .cu translation unit')
    source = sources[0]
    dummy = out / 'empty.cu'
    dummy.write_text('')
    # Keep the generated unit only; host support files (`-x cu <file>`) and
    # the link step do not reach its preprocessor.
    flags, skip = [], False
    for token in command:
        if skip:
            skip = False
            continue
        if token == '-o':
            skip = True
            continue
        if token == '-shared' or token.startswith('-l') or token.startswith('-L'):
            continue
        if not token.startswith('-') and Path(token).suffix in ('.cpp', '.cc', '.cxx', '.c'):
            if flags[-2:] == ['-x', 'cu']:
                del flags[-2:]
            continue
        flags.append(str(dummy) if token == source else token)
    probe = subprocess.run(flags + ['-c', '-o', str(out / 'empty.o')], stdout=subprocess.PIPE,
                           stderr=subprocess.STDOUT, text=True, env=dict(os.environ, WCUDA_DEBUG='1'))
    lines = [line.strip()[2:-2].strip() for line in probe.stdout.splitlines()
             if line.strip().startswith('/*') and '/mxcc ' in line]
    if probe.returncode or not lines:
        raise RuntimeError('cucc did not report its mxcc invocation: ' + probe.stdout[-2000:])
    (out / 'cucc-debug.log').write_text(probe.stdout)
    mxcc = shlex.split(os.path.expandvars(lines[-1]))
    base, skip = [], False
    for token in mxcc:
        if skip:
            skip = False
            continue
        if token == '-o':
            skip = True
            continue
        if token == '-c':
            continue
        base.append(source if token == str(dummy) else token)
    if source not in base:
        raise ValueError('mxcc invocation does not name the translation unit')
    rows = []
    for index, side in enumerate(('host', 'device')):
        table = out / f'{index:02d}-{side}.macros.txt'
        run_flags = base + ['-E', '-dM', f'-maca-{side}-only', '-o', str(table)]
        run = subprocess.run(run_flags, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if run.returncode or not table.is_file():
            raise RuntimeError('mxcc preprocessing failed: ' + run.stderr[-2000:])
        macros = macro_definitions(table.read_text())
        arch = macros.get('__MACA_ARCH__')
        if (side == 'device') != (arch is not None):
            raise ValueError('mxcc side preprocessing did not select the expected side')
        compiler = run_flags[0]
        rows.append(dict(source=source, phase='host' if side == 'host' else 'device_' + arch,
                         command=run_flags, capture_command=run_flags, macros_file=str(table),
                         macros_sha256=sha(table), macros=macros, preprocessor_path=compiler,
                         preprocessor_sha256=sha(compiler), source_sha256=sha(source)))
    record = dict(schema='tilemega.dm1.macros.v1', evidence='verified', compiler_command=command,
                  compiler_sha256=sha(command[0]), dryrun_sha256=sha(out / 'cucc-debug.log'),
                  translation_units=rows)
    (out / 'capture.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def capture(command, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    if Path(command[0]).name == 'cucc':
        return capture_cucc(command, out)
    dry = subprocess.run([command[0], '--dryrun', *command[1:]],
                         check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True).stdout
    (out / 'nvcc-dryrun.log').write_text(dry)
    rows = preprocessor_invocations(dry)
    for index, row in enumerate(rows):
        compiler = shutil.which(row['command'][0])
        if compiler is None:
            raise FileNotFoundError(row['command'][0])
        flags = list(row['command'])
        position = flags.index('-o')
        flags[position + 1] = str(out / f'{index:02d}-{row["phase"]}.macros.txt')
        flags.append('-dM')
        # Execute argument vectors only. None of nvcc's shell assignments or
        # compilation/link commands are executed by this capture.
        subprocess.run(flags, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        table = Path(flags[position + 1])
        row.update(capture_command=flags, macros_file=str(table),
                   macros_sha256=sha(table), macros=macro_definitions(table.read_text()),
                   preprocessor_path=compiler, preprocessor_sha256=sha(compiler),
                   source_sha256=sha(row['source']))
    record = dict(schema='tilemega.dm1.macros.v1', evidence='verified',
                  compiler_command=command, compiler_sha256=sha(command[0]),
                  dryrun_sha256=sha(out / 'nvcc-dryrun.log'), translation_units=rows)
    (out / 'capture.json').write_text(json.dumps(record, indent=2) + '\n')
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--command-file', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    record = capture(shlex.split(args.command_file.read_text()), args.out)
    print(json.dumps(dict(event='macro_capture_complete',
                          phases=len(record['translation_units']),
                          definitions=sum(len(row['macros']) for row in record['translation_units']))))


if __name__ == '__main__':
    main()
