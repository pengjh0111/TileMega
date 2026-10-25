#!/usr/bin/env python3
"""CPU-check domain.cpp against production geometry for the five dense models.

Run: python experiments/mx5/check_domain.py --out runs/mx5/domain/check1
Builds and all raw case outputs stay under the new output directory. R_next=200
is a conditional formula test input, never a measured resource reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import tarfile
import time

REPO = Path(__file__).resolve().parents[2]
MODELS = ('qwen3_0_6b', 'qwen3_1_7b', 'llama3_2_1b', 'llama3_2_3b', 'llama3_1_8b')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def oracle_source():
    calls = []
    for tn in (16, 32, 64, 128):
        for wn in (1, 2, 4, 8):
            for depth in range(1, 5):
                for part in ('kTileRange', 'kStreamK') if wn == 1 else ('kTileRange',):
                    calls.append(f'  Check<{tn}, {wn}, {depth}, mx::MxPart::{part}>(head,n,k,b,threads,op,rnext);')
    return r'''#include <tilemega/Backend/Mx/DecodeGeometryV2.h>
#include <cstdio>
#include <cstdlib>
#include <string>
namespace mx = tilemega::backend::mx;
template<int TN, int WN, int D, mx::MxPart Part>
void Check(bool head,int n,int k,int b,int threads,mx::MxOpV2 op,int rnext) {
  if (head && (WN < 4 || Part != mx::MxPart::kTileRange)) return;
  if (!mx::MxGemmLegalV2(n,k,b,TN,WN,D,op,Part)) return;
  int first=mx::MxFirstChunksV2<TN,WN,Part>(n,k,104);
  std::printf("%d %d %d %d %d",TN,WN,D,int(Part),first);
  if (rnext>=0) std::printf(" %d",mx::MxPrefetchWidthV2<TN,WN,D,Part>(n,k,104,threads,rnext));
  std::puts("");
}
int main(int argc,char**argv) {
  if(argc!=8)return 2;
  bool head=std::string(argv[1])=="head";
  int n=std::atoi(argv[2]),k=std::atoi(argv[3]),b=std::atoi(argv[4]);
  int threads=std::atoi(argv[5]),rnext=std::atoi(argv[7]);
  auto op=head?mx::MxOpV2::kStore:static_cast<mx::MxOpV2>(std::atoi(argv[6]));
''' + '\n'.join(calls) + '\n}\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cxx', default='c++')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists():
        parser.error('output already exists; preserve earlier CPU evidence')
    out.mkdir(parents=True)
    oracle = out / 'oracle.cpp'
    oracle.write_text(oracle_source())
    build_rows = []
    for name, source in [('domain', REPO / 'experiments/mx5/domain.cpp'), ('oracle', oracle)]:
        binary = out / name
        command = [args.cxx, '-std=c++17', '-O2', '-Wall', '-Wextra', '-pedantic', '-MMD',
                   '-MF', str(out / (name + '.d')), '-I' + str(REPO / 'include'), str(source), '-o', str(binary)]
        start = time.monotonic()
        result = subprocess.run(command, capture_output=True, text=True)
        (out / (name + '_build.log')).write_text(result.stdout + result.stderr)
        row = dict(name=name, command=command, returncode=result.returncode, seconds=time.monotonic() - start,
                   source_sha256=digest(source))
        if binary.exists():
            row['binary_sha256'] = digest(binary)
        build_rows.append(row)
        (out / 'build_identity.json').write_text(json.dumps(build_rows, indent=2))
        if result.returncode:
            raise SystemExit(result.returncode)
    dependencies = {REPO / 'experiments/mx5/domain.cpp', REPO / 'experiments/mx5/check_domain.py', oracle}
    for path in out.glob('*.d'):
        text = path.read_text().replace('\\\n', '')
        dependencies.update(Path(value) for value in shlex.split(text.split(':', 1)[1]))
    model_inputs = []
    requests = []
    for model in MODELS:
        path = REPO / 'configs/models' / (model + '.json')
        dependencies.add(path)
        dimensions = json.loads(path.read_text())['dimensions']
        model_inputs.append(dict(path=str(path), sha256=digest(path)))
        hidden, intermediate = dimensions['hidden_size'], dimensions['intermediate_size']
        dim, heads, hkv = (dimensions[field] for field in ('head_dim', 'num_attention_heads', 'num_key_value_heads'))
        shapes = [('qkv', 'gemm', (heads + 2 * hkv) * dim, hidden, 0),
                  ('o', 'gemm', hidden, heads * dim, 1),
                  ('gate_up', 'gemm', 2 * intermediate, hidden, 2),
                  ('down', 'gemm', hidden, intermediate, 1),
                  ('head', 'head', dimensions['vocab_size'], hidden, None)]
        for batch in (1, 2, 4, 8, 16):
            for stage, kind, n, k, op in shapes:
                for threads in (512, 576):
                    for rnext in (None, 200):
                        requests.append(dict(model=model, stage=stage, kind=kind, n=n, k=k, b=batch,
                                             threads=threads, op=op, r_next=rnext))
    requests.append(dict(model='boundary', stage='explicit_zero', kind='gemm', n=2048, k=2048,
                         b=1, threads=576, op=1, r_next=0))
    failures = []
    candidates = 0
    with (out / 'cases.jsonl').open('x') as stream:
        for request in requests:
            kind, n, k, batch, threads, op, rnext = (request[field] for field in
                                                    ('kind', 'n', 'k', 'b', 'threads', 'op', 'r_next'))
            command = [str(out / 'domain'), '--kind', kind, '--n', str(n), '--k', str(k),
                       '--batch', str(batch), '--threads', str(threads)]
            if op is not None:
                command += ['--op', str(op)]
            if rnext is not None:
                command += ['--r-next', str(rnext)]
            result = subprocess.run(command, capture_output=True, text=True)
            expected_command = [str(out / 'oracle'), kind, str(n), str(k), str(batch), str(threads),
                                str(op or 0), str(-1 if rnext is None else rnext)]
            expected = subprocess.run(expected_command, capture_output=True, text=True)
            row = dict(request=request, command=command, returncode=result.returncode, stdout=result.stdout,
                       stderr=result.stderr, oracle_command=expected_command, oracle_returncode=expected.returncode,
                       oracle_stdout=expected.stdout, oracle_stderr=expected.stderr)
            try:
                actual = [json.loads(line) for line in result.stdout.splitlines()]
                columns = ('tn', 'wn', 'depth', 'part', 'first_chunks') + (() if rnext is None else ('prefetch',))
                projected = [tuple(value[column] for column in columns) for value in actual]
                wanted = [tuple(map(int, line.split())) for line in expected.stdout.splitlines()]
                assert result.returncode == expected.returncode == 0
                assert len(projected) == len(set(projected)) and sorted(projected) == sorted(wanted)
                assert actual
                for value in actual:
                    assert all(value[field] == request[field] for field in ('kind', 'n', 'k', 'b', 'threads'))
                    assert ('op' in value) == (op is not None)
                    if op is not None:
                        assert value['op'] == op
                    assert ('prefetch' in value) == ('r_next' in value) == (rnext is not None)
                    if rnext is not None:
                        assert value['r_next'] == rnext
                candidates += len(actual)
                row['passed'] = True
            except (AssertionError, ValueError, KeyError, TypeError) as error:
                row['passed'] = False
                row['error'] = repr(error)
                failures.append(request)
            stream.write(json.dumps(row) + '\n')
    valid = ['--kind', 'gemm', '--n', '2048', '--k', '2048', '--batch', '1', '--op', '0']
    bad_arguments = [[], valid + ['--threads', '1024'], valid + ['--r-next', '-1'], valid + ['--op', '1'],
                     valid[:-2], valid + ['--unknown', '1'], valid + ['--n', '99999999999999999999'],
                     ['--kind', 'head', '--n', '32768', '--k', '1024', '--batch', '1', '--op', '0']]
    rejected = []
    for arguments in bad_arguments:
        command = [str(out / 'domain'), *arguments]
        result = subprocess.run(command, capture_output=True, text=True)
        passed = result.returncode != 0 and not result.stdout
        rejected.append(dict(command=command, returncode=result.returncode, stdout=result.stdout,
                             stderr=result.stderr, passed=passed))
        if not passed:
            failures.append(dict(arguments=arguments))
    (out / 'argument_checks.json').write_text(json.dumps(rejected, indent=2))
    paths = {str(path): digest(path) for path in sorted(dependencies)}
    with tarfile.open(out / 'sources.tar.gz', 'x:gz') as archive:
        for ordinal, path in enumerate(sorted(dependencies)):
            name = str(path.relative_to(REPO)) if path.is_relative_to(REPO) else f'generated/{ordinal}_{path.name}'
            archive.add(path, arcname=name)
    summary = dict(cases=len(requests), model_batch_stage_shapes=125, candidate_rows=candidates,
                   threads=[512, 576], conditional_r_next_examples=[200], explicit_zero_presence_checked=True,
                   measured_resources=False, models=model_inputs, argument_checks=len(rejected), failures=failures,
                   build_identity='build_identity.json', sources=paths,
                   source_archive_sha256=digest(out / 'sources.tar.gz'))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps({key: summary[key] for key in ('cases', 'candidate_rows', 'argument_checks', 'failures')}))
    if failures:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
