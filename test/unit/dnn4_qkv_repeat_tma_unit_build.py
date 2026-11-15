"""Build the original FP32 QKV oracle for both current B8 store choices."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--reference', type=Path, required=True)
parser.add_argument('--candidate', type=Path, required=True)
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES') == ''
args.out.mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
profiles = [dict(re.findall(r'^#define (TILEMEGA_F3_\w+) (\d+)\s*$',
                          Path(str(p) + '.cu').read_text(), re.M))
            for p in (args.reference, args.candidate)]
oracle = ROOT / 'test/unit/dnn4_store_bert_test.cu'
base = oracle.read_text().split('int main(int,char**) try {', 1)[0]
calls = [f'DenseCase<{m},2304,768,128,192,1,0,{parts},64,Slots,Store>();'
         for parts in (12, 1) for m in (1024, 1025)]
source = args.out / 'body.cu'
source.write_text(base + 'int main(int,char**) try {\n std::cout<<std::unitbuf;\n'
    ' constexpr int Slots=DNN4_QKV_ARM?4:5,Store=DNN4_QKV_ARM?1:0;\n' +
    '\n'.join(' ' + call for call in calls) +
    '\n std::cout<<"output_hash="<<std::hex<<hash<<"\\n";return 0;\n'
    '}catch(std::exception const&e){std::cerr<<e.what()<<"\\n";return 1;}\n')
rows = []
for flag, profile in enumerate(profiles):
    folder = args.out / f'body{flag}'
    folder.mkdir()
    binary = folder / 'dnn4_f3_test'
    command = ['taskset', '-c', '136-143', 'nvcc', '-std=c++17', '-O3',
        '-gencode', 'arch=compute_90a,code=sm_90a', '--expt-relaxed-constexpr',
        '-Xptxas', '-v', '-I' + str(ROOT / 'include'),
        '-I' + str(ROOT / 'third_party/cutlass/include'), '-DDNN4_FIXED_GRID_TEST',
        f'-DDNN4_QKV_ARM={flag}', *[f'-D{k}={v}' for k, v in sorted(profile.items())],
        str(source), '-o', str(binary)]
    with (folder / 'ptxas.log').open('w') as log:
        done = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    spills = list(map(int, re.findall(r'(\d+) bytes spill (?:stores|loads)',
                                     (folder / 'ptxas.log').read_text())))
    assert done.returncode == 0 and spills and not any(spills)
    inputs = {str(p): sha(p) for p in (ROOT / 'include').rglob('*') if p.is_file()}
    inputs.update({str(source): sha(source), str(oracle): sha(oracle)})
    build = folder / 'build.json'
    build.write_text(json.dumps(dict(binary=str(binary), binary_sha256=sha(binary),
        profile=profile, command=command, inputs=inputs, spill_bytes=0), indent=2) + '\n')
    rows.append(dict(flag=flag, build=str(build)))
(args.out / 'result.json').write_text(json.dumps(dict(passed=True, gpu_executed=False,
    rows=rows, calls=calls, oracle=str(oracle), oracle_sha256=sha(oracle)), indent=2) + '\n')
print(json.dumps(dict(passed=True, arms=2, cases=4)), flush=True)
