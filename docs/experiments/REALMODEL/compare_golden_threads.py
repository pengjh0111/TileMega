#!/usr/bin/env python3
"""Read-only check of CPU BF16 golden sensitivity; never rewrites fixtures."""
import json
from pathlib import Path
import re
import torch


def main():
    here = Path(__file__).resolve().parent
    work = here/'depth_work/l16'
    fixture = work/'export/fixture'
    manifest = json.loads((fixture/'manifest.json').read_text())
    program = torch.export.load(work/'export/exported_program.pt2')
    module = program.module()
    def read(path):
        return torch.frombuffer(bytearray(path.read_bytes()), dtype=torch.bfloat16)
    inputs = []
    for spec in program.graph_signature.input_specs:
        if spec.kind.name != 'USER_INPUT': continue
        tensor = read(fixture/f'input_{spec.arg.name}.bin')
        shape = ((1, manifest['seq'], manifest['hidden']) if spec.arg.name == 'hidden' else
                 (1, manifest['kv_heads'], manifest['past'], manifest['head_dim']))
        inputs.append(tensor.reshape(shape))
    log = (here/'depth_results/l16_r0.txt').read_text()
    buffers = re.findall(r'E2E_OUTPUT_DIFF index=(\d+) buffer=(\d+)', log)
    tm = [read(work/f'dump/buffer_{buffer}.bin').float() for _, buffer in buffers]
    recorded = [read(fixture/f'reference_{index}.bin') for index, _ in buffers]
    default_threads = torch.get_num_threads()
    rows = []
    for threads in dict.fromkeys((8, default_threads)):
        torch.set_num_threads(threads)
        with torch.no_grad():
            outputs = module(*inputs)
        outputs = [tensor.flatten() for tensor in outputs]
        if len(outputs) != len(tm) or len(outputs) != len(recorded):
            raise RuntimeError('output inventory differs between golden and harness')
        if threads == 8 and any(not torch.equal(a,b) for a,b in zip(outputs,recorded)):
            raise RuntimeError('stored 8-thread golden did not reproduce bitwise')
        mismatch, maximum = 0, 0.0
        for a,b in zip(tm,outputs):
            b = b.float()
            error = (a-b).abs()
            mismatch += int((error > .016+.016*b.abs()).sum())
            maximum = max(maximum, float(error.max()))
        rows.append({'threads': threads, 'mismatch_vs_tilemega': mismatch, 'max_abs': maximum,
                     'changed_from_stored': sum(int((a != b).sum()) for a,b in zip(outputs,recorded))})
    if rows[0]['mismatch_vs_tilemega'] != int(re.search(r'l05_vs_l0_mismatch=(\d+)',log)[1]):
        raise RuntimeError('diagnostic comparison differs from the unchanged harness criterion')
    print(json.dumps({'torch_version': torch.__version__, 'default_threads': default_threads,
                      'rows': rows}, indent=2))


if __name__ == '__main__':
    main()
