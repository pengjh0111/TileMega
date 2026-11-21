#!/usr/bin/env python3
"""Build a sparse role probe for a formal fused expert body."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from collect_a800_dense_service import sha
from prepare_a800_dense_service import ROOT, HEADERS


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--anchor', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--variant', choices=('full', 'live', 'segments'), required=True)
    p.add_argument('--copy-group', type=int, choices=(0, 4, 8))
    p.add_argument('--reserve-program-shared', action='store_true')
    p.add_argument('--gu-route-bases', action='store_true')
    p.add_argument('--live-rows', action='store_true')
    a = p.parse_args()
    anchor = json.loads(a.anchor.read_text())
    library = Path(anchor['decode'])
    if sha(library) != anchor['decode_sha256'] or anchor['coordinates']['residency'] != 1:
        raise ValueError('expected the immutable single-resident fused incumbent')
    build_identity = json.loads(Path(f'{library}.identity.json').read_text())
    if build_identity['binary_sha256'] != anchor['decode_sha256']:
        raise ValueError('incumbent build manifest has a different binary')
    generated = Path(str(library)+'.cu').read_text()
    alias = next(iter(re.findall(r'\bMoeFused<([^>]+)>', generated)))
    values = alias.split(',')
    if values[-3:] != ['true', 'false', 'true']:
        raise ValueError('probe requires SM80 local routing with inline combine')
    batch, ng, nd, hidden, tng, tkg, tnd, tkd, slots, wng, wkg, wnd, wkd, u, experts = (
        int(v) for v in values[:15])
    if (u != 16 or int(values[15]) != 8 or wng*wkg != 4 or wnd*wkd != 4 or batch != anchor['batch']):
        raise ValueError('fused stage has a different weight or route contract')
    floats, page = values[16:18]
    r, softmax_all = values[-6], values[-4]
    case = f'{anchor["model"]}-b{batch}-{a.variant}'
    variants = {}
    copy = a.copy_group if a.copy_group is not None else anchor['coordinates'].get('copy_issue_group', 0)
    if copy:
        if page == '64' and a.variant != 'segments':
            raise ValueError('grouped page64 copies require the segmented implementation')
        variants['TILEMEGA_SM80_COPY_ISSUE_GROUP'] = copy
        case += f'-copy{copy}'
    if a.variant == 'segments':
        if page != '64':
            raise ValueError('segments require paged weights')
        variants['TILEMEGA_SM80_MOE_PAGE_SEGMENTS'] = 1
    else:
        macro='TILEMEGA_SM80_MOE_ALIGNED_GATE_PAGES'
        if re.search(rf'^#define {macro}\b', generated, re.MULTILINE):
            variants[macro] = 1
    if a.variant == 'live' or a.live_rows:
        variants['TILEMEGA_SM80_ACTIVE_EPILOGUE'] = 1
        if a.variant != 'live':
            case += '-live-rows'
    if a.gu_route_bases:
        if tkg != 64 or batch > 16:
            raise ValueError('hoisted Gu routes require 64-wide tiles and at most 16 tokens')
        variants['TILEMEGA_SM80_MOE_GU_ROUTE_BASES'] = 1
        case += '-route-bases'
    shared = build_identity['shared_memory_bytes'] if a.reserve_program_shared else 0
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    cases = out/'cases.inc'
    params = (batch, ng, nd, hidden, tng, tkg, tnd, tkd, slots, wng, wnd, experts,
              floats, page, r, softmax_all, shared)
    cases.write_text(f'CASE("{case}",{",".join(map(str, params))});\n')
    binary = out/'probe'
    source = ROOT/'scripts/a800_fused_expert_service_probe.cu'
    command = ['/usr/local/cuda-12.8/bin/nvcc', '-std=c++17', '-arch=sm_80',
               '--expt-relaxed-constexpr', '-O3', '-Xptxas=-v',
               f'-I{ROOT}/include', f'-I{ROOT}/third_party/cutlass/include',
               f'-DTILEMEGA_FUSED_SERVICE_CASES="{cases}"',
               *(f'-D{k}={v}' for k, v in variants.items()), str(source), '-o', str(binary)]
    with (out/'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=300)
    headers = (*HEADERS, 'Backend/Sm90DenseMoe.cuh', 'Backend/Sm90DenseMark.cuh',
               'Backend/Sm80MoePageOffsets.h')
    (out/'probe.identity.json').write_text(json.dumps(dict(
        contract='sm80-fused-expert-service-v1', architecture='sm_80', roles=[32,128],
        source_sha=subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        source_file_sha256=sha(source), anchor_sha256=sha(a.anchor), body_parameters=params,
        source_modifiers=variants, cases=[case], binary_sha256=sha(binary), command=command,
        body_headers={name: sha(ROOT/'include/tilemega'/name) for name in headers}), indent=2)+'\n')


if __name__ == '__main__':
    main()
