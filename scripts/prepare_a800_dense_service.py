#!/usr/bin/env python3
"""Compile a bounded A800 service probe from a formal decode artifact's shapes."""
import argparse
import json
from pathlib import Path
import re
import subprocess

from collect_a800_dense_service import production_key, sha

ROOT = Path(__file__).resolve().parents[1]
HEADERS = ('Backend/Sm90DenseGemm.h', 'Backend/Sm80DenseService.cuh',
           'Backend/Sm90DenseLayout.h', 'Backend/Sm90DenseRing.cuh',
           'Backend/Sm90DenseEpilogue.h', 'Codegen/executor/Async.cuh')
EPILOGUE = {'qkv': ('store', 'kStore'), 'o': ('residual', 'kResidual'),
            'gate_up': ('swiglu', 'kSwiGLU'), 'down': ('residual', 'kResidual')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--anchor', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--families', nargs='+', choices=EPILOGUE, required=True)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--copy', type=int, choices=(0, 4, 8), default=0)
    parser.add_argument('--residual-copy', action='store_true')
    parser.add_argument('--reserve-program-shared', action='store_true')
    parser.add_argument('--epilogue-clock', action='store_true',
                        help='measure residual epilogue and side wait without mainloop timers')
    parser.add_argument('--u', nargs='+', type=int, choices=(4, 8, 16))
    parser.add_argument('--ring', action='append', default=[], metavar='FAMILY:SLOTS[:TILE_K[:TILE_N[:WARPS_N]]]',
                        help='compile an alternative ring depth and optional K and N tiles and N warp count')
    args = parser.parse_args()
    if args.epilogue_clock and any(family not in ('o', 'down') for family in args.families):
        parser.error('clock-based epilogue timing requires residual families')
    if args.residual_copy and (not args.copy or not any(f in ('o', 'down') for f in args.families)):
        parser.error('--residual-copy requires a copy group and at least one residual family')
    rings = {}
    for specification in args.ring:
        fields = specification.split(':')
        if (not 2 <= len(fields) <= 5 or fields[0] not in args.families
                or not all(field.isdecimal() for field in fields[1:])
                or not 2 <= int(fields[1]) <= 8
                or len(fields) >= 3 and int(fields[2]) < 64
                or len(fields) >= 4 and int(fields[3]) < 8
                or len(fields) == 5 and int(fields[4]) not in (1, 2, 4)):
            parser.error('--ring requires a selected family, 2 to 8 slots and optional K and N tiles and N warp count')
        family, depth = fields[:2]
        tile_k, tile_n, warps_n = (tuple(map(int, fields[2:])) + (None,)*3)[:3]
        variant = int(depth), tile_k, tile_n, warps_n
        if variant in rings.setdefault(family, []):
            parser.error('--ring repeats a family and geometry')
        rings[family].append(variant)
    anchor = json.loads(args.anchor.read_text())
    decode = Path(anchor['decode'])
    if sha(decode) != anchor['decode_sha256']:
        raise ValueError('anchor decode differs from its formal artifact')
    coords = anchor['coordinates']
    manifest = json.loads(Path(str(decode)+'.plan.json').read_text())
    reserve = (json.loads(Path(str(decode)+'.identity.json').read_text())['shared_memory_bytes']
               if args.reserve_program_shared else 0)
    gems = coords['gemms']
    batch, grid, residency = anchor['batch'], manifest['grid'], coords['residency']
    if grid != 108 * residency or len(set(args.families)) != len(args.families):
        raise ValueError('expected unique A800-SXM4-80GB service cases')
    # Generated body aliases carry the actual padded N, K and epilogue.
    # An export input shape can differ after QKV fusion and weight packing.
    generated = Path(str(decode)+'.cu').read_text()
    aliases = [tuple((*(int(v) for v in row[:7]), row[7], int(row[8]), row[9] == 'true'))
               for row in re.findall(r'AmpereGemm<(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),(\d+),\d+,'
                                     r'backend::ServingEpilogueOp::(\w+),(\d+),(true|false),', generated)]
    cases = {}
    definitions = []
    suffix = f'{anchor["model"]}-b{batch}'
    for family in args.families:
        g = gems[family]
        epilogue, operator = EPILOGUE[family]
        matching = list(dict.fromkeys(row for row in aliases if row[0] == batch
                                     and row[3:7] == (g['tile_n'], g['slot_bytes']//(2*g['tile_n']), g['slots'], g['warps_n'])
                                     and row[7] == operator))
        if family in ('o', 'down') and len(matching) == 2:
            matching = [matching[1 if family == 'down' else 0]]
        if len(matching) != 1:
            raise ValueError(f'{family} does not identify one production alias')
        _, n, k, incumbent_tn, incumbent_tk, incumbent_slots, incumbent_wn, _, packed_u, norm = matching[0]
        if 4//incumbent_wn != g['warps_k'] or norm != (epilogue != 'residual'):
            raise ValueError(f'incumbent {family} coordinate and body disagree')
        for slots, tile_k, tile_n, warps_n in rings.get(family, [(incumbent_slots, None, None, None)]):
            tn = incumbent_tn if tile_n is None else tile_n
            tk = incumbent_tk if tile_k is None else tile_k
            wn = incumbent_wn if warps_n is None else warps_n
            wk = 4//wn
            if n % tn and (family != 'gate_up' or coords.get('payload') != 'reuse'
                           or batch != 16 or residency != 2 or (n+tn-1)//tn > grid
                           or n % (2*packed_u) or (tn, tk, slots, wn) !=
                           (incumbent_tn, incumbent_tk, incumbent_slots, incumbent_wn)):
                raise ValueError('partial N probe requires the linked one-wave gate')
            if tn % (8 * wn):
                raise ValueError(f'{family} has unaligned N tile {tn}')
            if k % tk or tk % 64 or tk < 16 * wk * g['register_stages']:
                raise ValueError(f'{family} has unaligned K tile {tk}')
            for u in (args.u or (packed_u,)) if family == 'gate_up' else (16,):
                if family == 'gate_up' and tn % (2*u):
                    continue
                live = args.live and batch < 16 and (family == 'gate_up' or wk > 1)
                copy = args.copy if family == 'gate_up' or (args.residual_copy and family in ('o', 'down')) else 0
                side = max(2, (n//tn + grid-1)//grid) if epilogue == 'residual' else 0
                name = f'{suffix}-{family}-u{u}-rows{int(live)}-copy{copy}'
                if family in rings:
                    name += f'-slots{slots}'
                if tile_k is not None:
                    name += f'-tk{tk}'
                if tile_n is not None:
                    name += f'-tn{tn}'
                if warps_n is not None:
                    name += f'-wn{wn}'
                key = production_key(dict(epilogue=epilogue, grid=grid, n=n, k=k, batch=batch,
                    tn=tn, tk=tk, slots=slots, warps_n=wn, warps_k=wk, register_stages=g['register_stages'],
                    u=u, residual_tiles=side, input_norm=norm, live_rows=live, copy_issue_group=copy))
                if key in cases.values():
                    raise ValueError(f'duplicate production geometry {key}')
                cases[name] = key
                definitions.append(f'CASE("{name}",{batch},{n},{k},{tn},{tk},{slots},{g["register_stages"]},'
                                   f'{residency},{wn},ServingEpilogueOp::{operator},{u},{str(norm).lower()});')
    if not 0 < len(cases) <= 9:
        raise ValueError('provide one to nine distinct production service cases')
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    include = out/'cases.inc'
    include.write_text('\n'.join(definitions)+'\n')
    binary = out/'probe'
    command = ['/usr/local/cuda-12.8/bin/nvcc', '-std=c++17', '-arch=sm_80',
               '--expt-relaxed-constexpr', '-O3', '-Xptxas=-v',
               f'-I{ROOT}/include', f'-I{ROOT}/third_party/cutlass/include',
               f'-DTILEMEGA_SERVICE_CASES="{include}"', f'-DTILEMEGA_SM80_COPY_ISSUE_GROUP={args.copy}',
               f'-DTILEMEGA_SM80_ACTIVE_EPILOGUE={int(args.live)}',
               f'-DTILEMEGA_DENSE_SERVICE_RESERVE_SHARED={reserve}',
               *(['-DTILEMEGA_SM80_RESIDUAL_COPY_GROUP=1'] if args.residual_copy else []),
               *(['-DTILEMEGA_SM80_DENSE_EPILOGUE_CLOCK=1'] if args.epilogue_clock else []),
               str(ROOT/'scripts/a800_dense_service_probe.cu'), '-o', str(binary)]
    with (out/'build.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=260)
    listing = subprocess.check_output(['/usr/local/cuda-12.8/bin/cuobjdump', '--list-elf', str(binary)], text=True)
    if 'sm_80' not in listing:
        raise ValueError('probe has no linked SM80 image')
    (out/'probe.identity.json').write_text(json.dumps(dict(
        contract='sm80-dense-gemm-service-v1', source_sha=subprocess.check_output(
            ['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip(),
        anchor_sha256=sha(args.anchor), source_file_sha256=sha(ROOT/'scripts/a800_dense_service_probe.cu'),
        command=command, architecture='sm_80', roles=[32,128],
        reserved_program_shared=reserve,
        cases=list(cases), geometry_by_case=cases, binary_sha256=sha(binary),
        body_headers={name: sha(ROOT/'include/tilemega'/name) for name in HEADERS}), indent=2)+'\n')


if __name__ == '__main__':
    main()
