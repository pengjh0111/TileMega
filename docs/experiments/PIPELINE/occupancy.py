#!/usr/bin/env python3
"""B1-b occupancy budget: how large a prefetch page each cell can afford.

The page B1 appends after the `TaskSmem` union is dynamic shared memory on the
L2 worker kernel only, so it can cost residency.  Two independent answers are
produced for every cell and cross-checked:

  * the driver's own `cuOccupancyMaxActiveBlocksPerMultiprocessor`, queried on
    a real cubin built with the shipped flags -- the same call production makes
    through `TargetSpec::ActiveBlocksPerSM`;
  * F-40's closed form (`ORACLE/occupancy.sh:23-45`), generalised off its
    hard-coded 256-thread/8-warp shape, because these kernels run 128 threads.

Where the two disagree the driver wins and the difference is recorded.  Page 0
is validated against `ctas_per_sm`/`l2_ctas` as the harness itself printed them
in `PHASE2/raw/<cell>/correctness/selected/r0.log`, or -- for the real cells,
which B0 never ran -- one fresh run of this cell's own `control` binary.

Two answers per cell, not one: the sweep says how large a page the cell could
afford, and the `arms` table is B1-b's before/after proper -- the control cubin
against the prefetch cubin as `run.py` builds them, so the registers the
mechanism costs are measured rather than assumed unchanged.
"""
import json, os, pathlib, re, subprocess, sys

REPO = pathlib.Path('/root/TileMega')
FORK6 = REPO / 'docs/experiments/COSTMODEL/raw_kloop'
PHASE2 = REPO / 'docs/experiments/PHASE2/raw'
OUT = pathlib.Path(os.environ.get('PIPELINE_OUT', REPO / 'docs/experiments/PIPELINE/raw'))
WORK = pathlib.Path(os.environ.get('PIPELINE_WORK', '/tmp/pipeline_occ'))
CELLS = ['gqa2_s4', 'gqa2_s128', 'mha4_s4', 'mha4_s128', 'real_s4', 'real_s128']
ARCH = os.environ.get('PIPELINE_ARCH', 'sm_89')
THREADS = 128
NVCC = '/usr/local/cuda/bin/nvcc'
MANGLED = {
    'l2': '_ZN8tilemega7codegen18tilemega_l2_kernelEPKNS0_6ParamsEPNS0_12EventCounterEy',
    'l1': '_ZN8tilemega7codegen18tilemega_l1_kernelEPKNS0_6ParamsEPNS0_12EventCounterEy',
}
# JOINT/measure.py:32-41.  Omitting these was the first version's error: they
# change the L2 kernel's register count, so occupancy read off a bare compile
# is not the shipped one.
PROTOCOL = ['BARRIER_V2=1', 'EVENT_SOLO=1', 'EVENT_RED_PUBLISH=1', 'WAIT_POLICY=1',
            'WAIT_SPIN_ITERS=64', 'WAIT_BACKOFF_NS=64', 'WAIT_BACKOFF_GROW=1',
            'WAIT_BACKOFF_CAP_NS=64', 'SLOT_WINDOW=1']
INC = ['include', 'third_party/cutlass/include',
       'third_party/cutlass/tools/util/include', 'third_party/cutlass/test']
PAGES = [0, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384, 24576, 32768,
         49152, 65536, 84992]  # 84992 = the opt-in cap (101376) less the union


def build_probe():
    """The driver-API probe. Built here so the experiment needs nothing from
    outside the tree; it links the stub libcuda and resolves the real one at
    run time."""
    exe = WORK / 'occ_probe'
    src = REPO / 'docs/experiments/PIPELINE/occ_probe.cc'
    r = subprocess.run(['g++', '-O2', '-o', str(exe), str(src),
                        '-I/usr/local/cuda/include',
                        '-L/usr/local/cuda/lib64/stubs', '-lcuda'],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError('probe build failed:\n' + r.stderr[-2000:])
    return exe


def spec(cell):
    """`run.py`'s own arm specs, so the cubin here is the shipped compile: the
    source is the one re-projected at HEAD (`raw/regeneration.tsv`), which is
    the only one carrying the frontier field the mechanism reads."""
    return json.loads((OUT / cell / 'specs.json').read_text())


def shipped(cell):
    """What the harness printed for this cell on the runs that actually ran."""
    # On another architecture PHASE2's numbers belong to this machine's 4090,
    # not to the device under test, so ask the harness here instead.
    log = PHASE2 / cell / 'correctness/selected/r0.log'
    if ARCH != 'sm_89' or not log.exists():
        log = OUT / cell / 'baseline/control/r0.log'
        if not log.exists():
            sys.path.insert(0, str(REPO / 'docs/experiments/JOINT'))
            import measure
            model, seq = cell.rsplit('_s', 1)
            measure.run(OUT / cell, model, int(seq), 'control',
                        OUT / cell / 'baseline/control', 0, 0, 'occupancy')
    line = next(l for l in log.read_text().splitlines()
                if l.startswith('E2E_RESOURCE'))
    return dict(kv.split('=', 1) for kv in line.split()[1:])


def cubin(cell, arm='control'):
    s = spec(cell)[arm]
    out = WORK / f'{cell}.{arm}.cubin'
    cmd = [NVCC, '-std=c++17', '-O2', f'-arch={ARCH}', '-cubin',
           *['-DTILEMEGA_' + x for x in PROTOCOL + s.get('extra', [])],
           f"-DTILEMEGA_EVENT_KAPPA={s['kappa']}",
           f"-DTILEMEGA_RESIDENCY_CAP={s['residency']}",
           f"-DTILEMEGA_PLACEMENT={s['placement_macro']}",
           '-DTILEMEGA_TRACE_V2=0', '-DTILEMEGA_TRACE_PHASE=0',
           *['-I' + str(REPO / x) for x in INC],
           str(pathlib.Path(s['source']).resolve()), '-o', str(out)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'{cell} {arm} cubin failed:\n{r.stderr[-3000:]}')
    return out


def macro_page(cell):
    """The `TILEMEGA_PREFETCH_PAGE_BYTES` this cell's arms are built with; the
    worker allocates two of them, so the appended bytes are twice this."""
    extra = spec(cell)['prefetch'].get('extra', [])
    for x in extra:
        if x.startswith('PREFETCH_PAGE_BYTES='):
            return int(x.split('=', 1)[1])
    return 1024


def probe(cub, kernel, pages, base):
    args = [str(WORK / 'occ_probe'), str(cub), MANGLED[kernel], str(THREADS),
            *[str(base + p) for p in pages]]
    r = subprocess.run(args, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f'probe failed:\n{r.stdout}\n{r.stderr}')
    head = dict(kv.split('=', 1) for kv in r.stdout.splitlines()[0].split())
    got = {}
    for line in r.stdout.splitlines()[1:]:
        kv = dict(k.split('=', 1) for k in line.split())
        got[int(kv['dyn']) - base] = int(kv['ctas'])
    return int(head['regs']), got


def closed_form(regs, smem, threads=THREADS, reserve=0):
    """F-40, generalised to any CTA width; `reserve` is the driver's per-CTA
    shared-memory overhead, which F-40's sm_89 fit never had to model because
    every configuration in it was register bound."""
    warps = threads // 32
    per_cta_regs = warps * -(-regs * 32 // 256) * 256
    reg_lim = 65536 // per_cta_regs
    smem_lim = 102400 // (smem + reserve) if smem + reserve else 1 << 30
    return min(reg_lim, smem_lim, 1536 // threads), reg_lim, smem_lim


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    build_probe()
    rows, notes, arms = [], [], []
    for cell in CELLS:
        s, sh = spec(cell)['control'], shipped(cell)
        page = macro_page(cell)
        cub = cubin(cell)
        base = int(sh['task_smem'])
        regs, l2 = probe(cub, 'l2', PAGES, base)
        _, l1 = probe(cub, 'l1', [0], base)
        # `page` here is the dynamic bytes appended after the union, which is
        # twice the macro: the worker pages the copy so slot+1 can land while
        # slot still reads its own page.
        cap = int(s['residency'])
        afford = max(p for p in PAGES if l2[p] >= cap)
        free = max(p for p in PAGES if l2[p] >= l2[0])
        for p in PAGES:
            cf, rl, sl = closed_form(regs, base + p)
            cfr, _, slr = closed_form(regs, base + p, reserve=1024)
            rows.append(dict(cell=cell, config=pathlib.Path(s['source']).parent.name,
                             residency=cap, appended=p, macro_page=p // 2,
                             smem=base + p, regs=regs,
                             driver_ctas=l2[p], f40_ctas=cf, f40_reserved_ctas=cfr,
                             reg_lim=rl, smem_lim=sl, smem_lim_reserved=slr))
        # B1-b before/after: the two cubins `run.py` ships, each probed at the
        # shared memory its own harness would request.
        for arm in ('control', 'prefetch', 'inline'):
            appended = 0 if arm == 'control' else 2 * page
            aregs, actas = probe(cubin(cell, arm), 'l2', [appended], base)
            cf, rl, sl = closed_form(aregs, base + appended, reserve=1024)
            arms.append(dict(cell=cell, arm=arm, macro_page=0 if arm == 'control' else page,
                             smem_bytes=base + appended, appended=appended, regs=aregs,
                             driver_ctas=actas[appended], f40_reserved_ctas=cf,
                             reg_lim=rl, smem_lim_reserved=sl, residency=cap,
                             keeps_residency=int(actas[appended] >= cap)))
        notes.append(dict(cell=cell, regs_shipped=int(sh['reg']), regs_here=regs,
                          l2_ctas_shipped=int(sh['l2_ctas']), l2_ctas_here=l2[0],
                          l1_ctas_shipped=int(sh['l1_ctas']), l1_ctas_here=l1[0],
                          residency_used=cap, ctas_per_sm_shipped=int(sh['ctas_per_sm']),
                          free_appended_bytes=free, affordable_appended_bytes=afford,
                          macro_page_used=page, appended_used=2 * page))
    def write(name, recs):
        with (OUT / name).open('w') as f:
            f.write('\t'.join(recs[0]) + '\n')
            for r in recs:
                f.write('\t'.join(str(v) for v in r.values()) + '\n')
    write('occupancy.tsv', rows)
    write('occupancy_cells.tsv', notes)
    write('occupancy_arms.tsv', arms)
    for n in notes:
        print(n)
    bad = [n for n in notes if n['l2_ctas_here'] != n['l2_ctas_shipped']]
    print(('MISMATCH vs harness: ' + str(bad)) if bad
          else 'page-0 driver occupancy matches the harness on all cells')
    lost = [a for a in arms if not a['keeps_residency']]
    print(('RESIDENCY LOST: ' + str(lost)) if lost
          else 'every arm keeps the residency its cell was selected at')


if __name__ == '__main__':
    sys.exit(main())
