#!/usr/bin/env python3
"""B1-e: paired ratios and CIs for the six cells, against R6's selected config.

`control` is FORK6's selected configuration verbatim, so the ratio against it
is the ratio against R6's choice.  A round contributes one sample per arm and
the ratio is formed inside the round, so no median crosses a session boundary.
Three ratios are published from the same rounds: `prefetch/control` is what the
round buys, `inline/control` is what the storage and the instructions cost on
their own, and `prefetch/inline` is the overlap with the storage held fixed.

No threshold: R7 §5.2 asks for the numbers and a diagnosis, not a verdict.
"""
import argparse, csv, json, math, os, pathlib, random, re, statistics, sys

HERE = pathlib.Path(__file__).resolve().parent
# The sm_120 runner keeps its rounds outside the repository.
RAW = pathlib.Path(os.environ.get('PIPELINE_OUT', HERE / 'raw'))
CELLS = ['gqa2_s4', 'gqa2_s128', 'mha4_s4', 'mha4_s128', 'real_s4', 'real_s128']
PAIRS = (('prefetch', 'control'), ('inline', 'control'), ('prefetch', 'inline'))
TIME = re.compile(r'^E2E_TIME .*?\bl2_ms=([0-9.]+)', re.M)
ISSUED = re.compile(r'^E2E_PREFETCH .*?\bissued=(\d+).*?\bdeclared=(\d+)', re.M)
DECL = re.compile(r'^E2E_PREFETCH slots=(\d+) declared=(\d+) issued=(\d+)', re.M)


def samples(cell, arm, tag):
    """round -> l2_ms for one arm of one cell."""
    out, issued = {}, None
    for log in (RAW / cell / ('e2e' + tag) / arm).glob('r*.log'):
        text = log.read_text()
        m = TIME.search(text)
        if m:
            out[int(re.findall(r'\d+', log.name)[0])] = float(m.group(1))
        d = DECL.search(text)
        if d:
            issued = (int(d.group(1)), int(d.group(2)), int(d.group(3)))
    return out, issued


def bootstrap(values, draws=20000):
    rng = random.Random(20260919)
    n = len(values)
    estimates = sorted(statistics.median(values[rng.randrange(n)] for _ in range(n))
                       for _ in range(draws))
    return estimates[int(.025 * draws)], estimates[int(.975 * draws) - 1]


def signed_rank(values):
    """Wilcoxon on the deviation from an unchanged ratio."""
    values = [v for v in values if v]
    n = len(values)
    if n < 2:
        return float('nan')
    order = sorted(range(n), key=lambda i: abs(values[i]))
    ranks = [0.] * n
    tie = 0.
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(values[order[j + 1]]) == abs(values[order[i]]):
            j += 1
        rank = .5 * (i + j) + 1
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        count = j - i + 1
        tie += count ** 3 - count
        i = j + 1
    positive = sum(ranks[i] for i, v in enumerate(values) if v > 0)
    mean = n * (n + 1) / 4
    variance = n * (n + 1) * (2 * n + 1) / 24 - tie / 48
    z = (abs(positive - mean) - .5) / math.sqrt(variance)
    return math.erfc(z / math.sqrt(2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells', nargs='+', default=CELLS)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    out = []
    for cell in a.cells:
        arms = {arm: samples(cell, arm, a.tag) for arm in ('control', 'prefetch', 'inline')}
        if not all(v[0] for v in arms.values()):
            continue
        slots, declared, issued = arms['prefetch'][1] or (0, 0, 0)
        for arm, base in PAIRS:
            top, bottom = arms[arm][0], arms[base][0]
            rounds = sorted(set(top) & set(bottom))
            ratios = [top[r] / bottom[r] for r in rounds]
            lo, hi = bootstrap(ratios)
            out.append(dict(cell=cell, arm=arm, base=base, rounds=len(rounds),
                            base_median_ms=round(statistics.median(bottom[r] for r in rounds), 6),
                            arm_median_ms=round(statistics.median(top[r] for r in rounds), 6),
                            ratio=round(statistics.median(ratios), 4),
                            ci_lo=round(lo, 4), ci_hi=round(hi, 4),
                            p_wilcoxon=f'{signed_rank([x - 1 for x in ratios]):.3e}',
                            slots=slots, declared=declared, issued=issued))
    if not out:
        print('E2E no rounds; run `run.py e2e` first', file=sys.stderr)
        return 1
    dest = RAW / ('e2e' + a.tag + '.tsv')
    with dest.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(json.dumps(r), flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
