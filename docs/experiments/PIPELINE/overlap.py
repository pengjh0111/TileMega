#!/usr/bin/env python3
"""B1-c: the overlap itself, read off the B0 phase trace.

`prefetch_wait_cycles` brackets one issue and one `cp.async.wait_group` in both
timed arms.  The pipelined arm waits on a copy issued during the *previous*
slot's body; the inline arm issues and waits for the same bytes in place.  The
per-slot difference between the two is therefore the part of the fetch that ran
under the earlier task -- measured, not inferred from an end-to-end time.

Slots the rule refuses (`prefetch_issued=0`) run the same instruction sequence
with no copy in flight, and their wait is reported as the instrument's floor.
"""
import argparse, csv, json, pathlib, statistics, sys

HERE = pathlib.Path(__file__).resolve().parent
CELLS = ['gqa2_s4', 'gqa2_s128', 'mha4_s4', 'mha4_s128', 'real_s4', 'real_s128']
ARMS = ('prefetch', 'inline')


def table(p):
    with p.open() as f:
        return list(csv.DictReader(f, delimiter='\t'))


def rounds(cell, arm, tag):
    folder = HERE / 'raw' / cell / ('phase' + tag)
    return sorted(folder.glob(f'{arm}_r*/dump/phases.tsv'))


def collect(cell, arm, tag):
    """Median wait per slot across rounds, so one scheduling hiccup cannot move
    a slot's number."""
    waits, issued, body = {}, {}, {}
    files = rounds(cell, arm, tag)
    for f in files:
        rows = table(f)
        for row in rows:
            slot = int(row['slot'])
            waits.setdefault(slot, []).append(int(row['prefetch_wait_cycles']))
            issued[slot] = int(row['prefetch_issued'])
            body.setdefault(slot, []).append(
                int(row['run_end_cycles']) - int(row['run_begin_cycles']))
    return (len(files), {s: statistics.median(v) for s, v in waits.items()}, issued,
            {s: statistics.median(v) for s, v in body.items()})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--cells', nargs='+', default=CELLS)
    ap.add_argument('--tag', default='')
    a = ap.parse_args()
    out, missing = [], []
    for cell in a.cells:
        data = {arm: collect(cell, arm, a.tag) for arm in ARMS}
        if any(d[0] == 0 for d in data.values()):
            missing.append(cell)
            continue
        n_pre, pre, issued, body = data['prefetch']
        n_in, inl, issued_in, body_in = data['inline']
        if issued != issued_in:
            raise RuntimeError('the two arms disagree on which slots prefetch: ' + cell)
        hot = sorted(s for s, v in issued.items() if v)
        cold = sorted(s for s, v in issued.items() if not v)
        gain = [inl[s] - pre[s] for s in hot]
        row = dict(cell=cell, rounds_prefetch=n_pre, rounds_inline=n_in,
                   slots=len(issued), issued_slots=len(hot),
                   prefetch_wait_issued=round(statistics.median(pre[s] for s in hot), 1) if hot else '',
                   inline_wait_issued=round(statistics.median(inl[s] for s in hot), 1) if hot else '',
                   prefetch_wait_idle=round(statistics.median(pre[s] for s in cold), 1) if cold else '',
                   inline_wait_idle=round(statistics.median(inl[s] for s in cold), 1) if cold else '',
                   overlap_cycles_median=round(statistics.median(gain), 1) if gain else '',
                   overlap_cycles_total=round(sum(gain), 1) if gain else '',
                   overlap_slots_positive=sum(g > 0 for g in gain),
                   # The window the copy has to hide in: the body it runs under.
                   prev_body_cycles_median=round(statistics.median(
                       body[s - 1] for s in hot if s - 1 in body), 1) if hot else '',
                   # The pipelined arm issues slot s's copy inside slot s-1's
                   # body, where the wait bracket cannot see it.  Charging that
                   # body against the inline arm's same body is what separates
                   # a fetch that was removed from one that was relocated.
                   prev_body_inline_cycles_median=round(statistics.median(
                       body_in[s - 1] for s in hot if s - 1 in body_in), 1) if hot else '')
        # Everything above is a difference of two `clock64()` reads taken by one
        # slot on one SM, which is the only comparison that clock supports: the
        # counter is per-SM and the dump carries no worker identity, so the
        # schedule has no common time base and no nanosecond conversion is
        # attempted here.  The share below is the same cycles against the body
        # the copy had to hide under, which is the quantity B1-c is about.
        if row['prev_body_cycles_median'] and row['prev_body_inline_cycles_median']:
            row['prev_body_delta_cycles'] = round(
                row['prev_body_cycles_median'] - row['prev_body_inline_cycles_median'], 1)
            row['net_cycles_per_issue'] = round(
                row['overlap_cycles_median'] - row['prev_body_delta_cycles'], 1)
        else:
            row['prev_body_delta_cycles'] = row['net_cycles_per_issue'] = ''
        row['overlap_share_of_prev_body'] = (
            round(row['overlap_cycles_median'] / row['prev_body_cycles_median'], 4)
            if row['overlap_cycles_median'] and row['prev_body_cycles_median'] else '')
        # Against the whole schedule: every slot's body summed over the inline
        # arm.  That is aggregate work across all workers, not a makespan --
        # `clock64()` is per-SM and the dump has no worker column, so no makespan
        # is available here.  With the workers balanced the two are proportional,
        # which makes this an order-of-magnitude bound on what the overlap could
        # move end to end, not a prediction of it.
        row['body_total_inline_cycles'] = sum(body_in.values())
        row['net_recovered_cycles'] = (
            round(row['net_cycles_per_issue'] * len(hot), 1)
            if row['net_cycles_per_issue'] != '' else '')
        row['net_share_of_body_total'] = (
            round(row['net_recovered_cycles'] / row['body_total_inline_cycles'], 6)
            if row['net_recovered_cycles'] != '' and row['body_total_inline_cycles'] else '')
        out.append(row)
    if not out:
        print('OVERLAP no phase dumps; run `run.py phase` first', file=sys.stderr)
        return 1
    dest = HERE / 'raw' / ('overlap' + a.tag + '.tsv')
    with dest.open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]), delimiter='\t', lineterminator='\n')
        w.writeheader()
        w.writerows(out)
    for r in out:
        print(json.dumps(r), flush=True)
    if missing:
        print('OVERLAP missing dumps for ' + ' '.join(missing))
    return 0


if __name__ == '__main__':
    sys.exit(main())
