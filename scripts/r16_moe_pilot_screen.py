"""Conservative, CPU-only comparison for a predeclared MoE pilot screen."""


def evaluate(incumbent, pilot_e2e, pilot_ttft, ratio=1.10):
    if len(incumbent) != 3 or len(pilot_e2e) != 3 or len(pilot_ttft) != 3:
        raise ValueError('screen requires three incumbent processes and three pilot repeats')
    import math
    e2e = [float(x['e2e_seconds']) for x in incumbent]
    tpot = [float(x['tpot_seconds']) for x in incumbent]
    values = e2e + tpot + list(pilot_e2e) + list(pilot_ttft)
    if not all(math.isfinite(v) and v > 0 for v in values) or not math.isfinite(ratio) or ratio < 1.10:
        raise ValueError('invalid screen values or weakened ratio')
    conservative = [(v - max(pilot_ttft)) / 1023 for v in pilot_e2e]
    spread = lambda xs: max(xs) - min(xs)
    e2e_gap = min(pilot_e2e) - max(e2e)
    tpot_gap = min(conservative) - max(tpot)
    checks = {
        'all_e2e_slower': min(pilot_e2e) > ratio * max(e2e),
        'all_conservative_tpot_slower': min(conservative) > ratio * max(tpot),
        'e2e_exceeds_both_ranges': e2e_gap > max(spread(pilot_e2e), spread(e2e)),
        'tpot_exceeds_both_ranges': tpot_gap > max(spread(conservative), spread(tpot)),
    }
    return dict(screened_out=all(checks.values()), checks=checks, ratio=ratio,
                pilot_conservative_tpot_seconds=conservative,
                minimum_e2e_gap=e2e_gap, minimum_tpot_gap=tpot_gap)
