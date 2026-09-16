#!/usr/bin/env python3
"""Fresh old-geometry loop calibration; independent of the frozen FORK6 cohort."""
import kloop

def control(model,seq):
    return {'selected':dict(source=str(kloop.measure.source(model)),placement_macro='5',
        kappa='1',residency='0',m='128',n='128',k='16',stages='3',split='1',extra=['TRACE_KLOOP=1'])}
kloop.specs=control
if __name__=='__main__':kloop.main()
