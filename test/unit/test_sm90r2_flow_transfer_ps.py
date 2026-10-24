"""Analytic processor-sharing conservation and finite autonomous causality."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from fit_sm90r2_flow_transfers_ps import fluid_completions,autonomous,profile
from fit_sm90r2_flow_transfers import Trial
assert fluid_completions([0,0],10)==[20,20]
assert fluid_completions([0,5],10)==[15,20]
assert fluid_completions([0,30],10)==[10,40]
assert fluid_completions([0,0,0],10)==[30,30,30]
t=Trial((0,'synthetic',0,2,1,1),[(0,0,0,20,120,130),(1,0,0,20,120,130)])
loss,m=profile([t],10,10);assert loss<1e-12 and abs(m['propagation_observation_lag_ns']-100)<1e-9
assert m['packet_response_ns']==20
trace=autonomous({0:0},4,m);assert trace['packets'][0,2][0]==120
for s in range(1,4):assert trace['packets'][0,s][0]>=trace['packets'][0,s-1][1]
for s in range(2,4):assert trace['packets'][0,s][0]>=trace['empty'][0,s-2]
two=autonomous({0:0,1:0},1,m);assert two['packets'][0,0][2]==120 and two['packets'][1,0][2]==120
print('analytic shared-capacity, concurrent completions and autonomous S2 tests passed')

for progress,returned in [([],90),([30,60],50),([30,float('nan')],90)]:
 try:autonomous({0:0},2,m,progress,returned)
 except ValueError:pass
 else:raise AssertionError('invalid connected response accepted')
for arrivals,work in [([0,float('nan')],10),([0],float('nan'))]:
 try:fluid_completions(arrivals,work)
 except ValueError:pass
 else:raise AssertionError('nonfinite fluid state accepted')
