"""Synthetic analytic checks only; fitting/validation uses separately held data."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
from fit_sm90r2_flow_transfers import Trial,predict,profile,score

base=1<<60
t=Trial((0,'synthetic',0,2,1,1),[(0,0,base,base+20,base+100,base+110),
    (1,0,base,base+30,base+110,base+120)])
model=dict(admission_occupancy_ns=10,response_lag_ns=100,packet_base_ns=20,
    packet_queue_fraction=1,reader_retire_observer_ns=10)
p=predict(t,model);assert p[0,0]==(0.,20.,100.)and p[1,0]==(0.,30.,110.)
loss,fit=profile([t],10);assert loss<1e-12 and abs(fit['response_lag_ns']-100)<1e-9
assert abs(fit['packet_base_ns']-20)<1e-9 and abs(fit['packet_queue_fraction']-1)<1e-9
assert score(t,model,'conditional')['envelope_relative_error']==0
many=Trial((0,'synthetic',0,1,4,1),[(0,s,base,base+20,base+100,base+110)for s in range(4)])
finite=predict(many,model,True)
for s in range(1,4):assert finite[0,s][0]>=finite[0,s-1][1]
for s in range(2,4):assert finite[0,s][0]>=finite[0,s-2][2]+10
assert finite[0,2][0]==110
reverse=predict(t,model,False,True);assert reverse[1,0][2]==100 and reverse[0,0][2]==110
print('analytic FIFO, paired WLS, integer deltas and finite S2 tests passed')

from rollout_sm90r2_flow_transfers import autonomous
a=autonomous({0:0},4,model)
assert a["packets"][0,2][0]==110
for s in range(2,4):assert a["packets"][0,s][0]>=a["empty"][0,s-2]
connected=autonomous({0:0},2,model,[30,60],90)
assert connected["consumer_returns"][0]==190
for progress,returned in [([],90),([30,60],50),([30,float('nan')],90)]:
 try:autonomous({0:0},2,model,progress,returned)
 except ValueError:pass
 else:raise AssertionError('invalid connected response accepted')
try:autonomous({0:float('nan')},2,model)
except ValueError:pass
else:raise AssertionError('nonfinite source entry accepted')
print("autonomous entry-only and connected-progress tests passed")
