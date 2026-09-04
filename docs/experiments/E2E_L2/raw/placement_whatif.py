import re, math, collections
rows=[]
for line in open('/tmp/waitset.txt'):
    d=dict(kv.split('=') for kv in line.split()[1:])
    rows.append({k:int(v) for k,v in d.items()})

def groups_strided(lo,hi,grid,live,kap):
    if lo>=hi: return 0
    if hi-lo>=grid: return (live-1)//kap+1
    a,b=lo%grid,(hi-1)%grid
    if a<=b: return b//kap-a//kap+1
    return (live-1)//kap-a//kap+1 + b//kap+1

def groups_blocked(lo,hi,chunk,kap):
    if lo>=hi: return 0
    return (hi-1)//chunk//kap - lo//chunk//kap + 1

for seq in sorted({r['seq'] for r in rows}):
    print('seq=%d' % seq)
    for kap in (1,2,4,8,16,32):
        tot={'strided':[0,0],'blocked':[0,0]}
        for r in rows:
            if r['seq']!=seq: continue
            grid=r['grid']; pt=r['p_tasks']; ct=r['c_tasks']
            live=min(pt,grid) or 1
            pchunk=max(1,math.ceil(pt/grid)); pcta=math.ceil(pt/pchunk)
            cchunk=max(1,math.ceil(ct/grid)); ccta=math.ceil(ct/cchunk)
            gs=(live-1)//kap+1
            gb=(pcta-1)//kap+1
            # strided: consumer CTAs 0..live_c-1, each owning c, c+grid, ...
            live_c=min(ct,grid) or 1
            ps=0
            for c in range(live_c):
                if r['map']==1: ps+=gs; continue
                for t in range(c,ct,grid):
                    at=(t//r['div'])*r['scale']+r['offset']
                    ps+=groups_strided(max(at,0),min(at+r['count'],pt),grid,live,kap)
            pb=0
            for c in range(ccta):
                if r['map']==1: pb+=gb; continue
                lo=hi=None
                for t in range(c*cchunk,min((c+1)*cchunk,ct)):
                    at=(t//r['div'])*r['scale']+r['offset']
                    a,b=max(at,0),min(at+r['count'],pt)
                    if a<b: lo=a if lo is None else min(lo,a); hi=b if hi is None else max(hi,b)
                if lo is not None: pb+=groups_blocked(lo,hi,pchunk,kap)
            tot['strided'][0]+=ps; tot['strided'][1]+=live_c*gs
            tot['blocked'][0]+=pb; tot['blocked'][1]+=ccta*gb
        s,r_=tot['strided']; b,rb=tot['blocked']
        print('  kappa=%-3d strided polls=%-8d relaxed=%-8d narrow=%.3fx | blocked polls=%-8d relaxed=%-8d narrow=%.3fx'
              % (kap,s,r_,r_/s if s else 0,b,rb,rb/b if b else 0))
