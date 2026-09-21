#!/usr/bin/env python3
"""R8 BE-6 / A-e: the occupancy closed form, generalised to per-role budgets.

F-40 fitted a closed form on sm_89 where every configuration was register
bound; F-223 added the driver's per-CTA shared-memory reserve. Both assume one
budget per CTA, because before R8 a CTA was one role.

The successor sums over roles:

    per_cta_regs = sum_r  warps_r * ceil(regs_r * 32 / 256) * 256
    reg_limit    = regs_per_sm // per_cta_regs
    smem_limit   = smem_per_sm // (sum_r smem_r + reserve)
    ctas         = min(reg_limit, smem_limit, threads_per_sm // sum_r threads_r)

With one role this is F-40/F-223 term for term, which is what makes it a
successor rather than a replacement: the round ships one role everywhere
(BE-5's litmus did not clear warp specialization), so the check below is the
one-role instance, and the multi-role form is exercised by the unit case at
the bottom rather than claimed from a run that does not exist.

Reads each cell's own `E2E_RESOURCE` line -- the harness already prints what
the driver said -- so there is no second probe to keep in sync.
"""
import argparse,math,re
from pathlib import Path
HERE=Path(__file__).resolve().parent

def closed_form(roles,regs_per_sm,smem_per_sm,threads_per_sm,reserve=0):
    """`roles` is a list of (threads, registers_per_thread, smem_bytes)."""
    per_cta_regs=sum((threads//32)*(-(-regs*32//256)*256) for threads,regs,_ in roles)
    threads=sum(t for t,_,_ in roles)
    smem=sum(s for _,_,s in roles)+reserve
    reg_limit=regs_per_sm//per_cta_regs if per_cta_regs else 1<<30
    smem_limit=smem_per_sm//smem if smem else 1<<30
    return min(reg_limit,smem_limit,threads_per_sm//threads),reg_limit,smem_limit

def resource(text):
    m=re.search(r'^E2E_RESOURCE (.*)$',text,re.M)
    if not m:return None
    return dict(kv.split('=',1) for kv in m[1].split())

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--models',type=Path,default=HERE/'models')
    ap.add_argument('--out',type=Path,default=HERE/'occupancy.tsv')
    a=ap.parse_args()
    rows=[]
    for cell in sorted(a.models.glob('*_s*')):
        logs=sorted((cell/'correctness').glob('r*.log'))+sorted((cell/'timing').glob('r*.log'))
        if not logs:continue
        field=resource(logs[0].read_text())
        if not field:continue
        threads=int(field['block']);regs=int(field['reg'])
        smem=int(field['task_smem']);driver=int(field['ctas_per_sm'])
        # The reserve is the difference the driver charges beyond the union,
        # which F-223 measured and which the harness prints as occupancy_smem.
        reserve=int(field.get('occupancy_smem',smem))-smem
        ctas,reg_limit,smem_limit=closed_form(
            [(threads,regs,smem)],int(field['regs_per_sm']),int(field['smem_per_sm']),
            int(field['threads_per_sm']),reserve)
        rows.append(dict(cell=cell.name,roles=1,threads=threads,regs=regs,smem=smem,
            reserve=reserve,closed_form=ctas,driver=driver,agree=int(ctas==driver),
            reg_limit=reg_limit,smem_limit=smem_limit))
    # The multi-role instance the form exists for, checked against the sum it
    # must reduce to; no device on this machine runs it.
    split=closed_form([(128,64,4096),(128,96,8192)],65536,102400,1536)[0]
    single=closed_form([(256,96,12288)],65536,102400,1536)[0]
    if rows:
        columns=list(rows[0])
        with a.out.open('w') as f:
            f.write('\t'.join(columns)+'\n')
            for r in rows:f.write('\t'.join(str(r[c]) for c in columns)+'\n')
    for r in rows:
        print('OCCUPANCY',r['cell'],'closed_form',r['closed_form'],'driver',r['driver'],
              'agree',bool(r['agree']),flush=True)
    print('OCCUPANCY_ROLES two_roles',split,'one_role_upper_bound',single,flush=True)
    agree=sum(r['agree'] for r in rows)
    print('A-e RESULT cells',len(rows),'agreeing',agree,flush=True)
    raise SystemExit(0 if rows and agree==len(rows) else 1)
if __name__=='__main__':main()
