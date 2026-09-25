#!/usr/bin/env python3
"""Wait for recorded test runners, then regenerate an unreviewed report.

Does not launch duplicate searches, change tests, commit, or push. A completed
status means the runners ended and all checks ran; it does not mean gates pass.
"""
import argparse,datetime,json,pathlib,subprocess,sys,time

E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]

def identity(pid):
    try:
        # comm may contain spaces or parentheses; fields begin after its last ).
        fields=pathlib.Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return fields[19] if fields[0]!='Z' else None
    except FileNotFoundError:return None

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('manifest',type=pathlib.Path)
    args=parser.parse_args();jobs=json.loads(args.manifest.read_text())['jobs']
    out=E/'queued_checks';out.mkdir(exist_ok=True)
    while True:
        pending=[j for j in jobs if identity(j['pid'])==j['start_ticks']]
        blocked=(E/'global_stop.json').exists()
        state=dict(state='waiting' if pending and not blocked else 'checking',
                   observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   pending=[j['name'] for j in pending],global_stop=blocked)
        (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
        if not pending or blocked:break
        time.sleep(30)
    command=[sys.executable,str(E/'write_summary.py')]
    (out/'command.json').write_text(json.dumps(command)+'\n')
    with (out/'report.log').open('x') as log:
        code=subprocess.call(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    state.update(state='checks_finished_awaiting_review',report_exit=code,
                 observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (out/'status.json').write_text(json.dumps(state,indent=2)+'\n')
    print(json.dumps(state),flush=True)
    return code

if __name__=='__main__':raise SystemExit(main())
