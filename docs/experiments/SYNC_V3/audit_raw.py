#!/usr/bin/env python3
"""Raw-evidence readers shared by the final verifier and analysis scripts."""
import csv
import hashlib
import json
from pathlib import Path
import re
import subprocess

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


def result(path, metadata=True):
    text = path.read_text()
    statuses = re.findall(r'^RESULT status=(\S+)', text, re.M)
    if len(statuses) != 1:
        raise ValueError(f'{path}: expected exactly one RESULT')
    meta = json.loads(path.with_suffix('.json').read_text()) if metadata else {}
    return statuses[0], meta


def litmus(raw=None):
    raw = raw or HERE / 'litmus_v3'
    build = json.loads((raw / 'build.json').read_text())
    if hashlib.sha256((HERE / 'litmus.cu').read_bytes()).hexdigest() != build['source_sha256']:
        raise ValueError('litmus source differs from the compiled experiment')
    details = []
    stamps = set()
    for suite, negative in (('cache', 'no_fence'), ('skew', 'no_barrier')):
        for grid in (64, 128, 256):
            for tile in (1024, 4096):
                for arm in ('per_writer', 'thread0_fence', negative):
                    folder = raw / 'scan' / suite / f'g{grid}_t{tile}' / arm
                    paths = sorted(folder.glob('r*.log'))
                    if len(paths) != 50:
                        raise ValueError(f'{folder}: {len(paths)}/50 raw runs')
                    passed = 0
                    expected = 'MISMATCH' if arm == negative else 'pass'
                    for i in range(50):
                        status, meta = result(folder / f'r{i}.log')
                        if meta['time_ns'] in stamps:
                            raise ValueError(f'{folder}: duplicated process timestamp')
                        stamps.add(meta['time_ns'])
                        command = meta['command']
                        skew = command[command.index('--writer-skew-cycles')+1]
                        if skew != ('2000000' if suite == 'skew' else '0'):
                            raise ValueError(f'{folder}: stress changed between arms')
                        if ('--no-acquire-fence' in command) != (suite == 'cache'):
                            raise ValueError(f'{folder}: inconsistent consumer acquire')
                        if command[command.index('--release')+1] != arm or meta['round'] != i:
                            raise ValueError(f'{folder}: mislabeled process')
                        passed += status == expected and meta['exit_code'] == (1 if arm == negative else 0)
                    details.append(f'{suite}/g{grid}/t{tile}/{arm}={passed}/50')
                    if passed != 50:
                        raise ValueError(details[-1])
    return '; '.join(details)


def five_arm(raw):
    details = []
    sessions = set()
    for model in ('gqa2', 'mha4'):
        for seq in (4, 128):
            for placement in (0, 5):
                for arm in ('full', 'nofence', 'nowait', 'neither', 'l1nosync'):
                    folder = raw / 'paired' / f'{model}_s{seq}_p{placement}' / arm
                    if len(list(folder.glob('r*.log'))) != 25:
                        raise ValueError(f'{folder}: not 25 logs')
                    for i in range(25):
                        path = folder / f'r{i}.log'
                        status, meta = result(path)
                        if arm == 'full' and (status != 'PASS' or meta['exit_code']):
                            raise ValueError(f'{path}: full correctness failed')
                        if meta['exit_code'] not in (0, 1):
                            raise ValueError(f'{path}: process did not complete normally')
                        if len(re.findall(r'^E2E_TIME ', path.read_text(), re.M)) != 1:
                            raise ValueError(f'{path}: missing timing')
                        combo = (0 if placement == 0 else 5) + ('full', 'nofence', 'nowait', 'neither', 'l1nosync').index(arm)
                        if (i + meta['order']) % 10 != combo or meta['round'] != i:
                            raise ValueError(f'{path}: arm order is not rotated')
                        sessions.add(meta['session'])
                    details.append(f'{model}/s{seq}/p{placement}/{arm}=25/25')
    if len(sessions) != 1:
        raise ValueError(f'mixed measurement sessions: {sessions}')
    return '; '.join(details)


def targets(analyzer):
    frozen = json.loads((HERE / 'targets_raw/frozen.json').read_text())
    target = HERE / 'targets.tsv'
    if hashlib.sha256(target.read_bytes()).hexdigest() != frozen['target_sha256']:
        raise ValueError('frozen targets changed')
    rows = list(csv.DictReader(target.open(), delimiter='\t'))
    expected = {(c, m, str(s)) for c in ('legacy_grid_stride', 'balanced', 'rotate', 'eft', 'wavefront', 'chain')
                for m in ('gqa2', 'mha4') for s in (4, 128)}
    if len(rows) != 24 or {(r['candidate'], r['model'], r['seq']) for r in rows} != expected:
        raise ValueError('targets do not cover exactly six candidates x four cells')
    for row in rows:
        r = analyzer.analyze(REPO / row['dump'], REPO / row['source'], 1)
        floor = max(r['cp_corrected_ns'], r['queue_lb_ns']) / 1e6
        measured = r['measured_l2_ms']
        ceiling = (floor + measured) / 2
        if floor > measured or any(abs(float(row[k])-v) > 1e-9 for k, v in
                                  (('floor_ms', floor), ('measured_A_ms', measured), ('target_ms', ceiling))):
            raise ValueError(f'{row["dump"]}: target does not reproduce from raw dump')
        print(f'TARGET {row["candidate"]}/{row["model"]}/s{row["seq"]}: floor={floor:.9f} measured_A={measured:.9f} target={ceiling:.9f} ms evidence={row["dump"]}',flush=True)
    for p, digest in frozen['inputs'].items():
        if hashlib.sha256((REPO / p).read_bytes()).hexdigest() != digest:
            raise ValueError(f'frozen input modified: {p}')
    return '24/24 candidate-specific targets reproduce; 0 targets below their own floor'


def correctness(raw, subset=False, chain=False):
    details = []
    for model in ('gqa2', 'mha4'):
        cells = ((1, 0), (128, 512), (2048, 0)) if subset else ((4, 3), (128, 3))
        for seq, past in cells:
            folder = (raw/'correctness'/f'{model}_s{seq}'/'chain' if chain else
                      raw/('seqscan' if subset else 'correctness')/f'{model}_s{seq}_p{past}')
            if len(list(folder.glob('r*.log'))) != 50:
                raise ValueError(f'{folder}: expected 50 fresh process logs')
            stamps = set()
            digests = set()
            for i in range(50):
                status, meta = result(folder/f'r{i}.log')
                if status!='PASS' or meta['exit_code'] or meta['round']!=i:
                    raise ValueError(f'{folder}/r{i}: correctness failed')
                stamps.add(meta['time_ns'])
                digests.add(meta['binary_sha256'])
            if len(stamps)!=50 or len(digests)!=1:
                raise ValueError(f'{folder}: duplicated process timestamps or mixed binaries')
            key=f'{model}_s{seq}_chain' if chain else f'{model}_p0_full'
            build=json.loads((raw/'log'/f'{key}.build.json').read_text())
            if digests!={build['binary_sha256']} or build.get('exit_code',0):
                raise ValueError(f'{folder}: correctness executable differs from the compiled configuration')
            details.append(f'{model}/s{seq}/p{past}=50/50')
    return '; '.join(details)


def realwidth(raw):
    configurations = ('baseline','c1','c2','window2','local2')
    sessions = set()
    stamps = set()
    for index, name in enumerate(configurations):
        for ai, arm in enumerate(('full','nofence','nowait','neither','l1nosync')):
            folder = raw/name/'paired/real_s4_p0'/arm
            if len(list(folder.glob('r*.log')))!=25:
                raise ValueError(f'{folder}: expected 25 logs')
            build=json.loads((raw/name/'log'/f'real_p0_{arm}.build.json').read_text())
            if build['exit_code']:raise ValueError(f'{name}/{arm}: build failed')
            configuration_flags(name,arm,build['command'])
            for i in range(25):
                status, meta = result(folder/f'r{i}.log')
                if meta['binary_sha256']!=build['binary_sha256'] or meta['time_ns'] in stamps:
                    raise ValueError(f'{folder}/r{i}: build mismatch or duplicated timestamp')
                stamps.add(meta['time_ns'])
                if arm=='full' and (status!='PASS' or meta['exit_code']):
                    raise ValueError(f'{folder}/r{i}: full failed')
                if meta['exit_code'] not in (0,1):
                    raise ValueError(f'{folder}/r{i}: unsafe arm did not complete normally')
                if meta['round']!=i or (i+meta['order'])%25!=index*5+ai:
                    raise ValueError(f'{folder}/r{i}: wrong rotation')
                if len(re.findall(r'^E2E_TIME ', (folder/f'r{i}.log').read_text(), re.M))!=1:
                    raise ValueError(f'{folder}/r{i}: no timing')
                sessions.add(meta['session'])
    if len(sessions)!=1:
        raise ValueError('real-width matrix mixes sessions')
    return '5 configurations x 5 arms x 25 fresh processes; all full PASS'


def configuration_flags(name, arm, command):
    defines={}
    for token in command:
        if token.startswith('-D'):
            key,_,value=token[2:].partition('=')
            defines[key]=value or '1'
    expected={'TILEMEGA_BARRIER_V2':1,'TILEMEGA_EVENT_SOLO':1,'TILEMEGA_EVENT_RED_PUBLISH':1,
              'TILEMEGA_WAIT_POLICY':1,'TILEMEGA_EVENT_KAPPA':1,
              'TILEMEGA_RELEASE_AFTER_BARRIER':int(name!='baseline'),
              'TILEMEGA_ASYNC_PUBLISH':int(name not in ('baseline','c1')),
              'TILEMEGA_LOCAL_DEP_SMEM':int(name.startswith('local')),
              'TILEMEGA_SLOT_WINDOW':int(name[-1]) if name.startswith(('window','local')) else 1,
              'TILEMEGA_UNSAFE_NO_NOTIFY_FENCE':int(arm=='nofence'),
              'TILEMEGA_UNSAFE_NO_EVENT_WAIT':int(arm in ('nowait','neither')),
              'TILEMEGA_UNSAFE_NO_EVENT_NOTIFY':int(arm=='neither'),
              'TILEMEGA_UNSAFE_NO_GRID_SYNC':int(arm=='l1nosync')}
    for key,value in expected.items():
        actual=defines.get(key,'1' if key=='TILEMEGA_SLOT_WINDOW' else '0')
        if actual!=str(value):raise ValueError(f'{name}/{arm}: {key}={actual}, expected {value}')


def matrix(raw):
    manifest=HERE/'matrix_manifest.json'
    specs = json.loads(manifest.read_text())
    session=json.loads((raw/'session.json').read_text())
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=session['manifest_sha256'] or specs!=session['configurations']:
        raise ValueError('measurement manifest changed after session start')
    rule=HERE/'research_rule.json'
    frozen=subprocess.check_output(['git','show','3c98f7f0:docs/experiments/SYNC_V3/research_rule.json'],cwd=REPO)
    committed=int(subprocess.check_output(['git','show','-s','--format=%ct','3c98f7f0'],cwd=REPO,text=True))
    if rule.read_bytes()!=frozen or int(session['session'])<committed*10**9:
        raise ValueError('research evaluation changed or was not fixed before the corrected session')
    sessions = set()
    for ci, name in enumerate(specs):
        five_arm(raw/name)
        for path in (raw/name/'paired').glob('*/*/r*.json'):
            meta = json.loads(path.read_text())
            cell = path.parent.parent.name
            placement = int(cell.rsplit('p',1)[1])
            arm = path.parent.name
            ai = ('full','nofence','nowait','neither','l1nosync').index(arm)
            combo = ci*10 + (5 if placement==5 else 0) + ai
            if (meta['round']+meta['order'])%(10*len(specs))!=combo:
                raise ValueError(f'{path}: cross-configuration order not rotated')
            if meta['binary_sha256']!=meta['build']['binary_sha256']:
                raise ValueError(f'{path}: executable differs from build')
            configuration_flags(name,arm,meta['build']['command'])
            sessions.add(meta['session'])
    if sessions!={session['session']}:
        raise ValueError('protocol matrix mixes sessions or differs from its manifest')
    return f'{len(specs)} configurations x 8 cells x 5 arms x 25 paired fresh processes'


def chain_pairs(raw):
    sessions = set()
    stamps = set()
    for model in ('gqa2','mha4','real'):
        for seq in (4,128):
            for ai, arm in enumerate(('rotate','chain','chain_control','original')):
                folder = raw/'paired'/f'{model}_s{seq}'/arm
                if len(list(folder.glob('r*.log')))!=25:
                    raise ValueError(f'{folder}: expected 25 logs')
                build=json.loads((raw/'log'/f'{model}_s{seq}_{arm}.build.json').read_text())
                for i in range(25):
                    status, meta = result(folder/f'r{i}.log')
                    if meta['binary_sha256']!=build['binary_sha256'] or meta['time_ns'] in stamps:
                        raise ValueError(f'{folder}/r{i}: build mismatch or duplicated timestamp')
                    stamps.add(meta['time_ns'])
                    if status!='PASS' or meta['exit_code'] or (i+meta['order'])%4!=ai:
                        raise ValueError(f'{folder}/r{i}: result or rotation failed')
                    if len(re.findall(r'^E2E_TIME ', (folder/f'r{i}.log').read_text(), re.M))!=1:
                        raise ValueError(f'{folder}/r{i}: missing timing')
                    sessions.add(meta['session'])
    if len(sessions)!=1:
        raise ValueError('chain matrix mixes sessions')
    return '6 cells x 4 arms x 25 paired fresh processes; all PASS'


def chain_paths(raw):
    details=[]
    candidates=('legacy_grid_stride','rotate','balanced','eft','wavefront','chain')
    for model in ('gqa2','mha4','real'):
        for seq in (4,128):
            counts={}
            for candidate in candidates:
                path=raw/'replay/path'/f'{model}_s{seq}_{candidate}_path.tsv'
                rows=list(csv.DictReader(path.open(),delimiter='\t'))
                if not rows:raise ValueError(f'{path}: empty path')
                hops=0
                for a,b in zip(rows,rows[1:]):
                    edge=a['edge_to_next']
                    if edge not in ('h','s','q'):raise ValueError(f'{path}: invalid edge')
                    if (a['worker']!=b['worker'])!=(edge=='h'):
                        raise ValueError(f'{path}: edge classification disagrees with placement')
                    hops+=edge=='h'
                counts[candidate]=hops
            stem=f'{model}_s{seq}_chain.cu'
            if (raw/'on/plan'/stem).read_bytes()!=(raw/'replay/plan'/stem).read_bytes():
                raise ValueError(f'{stem}: path replay differs from measured Plan')
            details.append(f'{model}/s{seq}: '+','.join(f'{c}={counts[c]}' for c in candidates))
            if model=='mha4' and seq==128 and counts['chain']>counts['rotate']:
                raise ValueError(details[-1])
    return '; '.join(details)


def chain_rejections(raw):
    # These are direct solver diagnostic records: each row is an exact-price
    # bucket of rejected edges, before any acceptance verdict is computed.
    path=raw/'on/rejected_extensions.tsv'
    counts={}
    for r in csv.DictReader(path.open(),delimiter='\t'):
        if float(r['hop_ns'])>float(r['queue_ns']) or int(r['count'])<=0:
            raise ValueError(f'{path}: rejection violates the frozen strict-price test')
        key=(r['model'],r['seq'],r['source'])
        counts[key]=counts.get(key,0)+int(r['count'])
    if not all((m,str(s),'cost_model') in counts for m in ('gqa2','mha4','real') for s in (4,128)):
        raise ValueError('missing rejection census cell')
    return '; '.join(f'{m}/s{s}/{source}={n}' for (m,s,source),n in counts.items())


def target_positions():
    configs=('baseline','c1','c2')
    candidates=('balanced','eft','wavefront','chain')
    sessions=set()
    for ci,config in enumerate(configs):
        for ai,candidate in enumerate(candidates):
            for model in ('gqa2','mha4'):
                for seq in (4,128):
                    folder=HERE/'target_positions'/config/'paired'/f'{model}_s{seq}_{candidate}'
                    if len(list(folder.glob('r*.log')))!=25:raise ValueError(f'{folder}: expected 25 logs')
                    for i in range(25):
                        status,meta=result(folder/f'r{i}.log')
                        if status!='PASS' or meta['exit_code'] or (i+meta['order'])%12!=ci*4+ai:
                            raise ValueError(f'{folder}/r{i}: failed correctness or rotation')
                        binary=Path(meta['command'][0])
                        build=json.loads((HERE/'target_positions'/config/'log'/(binary.name+'.build.json')).read_text())
                        expected=4 if candidate=='balanced' else 0
                        if f'-DTILEMEGA_PLACEMENT={expected}' not in build['command']:
                            raise ValueError(f'{folder}/r{i}: placement differs from frozen Plan')
                        if meta['binary_sha256']!=build['binary_sha256']:
                            raise ValueError(f'{folder}/r{i}: executable differs from build')
                        configuration_flags(config,'full',build['command'])
                        sessions.add(meta['session'])
    if len(sessions)!=1:raise ValueError('target candidates mix sessions')
    from metrics import timing
    import statistics
    for row in csv.DictReader((HERE/'targets.tsv').open(),delimiter='\t'):
        measurements={}
        for config in configs:
            if row['candidate'] in ('legacy_grid_stride','rotate'):
                placement=0 if row['candidate']=='legacy_grid_stride' else 5
                folder=HERE/'ablation'/config/'paired'/f'{row["model"]}_s{row["seq"]}_p{placement}'/'full'
            else:
                folder=HERE/'target_positions'/config/'paired'/f'{row["model"]}_s{row["seq"]}_{row["candidate"]}'
            measurements[config]=statistics.median(timing(folder/f'r{i}.log')['l2_ms'] for i in range(25))
        best=min(measurements,key=measurements.get)
        value=measurements[best]
        print(f'TARGET_POSITION {row["candidate"]}/{row["model"]}/s{row["seq"]}: '
              f'configuration={best} L2={value:.6f} target={float(row["target_ms"]):.6f} '
              f'delta={value-float(row["target_ms"]):.6f} ms',flush=True)
    return '24 fixed candidate/cells measured: 16 additional x 3 W=1 configurations x 25; legacy/rotate from protocol matrix'
