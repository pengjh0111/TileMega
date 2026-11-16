"""Validate retained P-ABL gates without changing a live collector's source."""
from pathlib import Path
from sm90d1_common import read,sha
from sm90r6_ablate import ARMS


def validate_subset(arms,model,batch,excluded=()):
    if len(excluded)>1 or not set(excluded)<={'AB1a','AB1b'}:
        raise ValueError('only one unavailable AB1 arm may be excluded')
    expected=tuple(arm for arm in ARMS if arm not in excluded)
    if set(arms)!=set(expected):raise ValueError('P-ABL arm domain disagrees with declared exclusions')
    result={};identities={}
    for label in expected:
        arm=arms[label];binary=Path(arm['binary']).resolve();digest=sha(binary)
        identity=read(str(binary)+'.identity.json')
        identities[label]=identity
        if identity['binary_sha256']!=digest:raise ValueError('changed binary: '+label)
        c2=read(arm['c2'])
        if (not c2['passed'] or len(c2['processes'])<5 or c2['steps']!=1024 or
                c2['model']!=model or c2['batch']!=batch or
                c2['artifacts']['candidate']['sha256']!=digest):
            raise ValueError('missing winner C-2 evidence: '+label)
        if label in ('AB4b','AB4c'):
            c1=read(arm['c1']);spec=c1['spec']
            if (not c1['pass'] or not c1['c1_pass'] or spec['correctness']['c1']!=256 or
                    spec['model']!=model or spec['batch']!=batch or
                    {a['decode_sha256'] for a in spec['arms'].values()}!={digest}):
                raise ValueError('missing 256-position C-1 evidence: '+label)
        result[label]=dict(decode=str(binary),prefill='prompt')
    if read(arms['AB4c']['c2'])['artifacts']['reference']['sha256']!=sha(arms['AB4b']['binary']):
        raise ValueError('AB4c must match the selected AB4b megakernel')
    reference,candidate=identities['AB4b'],identities['AB4c']
    if (candidate['plan'].get('execution')!='kernel_sequence' or
            any(reference['plan'][key]!=candidate['plan'][key] for key in
                ('native_dense','export_sha256','grid','runtime_batch')) or
            reference['shared_memory_bytes']!=candidate['shared_memory_bytes']):
        raise ValueError('AB4c changed the selected AB4b coordinates or launch geometry')
    return result

