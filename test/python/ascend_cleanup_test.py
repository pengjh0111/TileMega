"""Exercise AS-1 destructive cleanup only on a new private fixture tree."""
import importlib.util
import json
import os
from pathlib import Path
import time
import unittest

REPO=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('as1_cleanup',REPO/'scripts/as1_cleanup.py')
cleanup=importlib.util.module_from_spec(spec)
spec.loader.exec_module(cleanup)


class AscendCleanupTest(unittest.TestCase):
    def test_manifest_and_path_boundaries(self):
        fixture=cleanup.DEFAULT_WORK/'tmp'/('cleanup_contract_'+str(time.time_ns()))
        allowed=fixture/'build'
        allowed.mkdir(parents=True)
        names=['old','keep','selected','failure','incomplete','active','changed',
               'recent1','recent2','hardlink','symlink','outside']
        entries=[]
        value=Path('/proc',str(os.getpid()),'stat').read_text()
        active=dict(pid=os.getpid(),start_ticks=int(value[value.rfind(')')+2:].split()[19]))
        for index,name in enumerate(names):
            p=(fixture if name=='outside' else allowed)/name
            if name=='hardlink':
                os.link(allowed/'keep',p)
            elif name=='symlink':
                p.symlink_to(allowed/'keep')
            else:
                p.write_text(name)
            entry=dict(path=str(p),category='build',completed=True,regenerable=True,
                keep=False,active_refs=[],stamp=cleanup.file_stamp(p),
                build_id=name,created_at='2026-01-01T00:00:'+f'{index:02d}')
            if name=='keep':entry['keep']=True
            if name=='selected':entry['selected']=True
            if name=='failure':entry['category']='failure'
            if name=='incomplete':entry['completed']=False
            if name=='active':entry['active_refs']=[active]
            if name=='changed':p.write_text('changed since registration')
            if name.startswith('recent'):entry['created_at']='2027-'+name
            entries.append(entry)
        (fixture/'fixture_manifest.json').write_text(json.dumps(entries,indent=2)+'\n')
        roots=[allowed]
        items=cleanup.plan(entries,roots,keep_recent=2)
        # Planning must leave every file intact, including the removable one.
        self.assertTrue(all(Path(e['path']).exists() for e in entries))
        removable=[i for i in items if i['action']=='remove']
        self.assertEqual([Path(i['path']).name for i in removable],['old'])
        cleanup.apply_item(removable[0],roots)
        self.assertFalse((allowed/'old').exists())
        self.assertTrue(all(Path(e['path']).exists() for e in entries if Path(e['path']).name!='old'))
        result=dict(status='passed',boundary_cases=len(entries),dry_run_deleted=0,
            apply_removed=1,protected_survivors=len(entries)-1,fixture=str(fixture))
        (fixture/'result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result))


if __name__=='__main__':
    unittest.main()
