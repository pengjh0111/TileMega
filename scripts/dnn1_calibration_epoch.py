"""Build a new calibration epoch without changing any source recipe or target."""
from pathlib import Path
import sys,json,csv,copy,hashlib,shutil,argparse
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
import dnn1_constants as c
from dnn1_compact_constants import convert
RECIPE_MODES = [('base','base'),('live16','append'),('float4','replace'),('tm32','append'),
 ('mma32_math','append'),('dwpw','append'),('residual_scaled','replace'),('gated_dw','append'),
 ('global_pool','replace'),('global_pool_unique','append'),('pool_finish','append'),
 ('rowstat_finish','append'),('maxpool_flat','replace'),('sca_panel_64_128','append'),
 ('sca_panel_32','append')]
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--base-target',type=Path,default=R/'configs/targets/sm_90_h800.json')
p.add_argument('--recipe',action='append',required=True,metavar='NAME=COLLECTED_DIRECTORY')
p.add_argument('--expected-keys',type=int,default=1006)
a=p.parse_args();inputs={}
for entry in a.recipe:
 name,path=entry.split('=',1)
 if name in inputs:raise ValueError('duplicate recipe '+name)
 inputs[name]=Path(path)
if set(inputs)!={name for name,mode in RECIPE_MODES}:raise ValueError('all fifteen named recipes required')
O=a.out.resolve();O.mkdir(parents=True,exist_ok=False)
base_target=a.base_target;before=base_target.read_bytes()
marker=',\n  "sm90_forward_constants": '
text=before.decode()
if marker in text:prefix=text.split(marker,1)[0]+'\n}\n'
else:prefix=text
assert 'sm90_forward_constants' not in json.loads(prefix)
recipes=[(name,mode,inputs[name]) for name,mode in RECIPE_MODES]
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sections={};tables={};originals={};source_proofs=[]
for name,mode,source in recipes:
 source=source.resolve();d=O/'recipes'/name;d.mkdir(parents=True)
 for filename in ('sm90_forward_constants.json','DNN1_constants.csv','collection_receipt.json'):
  shutil.copyfile(source/filename,d/filename);assert sha(source/filename)==sha(d/filename)
 raw=json.loads((d/'sm90_forward_constants.json').read_text());spec=json.loads((Path(raw['raw_measurement_root'])/'queue-spec.json').read_text());c.verify_build(Path(spec['build']));compact,rows=convert(d/'sm90_forward_constants.json',d/'DNN1_constants.csv',allow_subset=mode!='base')
 target=d/'target.json';target.write_text(prefix);c.append_forward_section(target,compact)
 sections[name]=compact;tables[name]=rows;originals[name]=raw
 source_proofs.append(dict(name=name,mode=mode,source=str(source),raw_sha256=sha(d/'sm90_forward_constants.json'),original_recipe_target=str(target),original_recipe_target_sha256=sha(target),variant=raw['variant'],entries=sum(len(v) for g in raw['by_gpu'].values() for v in g.values())))
 # Decode every independent view before assembling the epoch.
 restored=json.loads(target.read_text())['sm90_forward_constants']
 assert all(restored['median_by_gpu'][g][i]==raw['by_gpu'][g][m][k]['median'] for g in ('0','5','7') for i,(m,k,u) in enumerate(restored['keys']))
combined=copy.deepcopy(sections['base']);rows=copy.deepcopy(tables['base']);origin=['base']*len(combined['keys']);indices={tuple(k):i for i,k in enumerate(combined['keys'])};combined['supplements']=[];replaced=[]
for name,mode,_ in recipes[1:]:
 extra=sections[name];ids=[]
 for j,key in enumerate(extra['keys']):
  key=tuple(key)
  if mode=='replace':
   assert key in indices and origin[indices[key]]=='base';i=indices[key];replaced.append(dict(index=i,key=key,old_recipe=origin[i],new_recipe=name,old_medians={g:combined['median_by_gpu'][g][i] for g in ('0','5','7')},new_medians={g:extra['median_by_gpu'][g][j] for g in ('0','5','7')}))
  else:
   assert key not in indices;i=len(combined['keys']);indices[key]=i;combined['keys'].append(list(key));origin.append(name);rows.append({})
   for g in ('0','5','7'):combined['median_by_gpu'][g].append(0)
  origin[i]=name;ids.append(i)
  for g in ('0','5','7'):combined['median_by_gpu'][g][i]=extra['median_by_gpu'][g][j]
  rows[i]=dict(tables[name][j],constant_id=i,clock_group=name+'/'+tables[name][j]['clock_group'])
 metric_keys={(m,k) for m,k,u in extra['keys']};combined['unstable']=[r for r in combined['unstable'] if (r['metric'],r['key']) not in metric_keys];combined['unstable'].extend(extra['unstable'])
 for g in ('0','5','7'):
  for group,clocks in extra['clock_groups'][g].items():combined['clock_groups'][g][name+'/'+group]=clocks
 delta={k:v for k,v in extra['header_sha256'].items() if sections['base']['header_sha256'].get(k)!=v}
 metadata=dict(variant=extra['variant'],raw_sources=extra['raw_sources'],key_indices=ids,scope=mode+' exact keys in this new epoch only; original recipe views unchanged',header_source='base header_sha256 with delta below; complete dictionary in SHA-bound raw source',header_sha256_delta=delta,probe_sha256=extra['probe_sha256'],clock_overlay=extra['clock_overlay'],method_source='raw_sources/sm90_forward_constants.json#method',process_scope=extra['process_scope'])
 combined['supplements'].append(metadata)
identity={r['name']:r['raw_sha256'] for r in source_proofs};combined['base_calibration_variant']=combined['variant'];combined['variant']=hashlib.sha256(json.dumps(identity,sort_keys=True,separators=(',',':')).encode()).hexdigest();combined['calibration_epoch']=dict(source_recipes=identity,scope='exact measured composition of all15 explicit source recipes; no averaging, interpolation, scaling or fitting',replaced_base_keys=len(replaced),uncovered_implementation=['NAF gated DW epilogue diagnostic excludes pool side-output; not an additive whole-ep cost','Semantic mathleaf acceptance is separate; this epoch has isolated primitives, no invented overlap coefficient','Unique pool Finish/store calibrated at exact keys; consuming cost model must replace ordinary store, not add both','Original rowstat_ns has width=0 and remains historical; full rowstat_finish_store_ns includes local sum/squares, cross-tile reduction and store and must replace both old rowstat and store','SCA actual ScalePanel primitives cover all six TM/liveK keys; production model correctness and E2E gates remain separate; full64 higherK and Rows transfer is explicitly structural, not measured every address']);combined['method']=dict(combined['method'],epoch='Every key uses its explicit recipe; supplements also identify replacements; preserved original recipe targets are SHA-bound in epoch receipt')
assert len(combined['keys'])==a.expected_keys and len(rows)==a.expected_keys
for i,(m,k,u) in enumerate(combined['keys']):
 for g in ('0','5','7'):
  raw=originals[origin[i]]['by_gpu'][g][m][k];assert combined['median_by_gpu'][g][i]==raw['median'] and u==raw['unit'] and rows[i]['gpu'+g+'_median']==raw['median']
target=O/'target.json';target.write_text(prefix);c.append_forward_section(target,combined)
with (O/'DNN1_constants.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
assert base_target.read_bytes()==before
loaded=json.loads(target.read_text())['sm90_forward_constants'];assert loaded['keys']==[list(k) for k in combined['keys']] and loaded['median_by_gpu']==combined['median_by_gpu']
proof=dict(status='new_epoch_exact_composition',source_recipes=source_proofs,variant=combined['variant'],keys=len(rows),exact_medians=3*len(rows),replacement_details=replaced,key_recipe_by_index=origin,base_target=str(base_target),base_target_sha256=sha(base_target),base_target_unchanged=True,new_target_sha256=sha(target),new_csv_sha256=sha(O/'DNN1_constants.csv'),source_script_sha256=sha(__file__),section_bytes=len(json.dumps(combined,separators=(',',':')).encode()),csv_bytes=(O/'DNN1_constants.csv').stat().st_size,uncovered=combined['calibration_epoch']['uncovered_implementation']);(O/'epoch_receipt.json').write_text(json.dumps(proof,indent=2)+'\n');print(json.dumps({k:proof[k] for k in ('status','keys','exact_medians','section_bytes','csv_bytes')}),flush=True)
