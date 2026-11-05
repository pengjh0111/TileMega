"""Inventory actual typed register-chain Apply calls in generated model plans."""
from pathlib import Path
import argparse,json,re,hashlib,itertools,sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent

def args(text):
 out=[];start=0;depth=0
 for i,c in enumerate(text):
  depth+=(c=='<')-(c=='>')
  if c==',' and depth==0:out.append(text[start:i]);start=i+1
 out.append(text[start:]);return out

def template(text,start):
 angle=text.index('<',start);depth=1;end=angle+1
 while depth:depth+=(text[end]=='<')-(text[end]=='>');end+=1
 return text[start:end],args(text[angle+1:end-1])

def inventory(paths):
 keys={};sources={};excluded={};unknown=[];instances=0
 for path in paths:
  text=path.read_text();sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
  for match in re.finditer(r'^using Body_(\d+)=(.*);$',text,re.M):
   stage=int(match[1]);body=match[2];at=body.find('Sm90ForwardEpilogueChain<')
   if at<0:continue
   cpp,chain=template(body,at);m,n,write,shuffle,cw,ch,pitch=map(int,chain[:7]);steps=[];skip=None
   bind=re.search(r'struct Bind_'+str(stage)+r' \{(.*?)\n\};',text,re.S)[1]
   mask=sum((re.search(r'kVector'+str(v)+r'=(\w+)',bind)[1]!='kAbsent')<<v for v in range(4))
   residual=False;shuffled_residual=False
   for step in chain[7:]:
    _,a=template(step,0);kind=a[0].split('::')[-1];v=int(a[1]) if len(a)>1 else 0
    if kind in ('kScale','kBias'):steps.append(kind[1:].lower()+str(v))
    elif kind=='kResidual':
     residual=True;residual_map=int(a[6]) if len(a)>6 else 0;shuffled_residual|=residual_map==2;steps.append('residual'+str(v)+'map'+str(residual_map))
    elif kind=='kActivation' and a[2].split('::')[-1] in ('kRelu','kRelu6'):steps.append(a[2].split('::')[-1][1:].lower())
    else:skip=kind
   if skip or len(steps)<2:excluded[skip or 'single_or_empty']=excluded.get(skip or 'single_or_empty',0)+1;continue
   shape=re.search(r'struct Shape_'+str(stage)+r' \{(.*?)\n\};',text,re.S)[1]
   shape={k:int(v) for k,v in re.findall(r'(k\w+)=(\d+)',shape)}
   resid_stride=shape['kResidualStride'] if residual else 0
   name=body.split('<',1)[0];_,ba=template(body,0)
   if name in ('Sm90ForwardGemm','Sm90ForwardGemm32'):
    rows=int(ba[6]);cols=int(ba[9]);rowclasses=sorted({min(m,rows),rows%m}-{0});colclasses=sorted({min(n,cols),cols%n}-{0});geometry=[(r,c,0,0) for r,c in itertools.product(rowclasses,colclasses)]
   elif name in ('Sm90ForwardDepthwise','Sm90ForwardDwPw','Sm90ForwardInvertedResidual'):
    tw,th=map(int,ba[:2]);w,h=shape['kOutputWidth'],shape['kOutputHeight'];cols=shape['kOutChannels']
    if name=='Sm90ForwardDepthwise':cols=shape['kChannels']
    widths=sorted({min(tw,w),w%tw}-{0});heights=sorted({min(th,h),h%th}-{0});colclasses=sorted({min(n,cols),cols%n}-{0});geometry=[(lw*lh,c,lw,w) for lw,lh,c in itertools.product(widths,heights,colclasses)]
   else:unknown.append(dict(source=str(path),stage=stage,body=name));continue
   for rows,columns,tilewidth,imagewidth in geometry:
    key=dict(tm=m,tn=n,live_rows=rows,live_columns=columns,chain='+'.join(steps),vector_mask=mask,
      write=write,shuffle=shuffle,image_width=cw if shuffled_residual else 0,image_height=ch if shuffled_residual else 0,
      residual_stride=resid_stride,tile_width=tilewidth if residual else 0,residual_image_width=imagewidth if residual else 0,
      residual_row_base=0,channel=0)
    keytext=';'.join(k+'='+str(v) for k,v in key.items());entry=keys.setdefault(keytext,dict(key=key,chain_cpp=cpp,instances=[]));entry['instances'].append(dict(source=str(path),stage=stage,body=name));instances+=1
 return dict(status='CPU inventory only; no measured chain costs',metric='epilogue_chain_apply_ns',unit='ns',keys=list(keys.values()),key_count=len(keys),instances=instances,source_sha256=sources,excluded_chain_instances=excluded,unsupported=unknown,
  scope='Actual emitted plan choices only. First channel/row/tile is an explicit structural transfer for otherwise equal memory mappings; all observed live-row/live-column and spatial-stride classes retained. Does not cover every future solver choice. Composite Apply excludes Finish/store/rowstats/pool; never add older constituent Apply costs.')

def generate(a):
 from dnn1_forward_recipe_probe import catalog_spans
 a.out.mkdir(parents=True,exist_ok=False);catalog=json.loads(a.required.read_text());source=a.base_source.read_text();body=(HERE/'dnn1_chain_apply_probe.cuh').read_text();registrations=[]
 for i,row in enumerate(catalog['keys']):
  k=row['key'];assert k['write'] in (0,1) and not k['image_width'] and not k['image_height'] and not k['residual_row_base'] and not k['channel'],'shuffle/full-offset requires separate mapped probe';body+=f'using ApplyChain{i}={row["chain_cpp"]};\n';code=[]
  for step in k['chain'].split('+'):
   if step.startswith('scale'):code+=['value*=0.5f;']
   elif step.startswith('bias'):code+=['value+=0.5f;']
   elif step=='relu':code+=['value=std::max(value,0.0f);']
   elif step=='relu6':code+=['value=std::min(std::max(value,0.0f),6.0f);']
   elif step.startswith('residual'):
    v=int(step[len('residual')]);
    if k['vector_mask']&(1<<v):code+=['{volatile float product=value*0.5f;value=product;}']
    code+=['value+=0.25f;']
   else:raise ValueError(step)
  key=';'.join(n+'='+str(v) for n,v in k.items());template=[f'ApplyChain{i}']+[str(k[n]) for n in ('tm','tn','live_rows','live_columns','vector_mask','residual_stride','tile_width','residual_image_width')]
  registrations.append(f'  ChainApplyCase<{",".join(template)}>({i},{json.dumps(key)},[](float value){{{"".join(code)}return value;}});')
 body+='\nvoid Catalog() {\n'+'\n'.join(registrations)+'\n}\n'
 for i,(start,end) in reversed(list(enumerate(catalog_spans(source)))):source=source[:start]+(body if i==0 else 'void Catalog() {}')+source[end:]
 output=a.out/'dnn1_constants.cu';output.write_text(source);sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();(a.out/'preparation.json').write_text(json.dumps(dict(status='CPU only; real compositeApply measurements pending',keys=len(registrations),required_sha256=sha(a.required),source_sha256=sha(output),production_epilogue_sha256=sha(a.source_root/'include/tilemega/Backend/Sm90ForwardEpilogue.cuh'),scope=catalog['scope'],limitations=['actual plans only; future solver geometries not covered','row/channel0 representative structural transfer; not every cache address','shuffled residual and nonzero offsets require a separate mapped probe and are rejected']),indent=2)+'\n')
 if a.build:
  import dnn1_constants as c;c.SOURCE=output
  c.build(argparse.Namespace(out=a.out/'build',source_root=a.source_root,implementation='fixed',nvcc='/usr/local/cuda/bin/nvcc',timeout_s=1200))


def main():
 p=argparse.ArgumentParser(description=__doc__)
 sub=p.add_subparsers(dest='command',required=True)
 keys=sub.add_parser('keys')
 keys.add_argument('--generated-cu',type=Path,nargs='+',required=True)
 keys.add_argument('--out',type=Path,required=True)
 gen=sub.add_parser('generate')
 gen.add_argument('--required',type=Path,required=True)
 gen.add_argument('--source-root',type=Path,required=True)
 gen.add_argument('--base-source',type=Path,default=HERE/'dnn1_constants.cu')
 gen.add_argument('--out',type=Path,required=True)
 gen.add_argument('--build',action='store_true')
 a=p.parse_args()
 if a.command=='generate':generate(a);return
 result=inventory(a.generated_cu)
 if result['unsupported']:raise ValueError('unsupported typed producer: '+str(result['unsupported']))
 a.out.parent.mkdir(parents=True,exist_ok=True)
 with a.out.open('x') as stream:json.dump(result,stream,indent=2);stream.write('\n')
 print(json.dumps({k:result[k] for k in ('key_count','instances','excluded_chain_instances','unsupported')}))


if __name__=='__main__':main()
