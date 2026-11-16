"""Strict immutable CPU-oracle cache loader; no online-reference fallback."""
from pathlib import Path
import hashlib,json
import torch
from safetensors.torch import load_file
RECIPE={'schema':1,'reference':'exported BF16 checkpoint promoted to FP32','threads':1,'deterministic_algorithms':True,'mkldnn_deterministic':True,'dtype_literal_promotion':'all BF16 args/kwargs to FP32 in FP32 graph','model':'nafnet'}
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(8<<20),b''):h.update(chunk)
 return h.hexdigest()
def load_reference_cache(cache,export,bridge,batch,inputs):
 cache,export,bridge=map(Path,(cache,export,bridge));m=json.loads((cache/'manifest.json').read_text())
 if inputs is None:raise ValueError('cached reference requires identified safetensors inputs')
 expected_recipe=hashlib.sha256(json.dumps(RECIPE,sort_keys=True,separators=(',',':')).encode()).hexdigest()
 if m['schema']!=1 or m['batch']!=batch or m['model']!='nafnet' or not m['cpu_only'] or m['cuda_initialized'] or m['recipe']!=RECIPE or m['recipe_sha256']!=expected_recipe:raise ValueError('reference cache recipe/domain mismatch')
 current={'exported_program':export/'exported_program.pt2','checkpoint':export/'checkpoint/model.safetensors','manifest':export/'manifest.json','bridge':bridge,'inputs':Path(inputs)}
 for k,p in current.items():
  if m['sources'][k]['sha256']!=sha(p):raise ValueError('reference cache source mismatch: '+k)
 torch_build={'version':torch.__version__,'init_sha256':sha(torch.__file__),'libtorch_cpu_sha256':sha(Path(torch.__file__).resolve().parent/'lib/libtorch_cpu.so')}
 if m['torch']!=torch_build:raise ValueError('reference cache torch build mismatch')
 refpath=cache/'reference.safetensors'
 if m['reference_file_sha256']!=sha(refpath):raise ValueError('reference cache tensor file mismatch')
 values=load_file(str(refpath),device='cpu');n=len(values)//2
 if n<1 or set(values)!={f'{kind}/{i}' for kind in ('fp32','bf16') for i in range(n)}:raise ValueError('reference cache output tree mismatch')
 for k,v in values.items():
  r=m['references'][k]
  if r['shape']!=list(v.shape) or r['dtype']!=str(v.dtype) or r['sha256']!=hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest():raise ValueError('reference cache output identity mismatch: '+k)
  if v.shape[0]!=batch or v.dtype!=(torch.float32 if k.startswith('fp32/') else torch.bfloat16):raise ValueError('reference cache output precision or batch mismatch')
 return [values[f'bf16/{i}'] for i in range(n)],[values[f'fp32/{i}'] for i in range(n)]
