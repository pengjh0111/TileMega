#!/usr/bin/env python3
"""Read captured public configs and audit current architecture coverage.
No weights, HF runtime, or model downloads are needed. Source hashes remain
in sources/manifest.json; this script never substitutes remembered dimensions.
"""
import ast,csv,json,re
from pathlib import Path
HERE=Path(__file__).resolve().parent;REPO=HERE.parents[2]
def write(name,rows):
    with (HERE/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t',lineterminator='\n');w.writeheader();w.writerows(rows)
def main():
    configs={n:json.loads((HERE/'sources'/p).read_text()) for n,p in [('llama','llama_config_public_copy.json'),('qwen','qwen_config.json')]}
    module=ast.parse((HERE/'sources/meta_sku_list.py').read_text())
    family=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='llama3_2_base_models')
    entry=next(n for n in ast.walk(family) if isinstance(n,ast.Call) and any(k.arg=='huggingface_repo' and isinstance(k.value,ast.Constant) and k.value.value=='meta-llama/Llama-3.2-1B' for k in n.keywords))
    args=next(k.value for k in entry.keywords if k.arg=='arch_args')
    meta={ast.literal_eval(k):ast.literal_eval(v) for k,v in zip(args.keys,args.values) if isinstance(v,ast.Constant)}
    c=configs['llama']
    for key,mkey in [('hidden_size','dim'),('num_hidden_layers','n_layers'),('num_attention_heads','n_heads'),('num_key_value_heads','n_kv_heads'),('rms_norm_eps','norm_eps'),('rope_theta','rope_theta')]:assert c[key]==meta[mkey]
    inner=int(meta['ffn_dim_multiplier']*int(2*(4*meta['dim'])/3))
    inner=meta['multiple_of']*((inner+meta['multiple_of']-1)//meta['multiple_of'])
    assert inner==c['intermediate_size']
    names=['num_hidden_layers','hidden_size','intermediate_size','num_attention_heads','num_key_value_heads','head_dim','rope_theta','rope_scaling','rms_norm_eps','vocab_size','tie_word_embeddings','max_position_embeddings']
    write('dimensions.tsv',[dict(model=n,**{k:json.dumps(c.get(k,c['hidden_size']//c['num_attention_heads'] if k=='head_dim' else None)) for k in names}) for n,c in configs.items()])
    text=(REPO/'include/tilemega/Codegen/tasks/TaskBase.h').read_text().split('enum class TaskKind')[1].split('};')[0]
    kinds=re.findall(r'\b(k\w+)\s*=\s*\d+',text)
    epsilon=float(re.search(r'rsqrtf\(rms\[0\] / hidden \+ ([\deE.+-]+)f', (REPO/'include/tilemega/Codegen/tasks/RMSNormTaskBody.h').read_text()).group(1))
    rows=[]
    entries=[
      ('token embedding','missing','no indexed embedding TaskKind / decoder importer input is hidden states'),
      ('input pre-attention RMSNorm','kRMSNorm','epsilon'),
      ('Q/K/V projections','kGemm','shape_supported; exact export path pending'),
      ('per-head Q/K RMSNorm','missing','Qwen only; existing RMSNorm owns one token row, not one token/head'),
      ('RoPE frequency scaling','kRoPE','config-dependent frequency table can be an input buffer'),
      ('RoPE rotation','kRoPE','partial: backend rounds positions and angles to ModelElement before sin/cos; public implementation uses FP32 angles'),
      ('KV cache append','kKVAppend','shape_supported'),
      ('GQA score/mask/softmax/value','kAttention','shape_supported; composite implementation, precise export check pending'),
      ('output projection','kGemm','shape_supported'),
      ('attention residual','kAdd / kGemm epilogue','shape_supported'),
      ('post-attention RMSNorm','kRMSNorm','epsilon'),
      ('gate/up projections','kGemm','shape_supported'),
      ('SiLU times gate','kElementwise','shape_supported'),
      ('down projection and residual','kGemm / kAdd','shape_supported'),
      ('final RMSNorm','kRMSNorm','partial: no standalone final-normalization stage in DecoderLayerPattern; epsilon'),
      ('vocabulary projection','kGemm','partial: collective can implement dimensions; decoder-only importer does not emit final LM head'),
      ('reshape/transpose/broadcast','layout metadata','absorbed by semantic access maps; no standalone task')]
    for model,c in configs.items():
      for op,kind,status in entries:
        if op=='per-head Q/K RMSNorm' and model=='llama':continue
        status=status.replace('epsilon',f'configured epsilon={c["rms_norm_eps"]}; backend epsilon={epsilon}; '+('parameter gap' if c['rms_norm_eps']!=epsilon else 'matches'))
        rows.append(dict(model=model,operator=op,task_kind=kind,status=status))
    write('coverage.tsv',rows)
    sites=[
      ('TaskKind enum','include/tilemega/Codegen/tasks/TaskBase.h','new distinct TaskKind'),
      ('OwnershipOf','include/tilemega/Codegen/tasks/TaskBase.h','new ownership classification'),
      ('TaskBody implementation','include/tilemega/Codegen/tasks/','new backend body'),
      ('resource traits and resource query','include/tilemega/Codegen/tasks/TaskResources.h','threads/shared source'),
      ('backend scalar dataflow','include/tilemega/Codegen/tasks/ScalarDataflow.h','control-flow latency declaration'),
      ('L1 dispatch','include/tilemega/Codegen/tasks/ModelHarness.cuh','RunStage'),
      ('L2 dispatch','include/tilemega/Codegen/tasks/ModelHarness.cuh','RunTask'),
      ('active ownership dispatch','include/tilemega/Codegen/tasks/ModelHarness.cuh','ActiveTaskCount'),
      ('frontend plan role','include/tilemega/Frontend/SemanticLifting.h','plan role and operator semantics'),
      ('frontend pattern','lib/Frontend/ModelPlan.cpp','recognize actual use-def subgraph'),
      ('semantic access lifting','lib/Frontend/SemanticLifting.cpp','read/write/reduction relations'),
      ('frontend runtime spelling','lib/Frontend/Frontend.cpp','PlanTaskKind to emitted TaskKind'),
      ('model stage representation','include/tilemega/Solver/ModelDescription.h','new kind identity, no prices'),
      ('model stage parser','lib/Solver/ModelDescription.cpp','new kind spelling, no prices'),
      ('ownership adapter if new map','lib/Solver/ScalarTaskWork.cpp','conditional for a new physical ownership relation')]
    write('extension_sites.tsv',[dict(index=i+1,site=s,path=p,condition=c) for i,(s,p,c) in enumerate(sites)])
    (HERE/'audit.json').write_text(json.dumps(dict(status='static_audit_verified; end_to_end_pending',task_kind_count=len(kinds),task_kinds=kinds,new_distinct_operator_sites=len(sites),generic_cost_formula_edits=0,derivation='DeriveTaskWork and TaskInstanceNs consume semantic access and backend traits; ownership adapter may still need extension',llama_config_provenance='public redistributed config with primary Meta SKU dimensional cross-check; official HF raw config returned401',normalization_position='pre-attention; pre-MLP; final model norm; Qwen additionally per-head Q/K norm before RoPE',maximal_subset_status='not yet established by executable export; dimension-only decoder is not full architecture coverage'),indent=2)+'\n')
    print(f'MODEL_AUDIT PASS configs=2 task_kinds={len(kinds)} extension_sites={len(sites)} generic_cost_formula_edits=0 end_to_end=pending')
if __name__=='__main__':main()
