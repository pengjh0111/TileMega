"""Deterministic MX-7 TP stage ranking; qualification and GPU execution are separate."""
import heapq
import math

STAGES=('qkv','attention','o','gate_up','down','head')
LENGTHS=(65,256,512,768,1088)


def positive(value):
    if type(value)not in(int,float)or not math.isfinite(value)or value<=0:
        raise ValueError('costs must be finite positive numbers')
    return float(value)


def attention_mean(values):
    if len(values)!=5:raise ValueError('attention requires all five length points')
    values=list(map(positive,values));point=0;total=0.
    for length in range(65,1088):
        while length>LENGTHS[point+1]:point+=1
        fraction=(length-LENGTHS[point])/(LENGTHS[point+1]-LENGTHS[point])
        total+=values[point]+fraction*(values[point+1]-values[point])
    return total/1023


def communication_cost(constants,stage,tp,batch,tn=None):
    name='t_head_x' if stage=='head' else 't_ar'
    matches=[]
    for row in constants[name]:
        key=row['key']
        if key['tp']==tp and key['b']==batch and (stage=='head' or
                (key['stage']==stage and key['tn']==tn)):
            matches.append(row)
    if len(matches)!=1:
        raise ValueError('one measured communication constant required per key')
    row=matches[0]
    if row['processes']not in(3,5):raise ValueError('communication cost requires three/five fresh processes')
    value=row['p90_ns']if row.get('noisy')else row['median_ns']
    if type(value)not in(int,float)or not math.isfinite(value):raise ValueError('finite communication delta required')
    return float(value)


def stage_lists(rows,model,batch,tp_constants):
    if model not in('qwen3_1_7b_tp2','qwen3_1_7b_tp4'):
        raise ValueError('unsupported TP model scope')
    tp=int(model.rsplit('tp',1)[1])
    result={name:[]for name in STAGES};attention={}
    for index,row in enumerate(rows):
        key=row['key']
        if key['model']!=model or key['b']!=batch:continue
        if row['processes']not in(3,5) or row['resources']['resident']!=1 or row['resources']['stack']or row['resources']['static_smem']:
            raise ValueError('cost row lacks three/five processes or qualified resources')
        if not 0<row['resources']['MTreg']<=255:raise ValueError('cost row exceeds512-thread register cap')
        cost=positive(row['p90_ns']if row.get('noisy')else row['median_ns']);stage=key['stage']
        if stage not in result:raise ValueError('unknown stage')
        if stage=='attention':
            if key.get('attn_version')!=2:raise ValueError('MX7 requires attention V2')
            identity=(key['prefetch'],key['b_min'])
            group=attention.setdefault(identity,{})
            if key['length']in group:raise ValueError('duplicate attention grid point')
            group[key['length']]=(cost,index,key)
        else:
            communication=communication_cost(tp_constants,stage,tp,batch,key['tn']) if stage in('o','down','head')else 0.
            result[stage].append(dict(ns=positive(cost+communication),base_ns=cost,
                                      communication_ns=communication,key=key,source_rows=[index]))
    for identity,grid in attention.items():
        if set(grid)!=set(LENGTHS):raise ValueError('incomplete attention length grid')
        result['attention'].append(dict(ns=attention_mean([grid[n][0]for n in LENGTHS]),
            key={k:v for k,v in grid[65][2].items()if k!='length'},source_rows=[grid[n][1]for n in LENGTHS]))
    for stage,choices in result.items():
        if not choices:raise ValueError('empty measured stage domain: '+stage)
        choices.sort(key=lambda row:(row['ns'],tuple(row['source_rows'])))
    return result


def k_best(lists,layers):
    if type(layers)is not int or layers<1:raise ValueError('positive layer count required')
    ordered=[lists[name]for name in STAGES]
    if any(not rows for rows in ordered):raise ValueError('empty stage domain')
    for rows in ordered:
        costs=[positive(row['ns'])for row in rows]
        if costs!=sorted(costs):raise ValueError('stage domain must be sorted')
    factors=(layers,layers,layers,layers,layers,1)
    def score(ranks):return sum(ordered[i][rank]['ns']*factors[i]for i,rank in enumerate(ranks))
    initial=(0,)*6;pending=[(score(initial),initial)];seen={initial};rank=0
    while pending:
        predicted,ranks=heapq.heappop(pending);rank+=1
        yield dict(candidate_rank=rank,predicted_ns=predicted,ranks=dict(zip(STAGES,ranks)),
            stages={name:ordered[i][ranks[i]]for i,name in enumerate(STAGES)})
        for stage in range(6):
            next_ranks=list(ranks);next_ranks[stage]+=1;next_ranks=tuple(next_ranks)
            if next_ranks[stage]<len(ordered[stage])and next_ranks not in seen:
                seen.add(next_ranks);heapq.heappush(pending,(score(next_ranks),next_ranks))


def pin(choice,cost_sha,tp,purpose='resource_probe'):
    if len(cost_sha)!=64 or any(c not in'0123456789abcdef'for c in cost_sha):raise ValueError('invalid cost table SHA256')
    if purpose not in('manual','resource_probe')or tp not in(2,4):raise ValueError('invalid TP pin purpose or degree')
    stages=choice['stages'];attn=stages['attention']['key']
    if any(stages[name]['key']['tp']!=tp for name in STAGES):raise ValueError('mixed TP scopes in choice')
    config=dict(threads=512,purpose=purpose,attn_version=2,split=1,b_min=attn['b_min'],
                attn_prefetch=attn['prefetch'],default_enabled=False,diagnostic=True,final=False,solver_selected=False)
    for stage in('qkv','o','gate_up','down','head'):
        key=stages[stage]['key'];config[stage]={name:key[name]for name in('tn','wn','d','prefetch','r_next')}
        config[stage]['part']='kTileRange'if key['part']else'kStreamK'
    evidence=dict(tp=tp,cost_table_sha256=cost_sha,candidate_rank=choice['candidate_rank'],
                  predicted_ns=positive(choice['predicted_ns']),
                  source_rows={stage:stages[stage]['source_rows']for stage in STAGES},
                  build_past_range='0:1086',algorithm='measured stages plus TP constants, k-best, resource qualification, top4 E2E one-percent rule')
    return dict(mx_decode_v2=config,evidence=evidence)


def select_measured(predicted,fresh_processes):
    if len(predicted)!=4 or len(fresh_processes)!=4:raise ValueError('four qualified combinations required')
    predicted=list(map(positive,predicted))
    if predicted!=sorted(predicted):raise ValueError('qualified combinations must follow prediction order')
    measured=[]
    for pair in fresh_processes:
        if len(pair)!=2:raise ValueError('two fresh processes per combination required')
        values=list(map(positive,pair));measured.append(sum(values)/2)
    best=min(range(len(measured)),key=lambda i:(measured[i],i));selected=best if measured[best]<=measured[0]*.99 else 0
    return dict(selected=selected,measured_ns=measured,pred_over_meas=[p/m for p,m in zip(predicted,measured)],
        threshold=.01,rule='fastest at least1percent faster than prediction-best, else prediction-best')
