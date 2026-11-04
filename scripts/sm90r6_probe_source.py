"""Compose isolated native stages from compiler-generated, typed stage aliases.

Packing recipes and tensor maps follow each geometry's own materialized plan.
Only target operands are imported from variants; prerequisite stages use A0.
"""
import json
import re


def array(source, name):
    pattern = r'constexpr std::array<[^\n]+> '+name+r'=\{\{\n(.*?)\n\}\};'
    match = re.search(pattern, source, re.S)
    if not match:
        raise ValueError('generated native array missing: '+name)
    return match, match[1].splitlines()


def template(text):
    kind, arguments = text.split('<', 1)
    if not arguments.endswith('>'):
        raise ValueError('incomplete generated template')
    # The imported target is a leaf Gemm/Moe, with no nested template arguments.
    arguments = arguments[:-1].split(',')
    if any('<' in a or '>' in a for a in arguments):
        raise ValueError('unexpected nested leaf template')
    return kind, arguments


def parse(source, coordinates):
    buffers = array(source, 'kNativeBuffers')[1]
    maps = array(source, 'kNativeTensorMaps')[1]
    layers = [[int(x) for x in re.findall(r'(\d+)u', row)]
              for row in array(source, 'kNativeLayerSlots')[1]]
    aliases = dict(re.findall(r'^using (\w+)=(.*);$', source, re.M))
    _, program = template(aliases['NativeProgram'].split(',Stage_0,LayerGroup<')[0]+'>')
    per_layer = 6 if coordinates['model'] == 'moe' else 5
    per_layer -= bool(coordinates['expert_fuse'])
    merge = coordinates['attention'].get('merge', 'la') == 'stage' and coordinates['attention']['chunks'] > 1
    route = coordinates.get('route', 'local') == 'stage'
    combine = coordinates.get('combine', 'la') == 'stage'
    per_layer += merge + route + combine
    families = ['qkv','attention'] + (['attention_merge'] if merge else []) + ['o']
    if coordinates['model'] == 'moe':
        families += ['router'] + (['route'] if route else [])
        families += ['expert_down'] if coordinates['expert_fuse'] else ['expert_gate_up','expert_down']
        families += ['combine'] if combine else []
    else:
        families += ['gate_up','down']
    assert len(families) == per_layer
    head = max(int(k[5:]) for k in aliases if k.startswith('Body_'))
    return dict(buffers=buffers, maps=maps, layers=layers, aliases=aliases,
                program=program, families=families, per_layer=per_layer, head=head)


def geometry_source(source, coordinates, family, geometry, swiglu_u):
    """Retile one typed leaf, its packed weight, and its activation descriptor.

    Other stages remain prerequisite definitions. This avoids rebuilding the
    full L-layer CTG for each diagnostic leaf with an unchanged def-use graph.
    """
    parsed=parse(source,coordinates)
    name='HeadGemm_'+str(parsed['head']) if family=='lm_head' else 'Body_'+str(1+parsed['families'].index(family))
    kind,args=template(parsed['aliases'][name])
    if kind not in ('Gemm','Moe'):raise ValueError('probe retile requires an unfused leaf')
    old_tn=int(args[3]);tn=geometry['tile_n'];tk=geometry['slot_bytes']//(2*tn)
    if geometry['slot_bytes']!=2*tn*tk or geometry['multicast']:
        raise ValueError('probe leaf requires a complete ordinary weight slot')
    args[3:8]=map(str,(tn,tk,geometry['slots'],geometry['warps_n'],geometry['warps_k']))
    args[9]=str(swiglu_u)
    wi,mi,ri=(11,16,15) if kind=='Gemm' else (14,22,23)
    args[ri]=str(geometry['register_stages'])
    if kind=='Gemm' and len(args)>17:args[17]='false'
    value=args[wi]
    weight=parsed['layers'][0][int(value.split('|')[1].rstrip('u'))] if value.startswith('kLayerSlot|') else int(value.rstrip('u'))
    buffers=list(parsed['buffers']);row=buffers[weight]
    string=r'("(?:\\.|[^"\\])*")'
    match=re.fullmatch(r'\s*\{'+string+r',(\d+),(\d+),(\d+)ull,(\d+)ull,'+string+r'\},',row)
    if not match:raise ValueError('unrecognized packed-weight descriptor')
    weight_name=json.loads(match[1]);recipe=json.loads(json.loads(match[6]))
    if recipe['kind']!='tile_pages':raise ValueError('probe weight must be explicitly tiled')
    recipe['tile_n'],recipe['tile_k']=tn,tk
    leaf=recipe
    while isinstance(leaf,dict):
        if leaf.get('kind') in ('gate_up_interleave','expert_stack') and 'u' in leaf:leaf['u']=swiglu_u
        leaf=leaf.get('source')
    weight_name=re.sub(r'@t\d+x\d+$',f'@t{tn}x{tk}',weight_name)
    buffers[weight]='  {'+','.join([json.dumps(weight_name),match[2],match[3],match[4]+'ull',match[5]+'ull',
        json.dumps(json.dumps(recipe,separators=(',',':')))])+'},'
    maps=list(parsed['maps'])
    if args[mi]!='kAbsent':
        map_id=int(args[mi].rstrip('u'))
        values=re.findall(r'\d+(?:ull|u)',maps[map_id])
        if len(values)!=13:raise ValueError('unrecognized activation descriptor')
        # Native 3D A and 4D expert-h maps both tile K in box[2].
        values[11]=str(tk//64)+'u'
        if tk%64:raise ValueError('native probe activation chunks require TK divisible by 64')
        maps[map_id]='  {'+','.join(values[:2])+',{'+','.join(values[2:6])+'},{'+','.join(values[6:9])+'},{'+','.join(values[9:13])+'}},'
    if kind=='Moe':args[13]='64' if (tn,tk)==(64,64) else '0'
    text=source.replace('using '+name+'='+parsed['aliases'][name]+';',
        'using '+name+'='+kind+'<'+','.join(args)+'>;')
    for array_name,rows,typ in [('kNativeBuffers',buffers,'tm_buffer_info'),('kNativeTensorMaps',maps,'NativeTensorMap')]:
        location,_=array(text,array_name)
        text=text[:location.start()]+f'constexpr std::array<{typ},{len(rows)}> {array_name}={{{{\n'+'\n'.join(rows)+'\n}};'+text[location.end():]
    location,rows=array(text,'kNativeStages')
    i=1+len(parsed['layers'])*parsed['per_layer'] if family=='lm_head' else 1+parsed['families'].index(family)
    # StageInfo's tasks field is after weight_bytes; its experts factor stays.
    row=rows[i];match=re.search(r'(\d+)ull,(\d+),',row)
    tasks=int(match[2])*old_tn
    if tasks%tn:raise ValueError('new target tile does not divide its task domain')
    rows[i]=row[:match.start(2)]+str(tasks//tn)+row[match.end(2):]
    text=text[:location.start()]+f'constexpr std::array<StageInfo,{len(rows)}> kNativeStages={{{{\n'+'\n'.join(rows)+'\n}};'+text[location.end():]
    return text


def compose(base_source, base_coordinates, variants, family):
    """Variants are (geometry, source, coordinates) in native enumeration order."""
    base = parse(base_source, base_coordinates)
    buffers = list(base['buffers']); maps = list(base['maps'])
    buffer_ids = {row:i for i,row in enumerate(buffers)}
    map_ids = {row:i for i,row in enumerate(maps)}
    generated = []; instances = []; ledger = []
    stage_rows = array(base_source, 'kNativeStages')[1]
    layers = len(base['layers'])
    prefix = [('Body_0',0,stage_rows[0])]
    if family == 'lm_head':
        for layer in range(layers):
            for j in range(base['per_layer']):
                prefix.append((f'Body_{1+j}',layer,stage_rows[1+layer*base['per_layer']+j]))
    else:
        index = base['families'].index(family)
        prefix += [(f'Body_{1+j}',0,stage_rows[1+j]) for j in range(index)]

    def append(body, layer, measured, info):
        index = len(instances)
        instances.append(dict(body=body,layer=layer,measured=measured,info=info))
        return index

    for variant_index,(geometry,source,coordinates) in enumerate(variants):
        variant = parse(source, coordinates)
        if variant['program'][:3] != base['program'][:3] or len(variant['layers']) != layers:
            raise ValueError('probe variants disagree on batch, capacity, residency or layers')
        # Restart prerequisites before each theta so in-place residual stages do
        # not inherit another geometry's repeated outputs. Prerequisites are untimed.
        for body,layer,info in prefix:append(body,layer,False,info)
        name = ('HeadGemm_'+str(variant['head'])) if family == 'lm_head' else 'Body_'+str(1+variant['families'].index(family))
        kind,args = template(variant['aliases'][name])
        if kind == 'Gemm':
            operands, descriptors = range(11,15), [16]
        elif kind == 'Moe':
            operands, descriptors = range(14,22), [22]
        else:
            raise ValueError('isolated target must be an unfused Gemm or Moe: '+kind)

        def buffer(old):
            row = variant['buffers'][old]
            if row not in buffer_ids:
                buffer_ids[row] = len(buffers);buffers.append(row)
            return buffer_ids[row]

        for i in operands:
            value = args[i]
            if value == 'kAbsent':continue
            old = variant['layers'][0][int(value.split('|')[1].rstrip('u'))] if value.startswith('kLayerSlot|') else int(value.rstrip('u'))
            args[i] = str(buffer(old))
        for i in descriptors:
            value = args[i]
            if value == 'kAbsent':continue
            row = variant['maps'][int(value.rstrip('u'))]
            match = re.match(r'(\s*\{)(\d+)u,', row)
            if not match:raise ValueError('unrecognized native tensor map')
            row = row[:match.start(2)]+str(buffer(int(match[2])))+row[match.end(2):]
            if row not in map_ids:
                map_ids[row] = len(maps);maps.append(row)
            args[i] = str(map_ids[row])
        target = f'ProbeBody{variant_index}'
        generated.append(f'using {target}={kind}<'+','.join(args)+'>;')
        if family == 'lm_head':
            # Publish per-CTA argmax partials without timing the argmax reducer.
            _,head_args = template(variant['aliases']['Body_'+str(variant['head'])])
            for i in range(1,4):head_args[i] = str(buffer(int(head_args[i].rstrip('u'))))
            generated.append(f'using ProbeHead{variant_index}=Head<{target},'+','.join(head_args[1:6])+',false,1,0,true>;')
            target = f'ProbeHead{variant_index}'
        target_info = array(source,'kNativeStages')[1][1+layers*variant['per_layer'] if family=='lm_head' else 1+variant['families'].index(family)]
        indices = [append(target,0,True,target_info) for _ in range(layers)]
        ledger.append(dict(variant=variant_index, geometry=geometry, stages=indices))

    source = base_source
    for name,rows,typ in [('kNativeBuffers',buffers,'tm_buffer_info'),('kNativeTensorMaps',maps,'NativeTensorMap')]:
        match,_ = array(source,name)
        replacement = f'constexpr std::array<{typ},{len(rows)}> {name}={{{{\n'+'\n'.join(rows)+'\n}};'
        source = source[:match.start()]+replacement+source[match.end():]
    original = base['aliases']['NativeProgram']
    # Preserve all A0 stage types and layer tables, enlarging only the binding's
    # buffer/map tables for the target's independently packed weight layouts.
    fields = original.split(',',5)
    fields[3],fields[4] = str(len(buffers)),str(len(maps))
    base_program = ','.join(fields)
    declarations = '\n'.join(generated)+'\nusing ProbeBase='+base_program+';\n'
    declarations += 'struct NativeProgram:ProbeBase {\n'
    declarations += f'  static constexpr bool kEmbed=false;\n  static constexpr unsigned kStages={len(instances)},kCounters=kStages;\n'
    declarations += '  static constexpr int kSharedBytes=[] {int n=ProbeBase::kSharedBytes;\n'
    for body in dict.fromkeys(x['body'] for x in instances):
        declarations += f'    if(n<{body}::kSharedBytes)n={body}::kSharedBytes;\n'
    declarations += '    return n;}();\n  static_assert(sizeof(Binding)+256<32764);\n'
    declarations += '  __device__ static __forceinline__ void Step(Binding const& b,int past,char* shared,float epsilon,GridCounter* events,unsigned long long iteration,TraceState trace,unsigned) {\n'
    # Reuse only untimed prerequisite counters across layers. The epoch grows
    # with that reuse; the measured target instances keep distinct counters.
    for i,row in enumerate(instances):
        within=i%(len(prefix)+layers)
        if family=='lm_head' and 1<=within<len(prefix):
            if within!=1:continue
            declarations += f'    #pragma unroll 1\n    for(int layer=0;layer<{layers};++layer) {{\n'
            for j in range(base['per_layer']):
                declarations += f'      ProbeInstance<NativeProgram,Body_{1+j},{i+j},false>(b,past,shared,epsilon,events,iteration*{layers}+layer,trace,layer);\n'
            declarations += '    }\n'
        else:
            declarations += f'    ProbeInstance<NativeProgram,{row["body"]},{i},{str(row["measured"]).lower()}>(b,past,shared,epsilon,events,iteration,trace,{row["layer"]});\n'
    declarations += '  }\n};'
    source = source.replace('using NativeProgram='+original+';',declarations)
    match,_ = array(source,'kNativeStages')
    rows = [x['info'] for x in instances]
    source = source[:match.start()]+f'constexpr std::array<StageInfo,{len(rows)}> kNativeStages={{{{\n'+'\n'.join(rows)+'\n}};'+source[match.end():]
    source = source.replace('#include <tilemega/Codegen/tasks/Sm90DenseProgram.cuh>',
        '#define TILEMEGA_TRACE_STAGE 1\n#include <tilemega/Codegen/tasks/Sm90DenseProbe.cuh>')
    return source,dict(family=family,layers=layers,variants=ledger,stages=len(instances),
        timed_stages=layers*len(variants),prefix_stages=len(prefix),
        prefix_execution='runtime layer loop for untimed head prerequisites' if family=='lm_head' else 'unrolled layer-zero prerequisites',
        scope='layer-zero isolated body repeated L times per theta; A0 prerequisites reset before each theta; V3 boundaries; no preissue')
