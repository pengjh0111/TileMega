#!/usr/bin/env python3
"""Reconstruct four raw traces with exact range queries and aggregate by space.

Task fixed and mainloop are combined: these binaries enable trace v2 only.
Per-space excess compares that space's wall span with its unique external-byte
streaming time; those overlapping space spans must not be summed as a path.
chain_categories.tsv instead sums the nonoverlapping realized-chain partition.
"""
import argparse,collections,csv,importlib.util,json,pathlib,time
E=pathlib.Path(__file__).resolve().parent;ROOT=E.parents[2]
spec=importlib.util.spec_from_file_location('trace_v2',E.parent/'TRACE_V2/analyze.py')
trace=importlib.util.module_from_spec(spec);spec.loader.exec_module(trace)
def rows(path):
 with path.open() as stream:return list(csv.DictReader(stream,delimiter='\t'))
def analyze(cell):
 dump=E/'trace'/cell/'dump';out=E/'trace_analysis'/cell;out.mkdir(parents=True,exist_ok=True)
 result,spaces,links=trace.analyze_task_space_trace(dump)
 mapping={int(r['stage']):r for r in rows(E/'flow_final'/cell/'home.flow_spaces.tsv')}
 logical=collections.defaultdict(lambda:[0.,0.])
 for row in rows(E/'stage_floor'/cell/'stages.tsv'):
  logical[row['op']][0]+=float(row['no_producer_read_bytes']);logical[row['op']][1]+=float(row['external_write_bytes'])
 last_gemm=max(s for s,r in mapping.items() if r['category']=='gemm')
 category={'gemm':'proj','rmsnorm':'norm','residual':'add','kv_append':'append','swiglu':'act','attention':'attn','embedding':'embed'}
 rate=json.loads((E/'fit/target.json').read_text())['calibration_by_dtype']['bf16']['pipelines']['dram_gbps']
 for row in spaces:
  m=mapping[row['stage']];name=m['name'];combine=name.endswith('.combine');base=name.removesuffix('.combine')
  external_read,external_write=logical[base]
  # The split producer reads external operands; its combiner writes the final
  # output. No-producer bytes cannot be charged a second time to the combiner.
  has_combine=any(x['name']==name+'.combine' for x in mapping.values())
  byte_count=(0 if combine else external_read)+(external_write if combine or not has_combine else 0)
  row.update(name=name,category='lm_head' if row['stage']==last_gemm else category.get(m['category'],m['category']),external_bytes=byte_count,
             streaming_ns=byte_count/rate,wall_span_ns=row['last_end_ns']-row['first_ready_ns'])
  row['space_excess_ns']=row['wall_span_ns']-row['streaming_ns']
 names={r['stage']:r for r in spaces};summary=collections.defaultdict(lambda:collections.defaultdict(float))
 for link in links:
  link['category']=names[link['stage']]['category'];link['name']=names[link['stage']]['name']
  aggregate=summary[link['category']];aggregate['links']+=1
  for key in ('wall_ns','task_fixed_plus_mainloop_ns','publish_ns','wait_hop_ns','barrier_ns','idle_ns'):aggregate[key]+=link[key]
 trace.write_rows(out/'task_spaces.tsv',spaces);trace.write_rows(out/'chain_links.tsv',links)
 trace.write_rows(out/'chain_categories.tsv',[dict(category=k,**v) for k,v in sorted(summary.items())])
 trace.write_rows(out/'reconstruction.tsv',[result])
 (out/'limitations.txt').write_text('Trace v2 combines TaskBody fixed and mainloop. Space wall spans overlap and are not additive. Chain categories partition the realized path. Streaming bytes are exact unique logical-stage read images; split combiners receive only external writes.\n')
 print(cell,result['cp_reconstructed_with_publish_ns'],len(links),flush=True)
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--wait',action='store_true');args=p.parse_args()
 pending={f'{m}_s{s}' for m in ('llama','qwen3') for s in (1,64)}
 while pending:
  for cell in sorted(pending):
   log=E/'trace'/cell/'activated.log'
   if not log.exists() or 'E2E_TIME ' not in log.read_text():continue
   analyze(cell);pending.remove(cell)
  if not args.wait:break
  if pending:time.sleep(30)
 if pending:raise SystemExit('missing completed traces: '+','.join(sorted(pending)))
if __name__=='__main__':main()
