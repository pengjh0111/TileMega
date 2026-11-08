#!/usr/bin/env python3
"""Reject broken logical coverage, edge variants and Repeat binding tables."""
import argparse,json,re,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'python'))
from tilemega.build.feature_contract import build_defines
from tilemega.dnn.f3_contract import dispatch

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--artifact',type=Path,required=True);p.add_argument('--flat',type=Path,required=True);a=p.parse_args()
 for artifact in [a.artifact,a.flat]:
  source=Path(str(artifact)+'.cu').read_text();manifest=json.loads(Path(str(artifact)+'.plan.json').read_text());defines=build_defines(source,[])
  assert len(dispatch(manifest,defines,source))==63
  if artifact==a.artifact:s,m,d=source,manifest,defines
 repetitions=int(re.search(r',Stage_0,Repeat<(6|12),',s)[1]);body=60//repetitions;canonical=1+body
 changes=[('repetition count',f'Repeat<{repetitions},Loop_0',f'Repeat<{repetitions-1},Loop_0'),
          ('logical prefix','>,Stage_61,Stage_62','>,Stage_60,Stage_62'),
          ('logical suffix','>,Stage_61,Stage_62','>,Stage_61,Stage_61'),
          ('edge statistics',f'GemmStage<Body_1,Bind_{canonical},Shape_1>',f'Wrap_{canonical}'),
          ('last edge',f'GemmStage<Body_60,Bind_{2*body},Shape_60>',f'Wrap_{2*body}'),
          ('loop binding',f'RepeatStage<Wrap_{canonical+1},Wrap_{canonical+1}>',f'RepeatStage<Wrap_{canonical+1},Wrap_{canonical+2}>'),
          ('table size','std::array<unsigned,2970> indices','std::array<unsigned,2969> indices'),
          ('remapped buffer','    0u,1u,2u,3u,','    1u,1u,2u,3u,'),
          ('prefetch stage','{Bind_1::kWeight,unsigned(Wrap_1::kWeightBytes)}','{Bind_6::kWeight,unsigned(Wrap_1::kWeightBytes)}'),
          ('prefetch count','std::array<RepeatWeight,63>','std::array<RepeatWeight,62>'),
          ('missing loop',f'using Loop_{body-2}=',f'using Lost_{body-2}=')]
 # B16 and future plans can have different buffer/map counts.
 size=re.search(r'std::array<unsigned,(\d+)> indices',s)[1]
 changes[6]=('table size',f'std::array<unsigned,{size}> indices',f'std::array<unsigned,{int(size)-1}> indices')
 for name,before,after in changes:
  assert before in s,name
  damaged=s.replace(before,after,1)
  try:dispatch(m,d,damaged)
  except ValueError:continue
  raise AssertionError('accepted corrupted '+name)
 print(json.dumps(dict(status='passed',evidence='verified',logical_stages=63,flat_stages=63,corruptions_rejected=len(changes))))
if __name__=='__main__':main()
