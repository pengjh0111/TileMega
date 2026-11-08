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
 changes=[('repetition count','Repeat<6,Loop_0','Repeat<5,Loop_0'),
          ('logical prefix','>,Stage_61,Stage_62','>,Stage_60,Stage_62'),
          ('logical suffix','>,Stage_61,Stage_62','>,Stage_61,Stage_61'),
          ('edge statistics','GemmStage<Body_1,Bind_11,Shape_1>','Wrap_11'),
          ('last edge','GemmStage<Body_60,Bind_20,Shape_60>','Wrap_20'),
          ('loop binding','RepeatStage<Wrap_12,Wrap_12>','RepeatStage<Wrap_12,Wrap_13>'),
          ('table size','std::array<unsigned,2970> indices','std::array<unsigned,2969> indices'),
          ('remapped buffer','    0u,1u,2u,3u,','    1u,1u,2u,3u,'),
          ('prefetch stage','{Bind_1::kWeight,unsigned(Wrap_1::kWeightBytes)}','{Bind_6::kWeight,unsigned(Wrap_1::kWeightBytes)}'),
          ('prefetch count','std::array<RepeatWeight,63>','std::array<RepeatWeight,62>'),
          ('missing loop','using Loop_8=','using Lost_8=')]
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
