"""Validate shared terminal coverage and reject corrupted terminal-sharing artifacts."""
from pathlib import Path
import copy,json,re,sys
import argparse
R=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(R/'python'))
from tilemega.build.feature_contract import build_defines
from tilemega.dnn.f3_contract import dispatch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--control',type=Path,required=True)
p.add_argument('--candidate',type=Path,required=True)
args=p.parse_args();original=args.control;candidate=args.candidate
s=Path(str(candidate)+'.cu').read_text()
m=json.loads(Path(str(candidate)+'.plan.json').read_text());d=build_defines(s,[])
assert len(dispatch(m,d,s))==63
assert len(dispatch(m,build_defines(Path(str(original)+'.cu').read_text(),[]),Path(str(original)+'.cu').read_text()))==63
body=re.search(r'using Body_10=([^;]+);',s)[0]
shape=re.search(r'struct Shape_60 \{[^}]+\};',s)[0]
bind=re.search(r'struct Bind_60 \{[^}]+\};',s)[0]
changes=[('profile disabled','TILEMEGA_F3_OPTIONAL_ROW_STATS 1','TILEMEGA_F3_OPTIONAL_ROW_STATS 0'),
 ('profile invalid','TILEMEGA_F3_OPTIONAL_ROW_STATS 1','TILEMEGA_F3_OPTIONAL_ROW_STATS 2'),
 ('missing optional wrapper',body,body.replace('Sm90F3OptionalRowStats<','Sm90F3WriteRowStats<')),
 ('wrong canonical wrapper','using Body_10=','using Body_70='),
 ('different shape',shape,shape.replace('kRows=2048','kRows=2049')),
 ('different terminal arithmetic','using Body_60=Sm90F3Gemm<128,128,5','using Body_60=Sm90F3Gemm<128,128,4'),
 ('terminal statistics present',bind,bind.replace('kRowPartials=kAbsent','kRowPartials=26u')),
 ('wrong shared loop','using Loop_4=RepeatStage<LoopWrap_4>;','using Loop_4=RepeatStage<LoopWrap_3>;'),
 ('missing loop','using Loop_3=','using Lost_3='),
 ('bad repetition count','Repeat<12,Loop_0','Repeat<6,Loop_0'),
 ('weight prefetch changed','{Bind_60::kWeight,unsigned(Wrap_60::kWeightBytes)}','{Bind_59::kWeight,unsigned(Wrap_60::kWeightBytes)}')]
normal_bind=re.search(r'struct Bind_55 \{[^}]+\};',s)[0]
changes.append(('nonterminal statistics absent',normal_bind,re.sub(r'kRowPartials=\d+u','kRowPartials=kAbsent',normal_bind)))
indices=re.search(r'array<unsigned,6060> indices=\{\{(.*?)\}\};',s,re.S)[1]
tokens=[v.strip() for v in indices.split(',') if v.strip()];assert tokens[11*505+26]=='kAbsent'
tokens[11*505+26]='26u';changes.append(('terminal remap changed',indices,','.join(tokens)+','))
for label,before,after in changes:
 assert before in s and before!=after,label
 damaged=s.replace(before,after,1)
 try:dispatch(m,build_defines(damaged,[]),damaged)
 except ValueError:continue
 raise AssertionError('accepted corruption: '+label)
print(dict(passed=True,corruptions=len(changes)))
