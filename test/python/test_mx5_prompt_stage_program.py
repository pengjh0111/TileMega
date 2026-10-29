"""Execute host-generated prompt stages, including the final-token head edge."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
DRIVER = r'''
#include "StageProgram.h"
#include <fstream>
#include <stdexcept>
using namespace tilemega::codegen;
int main(int argc,char** argv) {
  StageProgramSpec s;s.kind=StageProgramKind::kPrompt;s.loop.end="24";
  s.operand_parameters="int b";s.run_parameters="int b,int past,char* smem";
  s.buffer_row_type="BufferRow";s.entry_prefetch_operands="0";
  for(int i=0;i<6;++i) {
    StageProgramStage row{"S"+std::to_string(i),"Body<"+std::to_string(i)+">","Shape",
      std::to_string(i),std::to_string(i),i==5?"S0":"S"+std::to_string(i+1),"",i};
    if(i==5){row.terminal_body="Body<5>";row.terminal_next="Head";row.terminal_prefetch_operands="6";}
    s.stages.push_back(row);
  }
  s.tail_stages={{"Head","Body<6>","Shape","6","6","S0","0",6}};
  s.external_buffers={{"x","0",0,0,"","0",0}};
  bool rejected=false;
  try {EmitStageProgram(s);}catch(std::invalid_argument const& e) {
    rejected=std::string(e.what()).find("final-token tail predicate")!=std::string::npos;
  }
  if(!rejected)return 1;
  s.terminal_condition="past == 63";std::ofstream out(argv[1]);out<<EmitStageProgram(s);
  return out.good()?0:2;
}
'''
PREFIX = r'''
#include <cassert>
#include <cstdint>
#include <iostream>
#define __device__
#define __forceinline__ inline
int calls[7]{},prefetch[7]{};
struct Packet{int owner;};struct Shape{};
struct BufferRow{char const* name;int dtype;std::uint64_t n,b;char const* recipe;int slot,layer;};
template<int I> struct Body {
  using Prefetched=Packet;static constexpr int kId=I;
  static Packet Prefetch(int op,Shape){assert(op==I);++prefetch[I];return {I};}
};
namespace tilemega {namespace backend {namespace mx {}}
namespace codegen {namespace mx {
template<class Body,class Next>
auto RunStageV2(bool run,Packet incoming,int op,Shape,char*,Next next) {
  assert(run&&incoming.owner==Body::kId&&op==Body::kId);++calls[Body::kId];return next();
}
}}}
'''
SUFFIX = r'''
int main() {
  using P=tm_mx_plan::Plan;assert(P::kStageCount==145);
  int counters[145]{},generation=0;
  auto incoming=P::EntryPrefetch<P>(0);
  for(int past=0;past<64;++past) {
    int stage=0;
    auto barrier=[&]{assert(stage<145);++counters[stage++];++generation;};
    P::RunStep<P>(0,past,nullptr,incoming,barrier,[](int){return true;});
    assert(stage==(past==63?145:144));assert(incoming.owner==0);
    assert(calls[6]==(past==63?1:0));
  }
  for(int i=0;i<6;++i)assert(calls[i]==24*64);
  assert(calls[6]==1&&prefetch[6]==1);
  for(int i=0;i<144;++i)assert(counters[i]==64);
  assert(counters[144]==1&&generation==9217);
  std::cout<<"PASS 64 prompt tokens;144 layer counters x64;head counter x1;9217 releases\n";
}
'''


class PromptStageProgramTest(unittest.TestCase):
    def test_prompt_emission_executes_head_only_once(self):
        compiler = shutil.which('c++')
        if compiler is None:
            self.skipTest('host C++17 compiler required')
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / 'emit.cpp').write_text(DRIVER)
            subprocess.run([compiler, '-std=c++17', '-O2', '-UNDEBUG',
                            '-I' + str(ROOT / 'lib/Codegen'), str(d / 'emit.cpp'),
                            str(ROOT / 'lib/Codegen/StageProgram.cpp'), '-o', str(d / 'emit')],
                           check=True, capture_output=True, text=True)
            subprocess.run([str(d / 'emit'), str(d / 'generated.cpp')], check=True,
                           capture_output=True, text=True)
            generated = (d / 'generated.cpp').read_text()
            self.assertIn('l + 1 == 24 && (past == 63)', generated)
            self.assertNotIn('stage_descriptor', generated)
            (d / 'generated.cpp').write_text(PREFIX + generated + SUFFIX)
            subprocess.run([compiler, '-std=c++17', '-O2', '-UNDEBUG',
                            str(d / 'generated.cpp'), '-o', str(d / 'execute')],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([str(d / 'execute')], check=True,
                                    capture_output=True, text=True)
            self.assertIn('head counter x1;9217 releases', result.stdout)


if __name__ == '__main__':
    unittest.main()
