// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/tasks/Sm90DenseBoundaryDiagnostic.h>
#include <cassert>
#include <iostream>
#include <utility>

using namespace tilemega::codegen::native_dense::boundary_diagnostic;

int main() {
  assert(Parse("off").observe==0 && Parse("off").probe_mask==0);
  assert(Parse("lean").observe==1 && Parse("lean").probe_mask==0);
  assert(Parse("full").observe==1 && Parse("full").probe_mask==kAll);
  for(unsigned i=0;i<kClasses;++i) {
    auto only=Parse("only-class",kNames[i]),leave=Parse("leave-class-out",kNames[i]);
    assert(only.observe && leave.observe);
    assert(only.probe_mask==(1u<<i));
    assert((only.probe_mask|leave.probe_mask)==kAll && !(only.probe_mask&leave.probe_mask));
  }
  for(unsigned layers:{1u,16u,28u,32u}) {
    unsigned count=5*layers+3,seen[kClasses]{};
    assert(ValidStages(count));
    for(unsigned i=0;i<count;++i)++seen[static_cast<unsigned>(ClassOf(i,count))];
    for(unsigned i=0;i<kClasses;++i)
      assert(seen[i]==(i>=1 && i<=5?layers:1));
    assert(ClassOf(count-2,count)==Class::Head && ClassOf(count-1,count)==Class::Argmax);
  }
  for(unsigned count:{0u,3u,7u,9u,142u})assert(!ValidStages(count));
  unsigned rejected=0;
  for(auto pair:{std::pair{"unknown",""},std::pair{"lean","qkv"},
       std::pair{"full","qkv"},std::pair{"off","qkv"},
       std::pair{"only-class",""},std::pair{"leave-class-out","combine"}})
    try {Parse(pair.first,pair.second);}catch(std::invalid_argument const&) {++rejected;}
  assert(rejected==6);
  assert(Matches(Class::Qkv,"kGemm","layer.1.qkv.weight"));
  assert(Matches(Class::O,"kGemm","layer.1.o_proj.weight"));
  assert(Matches(Class::Head,"kGemm","lm_head.weight"));
  assert(!Matches(Class::O,"kGemm","layer.1.down_proj.weight"));
  assert(!Matches(Class::Attention,"kGemm","layer.1.qkv.weight"));
  std::cout<<"native boundary modes/class mapping passed\n";
}
