// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Analysis/ISLContext.h>
#include <cassert>
#include <iostream>
#include <limits>

namespace tilemega::tests::dm_physical_footprint_test {
int TestDmPhysicalFootprint(int,char**) {
  using namespace solver;
  analysis::IslContext isl;
  ModelDescription model;model.dm=true;model.dtype=ScalarType::kBF16;
  model.dims={13,3,16};model.dims.batch=2;model.memory_arena_bytes=1024;
  model.physical_buffers={
      {"a",{256,0,0,0,0,2,0}},
      {"b",{128,0,0,0,0,2,0}},
      {"c",{128,0,0,0,0,4,256}},
      {"retained",{3,0,0,0,0,4,{}}},
      {"external",{5,0,0,0,7,2,{}}},
      {"ids",{0,1,0,0,0,8,{}}}};
  assert(model.PhysicalFootprintBytes()==1178);
  assert(model.LiveFootprintBytes()==1178);
  std::set<std::string> names{"a","b","c"};
  assert(model.PhysicalFootprintBytes(&names)==768);
  names={"a","b"};assert(model.PhysicalFootprintBytes(&names)==512);
  names={};assert(model.PhysicalFootprintBytes(&names)==0);
  names={"a","retained","ids"};assert(model.PhysicalFootprintBytes(&names)==628);
  auto rejects=[](auto run) {bool rejected=false;try {run();}
    catch(std::exception const&){rejected=true;}assert(rejected);};
  names={"unknown"};rejects([&]{model.PhysicalFootprintBytes(&names);});
  auto key=model.PhysicalFootprintKey();
  auto changed=model;changed.physical_buffers.at("b").arena_offset=256;
  assert(changed.PhysicalFootprintKey()!=key);
  names={"b","c"};assert(changed.PhysicalFootprintBytes(&names)==512);
  changed=model;changed.physical_buffers.at("b").arena_offset=1024;
  rejects([&]{changed.PhysicalFootprintBytes();});
  changed=model;changed.physical_buffers.at("b").arena_offset=1;
  rejects([&]{changed.PhysicalFootprintBytes();});
  changed=model;changed.physical_buffers.at("ids").element_bytes=3;
  rejects([&]{changed.PhysicalFootprintBytes();});
  changed=model;changed.physical_buffers.at("ids").per_seq=UINT64_MAX;
  rejects([&]{changed.PhysicalFootprintBytes();});
  changed=model;changed.physical_buffers.at("ids").constant=UINT64_MAX;
  rejects([&]{changed.PhysicalFootprintBytes();});
  changed=model;changed.dims.batch_parameter="batch";
  rejects([&]{changed.PhysicalFootprintBytes();});
  analysis::ParamBinding bindings;bindings.Bind("batch",3);
  auto bound=changed.SubstituteParams(bindings);
  assert(bound.PhysicalFootprintBytes()==1192);
  changed=model;changed.dims.past=-1;rejects([&]{changed.PhysicalFootprintBytes();});
  changed=model;changed.dims.batch=0;rejects([&]{changed.PhysicalFootprintBytes();});
  ModelDescription legacy;legacy.dims={13,3,16};legacy.gemms={{5,7}};
  assert(legacy.LiveFootprintBytes()==4*(5*7+13*(5+7)));
  std::cout<<"DM physical footprint: arena unions, typed retained buffers, bound dimensions and rejection checks PASS\n";
  return 0;
}
}
