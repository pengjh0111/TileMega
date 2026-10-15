// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Codegen/DependencyTablePacking.h>
#include <cassert>
#include <cstdio>
#include <fstream>
#include <string>
int main(int argc,char** argv) {
  using tilemega::analysis::TaskInterval;
  tilemega::codegen::DependencyIntervalPool pool;
  unsigned comparisons=0;
  std::vector<std::vector<TaskInterval>> inputs;
  // Repeated layers share literal intervals, including padding and disjoint
  // windows. Each edge retains its own stride and consumer cardinality.
  for(unsigned layer=0;layer<48;++layer)for(unsigned shape=1;shape<=3;++shape) {
    std::vector<TaskInterval> rows;
    for(unsigned consumer=0;consumer<64*shape;++consumer)
      for(unsigned slot=0;slot<shape;++slot)
        rows.push_back({consumer*5+slot*2,slot==shape-1?0u:slot+1});
    inputs.push_back(std::move(rows));
  }
  std::vector<unsigned> offsets;
  for(auto const& edge:inputs)offsets.push_back(pool.Intern(edge));
  for(unsigned edge=0;edge<inputs.size();++edge)
    for(unsigned row=0;row<inputs[edge].size();++row) {
      auto expected=inputs[edge][row];auto actual=pool.Intervals().at(offsets[edge]+row);
      assert(actual.first==expected.first && actual.count==expected.count);++comparisons;
    }
  assert(pool.Intervals().size()==64*(1+4+9));
  auto before=pool.Intervals().size();
  auto changed=inputs[0];changed.back().count=77;
  assert(pool.Intern(changed)==before);
  assert(pool.Intervals().back().count==77);
  assert(pool.Intern(inputs[0])==offsets[0]);
  std::printf("PASS immutable interval pool %u exact row comparisons\n",comparisons);
  if(argc==3) {
    std::ifstream source(argv[1],std::ios::binary),ranges(argv[2]);
    assert(source && ranges);
    source.seekg(0,std::ios::end);auto bytes=source.tellg();source.seekg(0);
    assert(bytes>=0 && std::size_t(bytes)%sizeof(TaskInterval)==0);
    std::vector<TaskInterval> original(std::size_t(bytes)/sizeof(TaskInterval));
    source.read(reinterpret_cast<char*>(original.data()),bytes);assert(source);
    tilemega::codegen::DependencyIntervalPool real;
    std::vector<std::pair<unsigned,unsigned>> edges;
    unsigned first,count;
    while(ranges>>first>>count) {
      assert(first<=original.size() && count<=original.size()-first);
      std::vector<TaskInterval> edge(original.begin()+first,original.begin()+first+count);
      auto offset=real.Intern(edge);edges.emplace_back(offset,first);
      for(unsigned row=0;row<count;++row) {
        auto a=original[first+row];auto b=real.Intervals().at(offset+row);
        assert(a.first==b.first && a.count==b.count);
      }
    }
    assert(!edges.empty());
    std::printf("PASS real generated interval pool original=%zu compact=%zu edges=%zu\n",
        original.size(),real.Intervals().size(),edges.size());
  }
}
