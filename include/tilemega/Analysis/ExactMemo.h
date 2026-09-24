// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <any>
#include <cstdint>
#include <map>
#include <string>
#include <string_view>
#include <type_traits>
#include <initializer_list>

namespace tilemega::analysis {
// Search-scoped memoization of pure ISL calculations. Keys carry complete
// expressions and bindings; no successful module-verification result is cached.
struct ExactAnalysisMemo {
  std::map<std::string,std::any> values;
  std::size_t bytes=0;
  std::uint64_t hits=0,misses=0;
};
inline thread_local ExactAnalysisMemo* active_exact_memo=nullptr;
struct ScopedExactAnalysisMemo {
  ExactAnalysisMemo memo;
  ExactAnalysisMemo* previous=active_exact_memo;
  ScopedExactAnalysisMemo(){active_exact_memo=&memo;}
  ~ScopedExactAnalysisMemo(){active_exact_memo=previous;}
  ScopedExactAnalysisMemo(ScopedExactAnalysisMemo const&)=delete;
  ScopedExactAnalysisMemo& operator=(ScopedExactAnalysisMemo const&)=delete;
};
template<class Compute>
auto MemoExact(std::initializer_list<std::string_view> fields,Compute compute) -> decltype(compute()) {
  auto* cache=active_exact_memo;
  if(!cache)return compute();
  using Result=decltype(compute());
  std::string key;
  for(auto field:fields){key+=std::to_string(field.size());key+=':';key+=field;}
  if(auto found=cache->values.find(key);found!=cache->values.end()) {
    ++cache->hits;return std::any_cast<Result>(found->second);
  }
  ++cache->misses;auto value=compute();
  std::size_t bytes=key.size()+sizeof(Result);
  if constexpr(!std::is_same_v<Result,bool>)bytes+=value.ToString().size();
  constexpr std::size_t budget=32u*1024u*1024u;
  if(bytes<=budget) {
    if(cache->bytes+bytes>budget || cache->values.size()>=8192){cache->values.clear();cache->bytes=0;}
    cache->bytes+=bytes;cache->values.emplace(std::move(key),value);
  }
  return value;
}
}
