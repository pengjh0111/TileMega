// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/ISLContext.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/point.h>
#include <isl/val.h>
#include <array>
#include <functional>
#include <stdexcept>
namespace tilemega::analysis {
// Basic pieces may overlap: adjacency clients deduplicate their emitted edges.
inline void VisitFiniteRelation(IslContext& ctx,std::string const& text,int arity,
                   std::function<void(long const*)> const& visit) {
  if (arity<1 || arity>6) throw std::invalid_argument("unsupported relation arity");
  struct Sink { int arity; std::function<void(long const*)> const* visit; std::string error; } sink{arity,&visit,{}};
  auto point=[](isl_point* p,void* data)->isl_stat {
    auto& s=*static_cast<Sink*>(data);std::array<long,6> coordinates{};
    for(int i=0;i<s.arity;++i){auto v=isl_point_get_coordinate_val(p,isl_dim_set,i);coordinates[i]=isl_val_get_num_si(v);isl_val_free(v);}
    isl_point_free(p);
    try { (*s.visit)(coordinates.data()); } catch(std::exception const& e){s.error=e.what();return isl_stat_error;}
    return isl_stat_ok;
  };
  struct Parts { Sink* sink; decltype(point)* callback; } parts{&sink,&point};
  auto component=[](isl_basic_set* b,void* data)->isl_stat {
    auto& p=*static_cast<Parts*>(data);auto set=isl_set_from_basic_set(b);
    auto status=isl_set_foreach_point(set,*p.callback,p.sink);isl_set_free(set);return status;
  };
  auto map=isl_map_read_from_str(ctx.raw(),text.c_str());
  if(!map || isl_map_dim(map,isl_dim_param)!=0){isl_map_free(map);throw std::runtime_error("relation must be finite and bound");}
  auto set=isl_map_wrap(map);
  if(isl_set_is_bounded(set)!=isl_bool_true){isl_set_free(set);throw std::runtime_error("unbounded task relation");}
  auto status=isl_set_foreach_basic_set(set,component,&parts);isl_set_free(set);
  if(status!=isl_stat_ok)throw std::runtime_error("relation stream: "+sink.error);
}
} // namespace tilemega::analysis
