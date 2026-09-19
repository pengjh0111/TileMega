// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Analysis/ISLContext.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/point.h>
#include <isl/val.h>
#include <isl/ilp.h>
#include <limits>
#include <vector>
#include <utility>
#include <array>
#include <functional>
#include <stdexcept>
namespace tilemega::analysis {
// Exact box equality is required before bypassing ISL point allocation.
// Coupled coordinates and modular holes stay on the general enumerator.
inline bool FiniteBoxExtent(isl_set* set,int arity,long* low,long* high) {
  if(isl_set_is_box(set)!=isl_bool_true)return false;
  auto box=isl_set_universe(isl_set_get_space(set));
  for(int i=0;i<arity;++i) {
    auto lo=isl_set_dim_min_val(isl_set_copy(set),i);
    auto hi=isl_set_dim_max_val(isl_set_copy(set),i);
    bool fits=isl_val_is_int(lo)==isl_bool_true && isl_val_is_int(hi)==isl_bool_true &&
        isl_val_cmp_si(lo,std::numeric_limits<long>::min())>=0 &&
        isl_val_cmp_si(hi,std::numeric_limits<long>::max())<=0;
    if(!fits) {isl_val_free(lo);isl_val_free(hi);isl_set_free(box);return false;}
    low[i]=isl_val_get_num_si(lo);high[i]=isl_val_get_num_si(hi);
    box=isl_set_lower_bound_val(box,isl_dim_set,i,lo);
    box=isl_set_upper_bound_val(box,isl_dim_set,i,hi);
  }
  bool exact=isl_set_is_equal(set,box)==isl_bool_true;isl_set_free(box);
  return exact;
}

// Dense two-coordinate relations often become boxes after fixing one task
// coordinate. `min_slice_width` is what a slice must be worth before its ISL
// query is taken: a client that expands the slice back into points only profits
// on wide ones, while a client that consumes the box itself profits at any
// width. Sparse diagonals and modular holes keep the point enumerator.
inline bool VisitFiniteSliceRegions(isl_set* set,int arity,long min_slice_width,
    std::function<void(long const*,long const*)> const& region,
    std::function<isl_stat(isl_set*)> const& general) {
  if(arity!=4)return false;
  std::array<long,4> low{},high{};std::vector<int> varying;
  for(int i=0;i<arity;++i) {
    auto lo=isl_set_dim_min_val(isl_set_copy(set),i),hi=isl_set_dim_max_val(isl_set_copy(set),i);
    bool fits=isl_val_is_int(lo)==isl_bool_true && isl_val_is_int(hi)==isl_bool_true &&
        isl_val_cmp_si(lo,std::numeric_limits<long>::min())>=0 &&
        isl_val_cmp_si(hi,std::numeric_limits<long>::max())<=0;
    if(!fits){isl_val_free(lo);isl_val_free(hi);return false;}
    low[i]=isl_val_get_num_si(lo);high[i]=isl_val_get_num_si(hi);isl_val_free(lo);isl_val_free(hi);
    if(low[i]!=high[i])varying.push_back(i);
  }
  if(varying.size()!=2)return false;
  int axis=varying[0],other=varying[1];
  if(static_cast<long double>(high[axis])-low[axis]>static_cast<long double>(high[other])-low[other])std::swap(axis,other);
  auto slice_at=[&](long coordinate){return isl_set_fix_val(isl_set_copy(set),isl_dim_set,axis,isl_val_int_from_si(isl_set_get_ctx(set),coordinate));};
  auto probe=slice_at(low[axis]);
  bool wide=false;
  if(isl_set_is_box(probe)==isl_bool_true) {
    auto lo=isl_set_dim_min_val(isl_set_copy(probe),other),hi=isl_set_dim_max_val(isl_set_copy(probe),other);
    wide=isl_val_is_int(lo)==isl_bool_true && isl_val_is_int(hi)==isl_bool_true &&
        static_cast<long double>(isl_val_get_num_si(hi))-isl_val_get_num_si(lo)>=
            static_cast<long double>(min_slice_width);
    isl_val_free(lo);isl_val_free(hi);
  }
  isl_set_free(probe);if(!wide)return false;
  for(long coordinate=low[axis];;) {
    auto slice=slice_at(coordinate);
    try {
      std::array<long,6> slice_low{},slice_high{};
      if(isl_set_is_empty(slice)!=isl_bool_true) {
        if(FiniteBoxExtent(slice,arity,slice_low.data(),slice_high.data())) region(slice_low.data(),slice_high.data());
        else if(general(slice)!=isl_stat_ok) throw std::runtime_error("relation slice enumeration failed");
      }
    } catch(...) {isl_set_free(slice);throw;}
    isl_set_free(slice);if(coordinate==high[axis])break;++coordinate;
  }
  return true;
}

inline bool VisitFiniteSlices(isl_set* set,int arity,
    std::function<void(long const*)> const& visit,
    std::function<isl_stat(isl_set*)> const& general) {
  return VisitFiniteSliceRegions(set,arity,255,[&](long const* low,long const* high) {
    std::array<long,6> point{};std::copy_n(low,arity,point.begin());
    while(true) {
      visit(point.data());
      int i=arity-1;
      while(i>=0 && point[i]==high[i]) {point[i]=low[i];--i;}
      if(i<0)break;
      ++point[i];
    }
  },general);
}
// Every point of a dense piece, for clients that need the edges themselves.
inline void ExpandFiniteBox(long const* low,long const* high,int arity,
    std::function<void(long const*)> const& visit) {
  std::array<long,6> point{};std::copy_n(low,arity,point.begin());
  while(true) {
    visit(point.data());
    int i=arity-1;
    while(i>=0 && point[i]==high[i]) {point[i]=low[i];--i;}
    if(i<0)break;
    ++point[i];
  }
}

// Basic pieces may overlap: adjacency clients deduplicate their emitted edges.
// A dense piece is reported to `region` as one inclusive box per coordinate --
// the complete bipartite block it stands for -- so a client that only needs
// per-node aggregates pays its side lengths instead of their product. Pieces
// with coupled coordinates or modular holes still arrive point by point, so
// the union of what the two callbacks see is the same relation either way.
// The default `min_slice_width` is the one the point expansion below needs, so
// `VisitFiniteRelation` decomposes exactly as it did before regions existed.
inline void VisitFiniteRegions(IslContext& ctx,std::string const& text,int arity,
                   std::function<void(long const*,long const*)> const& region,
                   std::function<void(long const*)> const& visit,
                   long min_slice_width=255) {
  if (arity<1 || arity>6) throw std::invalid_argument("unsupported relation arity");
  struct Sink { int arity; long min_slice_width;
                std::function<void(long const*)> const* visit;
                std::function<void(long const*,long const*)> const* region; std::string error; }
      sink{arity,min_slice_width,&visit,&region,{}};
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
    try {
      std::array<long,6> low{},high{};
      if(FiniteBoxExtent(set,p.sink->arity,low.data(),high.data())) {
        (*p.sink->region)(low.data(),high.data());isl_set_free(set);return isl_stat_ok;
      }
      auto general=[&](isl_set* slice){return isl_set_foreach_point(slice,*p.callback,p.sink);};
      if(VisitFiniteSliceRegions(set,p.sink->arity,p.sink->min_slice_width,
                                 *p.sink->region,general)) {isl_set_free(set);return isl_stat_ok;}
    } catch(std::exception const& e) {p.sink->error=e.what();isl_set_free(set);return isl_stat_error;}
    auto status=isl_set_foreach_point(set,*p.callback,p.sink);isl_set_free(set);return status;
  };
  auto map=isl_map_read_from_str(ctx.raw(),text.c_str());
  if(!map || isl_map_dim(map,isl_dim_param)!=0){isl_map_free(map);throw std::runtime_error("relation must be finite and bound");}
  auto set=isl_map_wrap(map);
  if(isl_set_dim(set,isl_dim_set)!=arity){isl_set_free(set);throw std::invalid_argument("relation arity mismatch");}
  if(isl_set_is_bounded(set)!=isl_bool_true){isl_set_free(set);throw std::runtime_error("unbounded task relation");}
  auto status=isl_set_foreach_basic_set(set,component,&parts);isl_set_free(set);
  if(status!=isl_stat_ok)throw std::runtime_error("relation stream: "+sink.error);
}

inline void VisitFiniteRelation(IslContext& ctx,std::string const& text,int arity,
                   std::function<void(long const*)> const& visit) {
  VisitFiniteRegions(ctx,text,arity,
      [&](long const* low,long const* high){ExpandFiniteBox(low,high,arity,visit);},visit);
}
} // namespace tilemega::analysis
