// SPDX-License-Identifier: BSD-3-Clause
// Implementation-detail helpers shared by QuasiPolynomial.cpp,
// CouplingRelation.cpp, and CouplingDerivation.cpp. Not a public header: the
// public analysis API (ClosedForm.h, QuasiPolynomial.h, CouplingRelation.h)
// stays free of raw isl types by design, so Solver/Codegen clients never see
// them (mirrors ClosedForm.h's existing "no ISL types in the public
// interface" contract).
#pragma once

#include <isl/aff.h>
#include <isl/ctx.h>
#include <isl/map.h>
#include <isl/point.h>
#include <isl/polynomial.h>
#include <isl/printer.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/val.h>
#include <barvinok/isl.h>

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tilemega::analysis::isl_util {

/// Hand-built RAII for one isl reference-counted type. `Copy`/`Free` are the
/// type's own isl_*_copy/isl_*_free pair, so this is a thin ownership wrapper
/// around the C API's own refcounting -- not a reimplementation of it.
template <typename T, T* (*Copy)(T*), T* (*Free)(T*)>
class Obj {
 public:
  Obj() : ptr_(nullptr) {}
  /// Takes ownership of an __isl_give pointer.
  explicit Obj(T* give) : ptr_(give) {}
  Obj(Obj const& other) : ptr_(other.ptr_ ? Copy(other.ptr_) : nullptr) {}
  Obj(Obj&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }
  Obj& operator=(Obj other) {
    std::swap(ptr_, other.ptr_);
    return *this;
  }
  ~Obj() {
    // isl's _free functions return T* (always null, per __isl_null), not
    // void; the return value is intentionally discarded.
    if (ptr_) Free(ptr_);
  }
  T* get() const { return ptr_; }
  /// Releases ownership for an __isl_take call site.
  T* release() {
    T* p = ptr_;
    ptr_ = nullptr;
    return p;
  }
  explicit operator bool() const { return ptr_ != nullptr; }

 private:
  T* ptr_;
};

using Map = Obj<isl_map, isl_map_copy, isl_map_free>;
using Set = Obj<isl_set, isl_set_copy, isl_set_free>;
using PwQPolynomial =
    Obj<isl_pw_qpolynomial, isl_pw_qpolynomial_copy, isl_pw_qpolynomial_free>;
using PwAff = Obj<isl_pw_aff, isl_pw_aff_copy, isl_pw_aff_free>;
using Val = Obj<isl_val, isl_val_copy, isl_val_free>;
using Space = Obj<isl_space, isl_space_copy, isl_space_free>;
using Point = Obj<isl_point, isl_point_copy, isl_point_free>;

[[noreturn]] inline void Fail(isl_ctx* ctx, std::string const& what) {
  throw std::invalid_argument("isl: " + what);
}

inline Map ReadMap(isl_ctx* ctx, std::string const& text) {
  isl_map* map = isl_map_read_from_str(ctx, text.c_str());
  if (!map) Fail(ctx, "failed to parse map: " + text);
  return Map(map);
}

inline std::string ToString(isl_map* map /* borrowed */) {
  char* text = isl_map_to_str(map);
  std::string result = text ? text : "";
  free(text);
  return result;
}

inline Set ReadSet(isl_ctx* ctx, std::string const& text) {
  isl_set* set = isl_set_read_from_str(ctx, text.c_str());
  if (!set) Fail(ctx, "failed to parse set: " + text);
  return Set(set);
}

inline PwQPolynomial ReadPwQPolynomial(isl_ctx* ctx, std::string const& text) {
  // The stock parser combines pieces from left to right using general union
  // addition. Large finite task partitions then repeatedly intersect every
  // earlier piece. Parse each piece with ISL and prove disjoint supports
  // before taking its disjoint-add path; overlapping input keeps ISL's usual
  // additive semantics. No printed expression is evaluated by this splitter.
  auto open=text.find('{'),close=text.rfind('}');
  if(text.size()>4096 && open!=std::string::npos && close>open &&
      text.find('{',open+1)==std::string::npos &&
      text.find('}',open+1)==close && text.find('"')==std::string::npos &&
      text.find_first_not_of(" \t\r\n",close+1)==std::string::npos) {
    std::vector<PwQPolynomial> pieces;
    std::size_t first=open+1;
    while(first<close) {
      auto end=text.find(';',first);
      if(end==std::string::npos || end>close)end=close;
      auto fragment=text.substr(0,open+1)+text.substr(first,end-first)+" }";
      auto* piece=isl_pw_qpolynomial_read_from_str(ctx,fragment.c_str());
      if(!piece)Fail(ctx,"failed to parse quasi-polynomial piece: "+fragment);
      pieces.emplace_back(piece);first=end+1;
    }
    while(pieces.size()>1) {
      std::vector<PwQPolynomial> next;
      for(std::size_t i=0;i<pieces.size();i+=2) {
        if(i+1==pieces.size()) {next.push_back(std::move(pieces[i]));continue;}
        Set lhs(isl_pw_qpolynomial_domain(isl_pw_qpolynomial_copy(pieces[i].get())));
        Set rhs(isl_pw_qpolynomial_domain(isl_pw_qpolynomial_copy(pieces[i+1].get())));
        auto disjoint=isl_set_is_disjoint(lhs.get(),rhs.get());
        if(disjoint==isl_bool_error)Fail(ctx,"failed to prove polynomial support disjointness");
        auto* sum=disjoint==isl_bool_true?
            isl_pw_qpolynomial_add_disjoint(pieces[i].release(),pieces[i+1].release()):
            isl_pw_qpolynomial_add(pieces[i].release(),pieces[i+1].release());
        if(!sum)Fail(ctx,"failed to combine quasi-polynomial pieces");
        next.emplace_back(sum);
      }
      pieces=std::move(next);
    }
    if(!pieces.empty())return std::move(pieces.front());
  }
  isl_pw_qpolynomial* value = isl_pw_qpolynomial_read_from_str(ctx, text.c_str());
  if (!value) Fail(ctx, "failed to parse quasi-polynomial: " + text);
  return PwQPolynomial(value);
}

inline std::string ToString(isl_pw_qpolynomial* value /* borrowed */) {
  char* text = isl_pw_qpolynomial_to_str(value);
  std::string result = text ? text : "";
  free(text);
  return result;
}

}  // namespace tilemega::analysis::isl_util
