// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/SymbolicOracle.h>
#include <tilemega/Analysis/ISLContext.h>
#include <isl/aff.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/point.h>
#include <isl/val.h>
#include <isl/ilp.h>
#include <chrono>
#include <limits>
#include <stdexcept>
#include "OracleExpression.h"

namespace tilemega::analysis {
char const* ToString(OracleKind k) {switch(k){case OracleKind::Unique:return "Unique";case OracleKind::Rectangular:return "Rectangular";default:return "General";}}
char const* ToString(EdgeStructure k) {switch(k){case EdgeStructure::OneToOne:return "1:1";case EdgeStructure::ManyToOne:return "N:1";case EdgeStructure::OneToMany:return "1:N";default:return "M:N";}}
long OracleImage::Count() const {
  if(empty)return 0;if(!rectangular)return long(points.size());long n=1;
  for(auto [lo,hi]:box) {long width=hi-lo+1;if(width<=0)return 0;if(n>std::numeric_limits<long>::max()/width)throw std::overflow_error("oracle count overflow");n*=width;}
  return n;
}
void OracleImage::ForEach(std::function<void(std::vector<long> const&)> const& f) const {
  if(empty)return;if(!rectangular){for(auto const& p:points)f(p);return;}
  std::vector<long> coordinate(box.size());
  std::function<void(std::size_t)> visit=[&](std::size_t dim) {
    if(dim==box.size()){f(coordinate);return;}
    for(long v=box[dim].first;v<=box[dim].second;++v){coordinate[dim]=v;visit(dim+1);}
  };visit(0);
}
namespace {
long integer(isl_val* value) {
  if(!value || isl_val_is_int(value)!=isl_bool_true) {isl_val_free(value);throw std::runtime_error("noninteger Oracle coordinate");}
  long result=isl_val_get_num_si(value);isl_val_free(value);return result;
}
std::string reverseText(std::string const& text) {
  auto* map=isl_map_reverse(isl_map_read_from_str(SharedIslContext().raw(),text.c_str()));
  char* raw=isl_map_to_str(map);if(!raw){isl_map_free(map);throw std::invalid_argument("invalid Oracle relation");}
  std::string result(raw);free(raw);isl_map_free(map);return result;
}
EdgeStructure classify(std::string const& text) {
  auto* map=isl_map_read_from_str(SharedIslContext().raw(),text.c_str());
  auto bijective=isl_map_is_bijective(map),injective=isl_map_is_injective(map),single=isl_map_is_single_valued(map);
  isl_map_free(map);
  if(bijective<0 || injective<0 || single<0)throw std::invalid_argument("ISL structure classification failed");
  if(bijective)return EdgeStructure::OneToOne;
  if(single && !injective)return EdgeStructure::ManyToOne;
  if(injective && !single)return EdgeStructure::OneToMany;
  return EdgeStructure::ManyToMany;
}
}
struct SymbolicOracle::Impl {
  std::string text;OracleKind kind=OracleKind::General;
  isl_map* map=nullptr;isl_set* image=nullptr;isl_pw_multi_aff* unique=nullptr;
  mutable isl_set* theta_image=nullptr;
  mutable std::vector<long> bound_theta;
  mutable std::vector<isl_pw_aff*> theta_lower,theta_upper;
  OracleProgram unique_program;
  mutable OracleProgram box_program;
  mutable OracleProgram general_bounds;
  mutable OracleExpression general_membership;
  std::vector<isl_pw_aff*> lower,upper;
  int input=0,output=0,parameters=0;
  mutable std::uint64_t queries=0;mutable double ms=0;
  ~Impl(){for(auto* p:lower)isl_pw_aff_free(p);for(auto* p:upper)isl_pw_aff_free(p);for(auto* p:theta_lower)isl_pw_aff_free(p);for(auto* p:theta_upper)isl_pw_aff_free(p);isl_pw_multi_aff_free(unique);isl_set_free(theta_image);isl_set_free(image);isl_map_free(map);}
};
SymbolicOracle::SymbolicOracle(std::string const& text):impl_(std::make_shared<Impl>()) {
  auto& d=*impl_;d.text=text;d.map=isl_map_read_from_str(SharedIslContext().raw(),text.c_str());
  if(!d.map)throw std::invalid_argument("invalid Oracle map");
  d.input=isl_map_dim(d.map,isl_dim_in);d.output=isl_map_dim(d.map,isl_dim_out);d.parameters=isl_map_dim(d.map,isl_dim_param);
  auto* parameterized=isl_map_copy(d.map);
  for(int i=0;i<d.input;++i)parameterized=isl_map_set_dim_name(parameterized,isl_dim_in,i,("__tilemega_oracle_"+std::to_string(i)).c_str());
  parameterized=isl_map_move_dims(parameterized,isl_dim_param,d.parameters,isl_dim_in,0,d.input);
  auto* parameterized_unique=isl_map_copy(parameterized);
  d.image=isl_map_range(parameterized);
  if(isl_map_is_single_valued(d.map)==isl_bool_true) {
    d.unique=isl_pw_multi_aff_from_map(isl_map_copy(d.map));
    if(!d.unique)throw std::runtime_error("single-valued relation failed affine conversion");
    d.kind=OracleKind::Unique;
    auto* affine=isl_pw_multi_aff_from_map(parameterized_unique);parameterized_unique=nullptr;
    std::vector<isl_pw_aff*> expressions;
    for(int i=0;i<d.output;++i)expressions.push_back(isl_pw_multi_aff_get_pw_aff(affine,i));
    d.unique_program=OracleProgram::Build(isl_set_params(isl_set_copy(d.image)),expressions);
    for(auto* p:expressions)isl_pw_aff_free(p);isl_pw_multi_aff_free(affine);
  } else if(isl_set_is_box(d.image)==isl_bool_true) {
    // The predicate sees symbolic source coordinates as parameters, proving
    // every fiber, including strided/nonrectangular counterexamples.
    d.kind=OracleKind::Rectangular;
    for(int i=0;i<d.output;++i) {
      d.lower.push_back(isl_set_dim_min(isl_set_copy(d.image),i));
      d.upper.push_back(isl_set_dim_max(isl_set_copy(d.image),i));
      if(!d.lower.back() || !d.upper.back())throw std::runtime_error("box bound construction failed");
    }
  }
  isl_map_free(parameterized_unique);
}
OracleKind SymbolicOracle::kind() const{return impl_->kind;}
std::string const& SymbolicOracle::relation() const{return impl_->text;}
std::string SymbolicOracle::UniqueMapText() const {
  if(kind()!=OracleKind::Unique)throw std::invalid_argument("Oracle is not a Unique affine map");
  char* raw=isl_pw_multi_aff_to_str(impl_->unique);if(!raw)throw std::runtime_error("cannot print Unique mapping");
  std::string result(raw);free(raw);return result;
}
long OracleImage::MaximumLinear() const {
  if(empty)return -1;
  if(rectangular){if(box.size()!=1)throw std::invalid_argument("release maximum requires a linear fiber");return box.front().second;}
  if(points.empty() || points.front().size()!=1)throw std::invalid_argument("missing linear fiber points");
  long result=points.front().front();
  // Query already enumerated this exact General fiber. Its one-dimensional
  // maximum is its lexmax; deriving a symbolic lexmax over all theta and all
  // source coordinates can explode in piece count and adds no information.
  for(auto const& point:points){if(point.size()!=1)throw std::invalid_argument("release maximum requires a linear fiber");result=std::max(result,point.front());}
  return result;
}
long SymbolicOracle::MaximumLinear(std::vector<long> const& source,ParamBinding const& theta) const {
  return LinearRelease(source,theta).maximum;
}
OracleLinearRelease SymbolicOracle::LinearRelease(std::vector<long> const& source,ParamBinding const& theta) const {
  auto& d=*impl_;
  if(d.output!=1 || int(source.size())!=d.input)throw std::invalid_argument("release maximum requires linear fiber and matching source");
  if(d.kind!=OracleKind::General) {
    auto image=Query(source,theta);auto maximum=image.MaximumLinear();
    return {maximum,image.Count()==maximum+1};
  }
  auto start=std::chrono::steady_clock::now();++d.queries;
  struct Timer{Impl& d;std::chrono::steady_clock::time_point start;~Timer(){d.ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}} timer{d,start};
  // Bind one exact Presburger fiber before optimizing. No global symbolic AST
  // or membership scan over a potentially million-element bounding interval.
  auto* fiber=isl_set_copy(d.image);
  for(int i=0;i<d.parameters;++i) {
    char const* name=isl_map_get_dim_name(d.map,isl_dim_param,i);
    auto value=theta.values.find(name?name:"");
    if(value==theta.values.end()){isl_set_free(fiber);throw std::invalid_argument("Oracle theta is unbound");}
    fiber=isl_set_fix_val(fiber,isl_dim_param,i,isl_val_int_from_si(SharedIslContext().raw(),value->second));
  }
  for(int i=0;i<d.input;++i)fiber=isl_set_fix_val(fiber,isl_dim_param,d.parameters+i,isl_val_int_from_si(SharedIslContext().raw(),source[i]));
  // All parameters are now singleton-bound; projecting them out exposes the
  // exact constant fiber to ISL's box predicate (and avoids parametric hulls).
  fiber=isl_set_project_out(fiber,isl_dim_param,0,d.parameters+d.input);
  fiber=isl_set_coalesce(fiber);
  auto empty=isl_set_is_empty(fiber);
  if(empty==isl_bool_error){isl_set_free(fiber);throw std::runtime_error("release fiber emptiness failed");}
  if(empty){isl_set_free(fiber);return {};}
  long maximum=integer(isl_set_dim_max_val(isl_set_copy(fiber),0));
  auto box=isl_set_is_box(fiber);
  if(box==isl_bool_error){isl_set_free(fiber);throw std::runtime_error("release prefix proof failed");}
  bool prefix=box==isl_bool_true && integer(isl_set_dim_min_val(isl_set_copy(fiber),0))==0;
  isl_set_free(fiber);return {maximum,prefix};
}
std::uint64_t SymbolicOracle::queries() const{return impl_->queries;}
double SymbolicOracle::query_ms() const{return impl_->ms;}
OracleImage SymbolicOracle::Query(std::vector<long> const& source,ParamBinding const& theta) const {
  auto& d=*impl_;auto start=std::chrono::steady_clock::now();++d.queries;
  struct Timer{Impl& d;std::chrono::steady_clock::time_point start;~Timer(){d.ms+=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();}} timer{d,start};
  if(int(source.size())!=d.input)throw std::invalid_argument("Oracle source dimensionality");
  std::vector<long> values;
  for(int i=0;i<d.parameters;++i) {
    char const* name=isl_map_get_dim_name(d.map,isl_dim_param,i);
    auto it=theta.values.find(name ? name : "");
    if(it==theta.values.end())throw std::invalid_argument("Oracle theta is unbound");
    values.push_back(it->second);
  }
  if(d.kind==OracleKind::Unique) {
    auto arguments=values;arguments.insert(arguments.end(),source.begin(),source.end());
    bool nonempty=false;std::vector<long> coordinates;
    if(d.unique_program.Eval(arguments,nonempty,coordinates)) {
      OracleImage result;result.empty=!nonempty;
      if(nonempty)result.points.push_back(std::move(coordinates));return result;
    }
    auto* point=isl_point_zero(isl_pw_multi_aff_get_domain_space(d.unique));
    for(int i=0;i<d.parameters;++i)point=isl_point_set_coordinate_val(point,isl_dim_param,i,isl_val_int_from_si(SharedIslContext().raw(),values[i]));
    for(int i=0;i<d.input;++i)point=isl_point_set_coordinate_val(point,isl_dim_set,i,isl_val_int_from_si(SharedIslContext().raw(),source[i]));
    OracleImage result;std::vector<long> coordinate;
    if(d.output==0) {
      auto* domain=isl_pw_multi_aff_domain(isl_pw_multi_aff_copy(d.unique));
      auto* intersection=isl_set_intersect(domain,isl_set_from_point(isl_point_copy(point)));
      bool empty=isl_set_is_empty(intersection)==isl_bool_true;isl_set_free(intersection);
      if(empty){isl_point_free(point);return result;}
    }
    for(int i=0;i<d.output;++i) {
      auto* value=isl_pw_aff_eval(isl_pw_multi_aff_get_pw_aff(d.unique,i),isl_point_copy(point));
      if(value && isl_val_is_nan(value)==isl_bool_true){isl_val_free(value);isl_point_free(point);return result;}
      coordinate.push_back(integer(value));
    }
    isl_point_free(point);result.empty=false;result.points.push_back(std::move(coordinate));return result;
  }
  if(!d.theta_image || d.bound_theta!=values) {
    isl_set_free(d.theta_image);d.theta_image=isl_set_copy(d.image);d.bound_theta=values;
    for(auto* p:d.theta_lower)isl_pw_aff_free(p);d.theta_lower.clear();
    for(auto* p:d.theta_upper)isl_pw_aff_free(p);d.theta_upper.clear();
    for(int i=0;i<d.parameters;++i)d.theta_image=isl_set_fix_val(d.theta_image,isl_dim_param,i,isl_val_int_from_si(SharedIslContext().raw(),values[i]));
    d.theta_image=isl_set_coalesce(d.theta_image);
    // A General symbolic relation can become rectangular for a bound theta.
    // Prove every remaining source coordinate before caching these bounds.
    if(d.kind==OracleKind::General && isl_set_is_box(d.theta_image)==isl_bool_true) {
      for(int i=0;i<d.output;++i) {
        d.theta_lower.push_back(isl_set_dim_min(isl_set_copy(d.theta_image),i));
        d.theta_upper.push_back(isl_set_dim_max(isl_set_copy(d.theta_image),i));
      }
    }
    auto const& lower=d.kind==OracleKind::Rectangular?d.lower:d.theta_lower;
    auto const& upper=d.kind==OracleKind::Rectangular?d.upper:d.theta_upper;
    d.box_program={};
    d.general_bounds={};d.general_membership={};
    if(!lower.empty()) {
      std::vector<isl_pw_aff*> expressions;
      for(int i=0;i<d.output;++i){expressions.push_back(lower[i]);expressions.push_back(upper[i]);}
      d.box_program=OracleProgram::Build(isl_set_params(isl_set_copy(d.theta_image)),expressions);
    } else if(d.kind==OracleKind::General) {
      // Bounds delimit a local scan, never a predecessor set. Every emitted
      // point must satisfy the original Presburger membership predicate.
      std::vector<isl_pw_aff*> bounds;
      for(int i=0;i<d.output;++i){bounds.push_back(isl_set_dim_min(isl_set_copy(d.theta_image),i));bounds.push_back(isl_set_dim_max(isl_set_copy(d.theta_image),i));}
      d.general_bounds=OracleProgram::Build(isl_set_params(isl_set_copy(d.theta_image)),bounds);
      for(auto* p:bounds)isl_pw_aff_free(p);
      auto* membership=isl_set_copy(d.theta_image);
      for(int i=0;i<d.output;++i)membership=isl_set_set_dim_name(membership,isl_dim_set,i,("__tilemega_image_"+std::to_string(i)).c_str());
      membership=isl_set_move_dims(membership,isl_dim_param,d.parameters+d.input,isl_dim_set,0,d.output);
      auto program=OracleProgram::Build(membership,{});d.general_membership=std::move(program.domain);
    }
  }
  auto const& lower=d.kind==OracleKind::Rectangular?d.lower:d.theta_lower;
  auto const& upper=d.kind==OracleKind::Rectangular?d.upper:d.theta_upper;
  if(!lower.empty()) {
    auto arguments=values;arguments.insert(arguments.end(),source.begin(),source.end());
    bool nonempty=false;std::vector<long> coordinates;
    if(d.box_program.Eval(arguments,nonempty,coordinates)) {
      OracleImage result;result.empty=!nonempty;result.rectangular=true;
      if(nonempty)for(int i=0;i<d.output;++i)result.box.emplace_back(coordinates[2*i],coordinates[2*i+1]);
      return result;
    }
    auto* point=isl_point_zero(isl_pw_aff_get_domain_space(lower.front()));
    for(int i=0;i<d.parameters;++i)point=isl_point_set_coordinate_val(point,isl_dim_param,i,isl_val_int_from_si(SharedIslContext().raw(),values[i]));
    for(int i=0;i<d.input;++i)point=isl_point_set_coordinate_val(point,isl_dim_param,d.parameters+i,isl_val_int_from_si(SharedIslContext().raw(),source[i]));
    OracleImage result;result.rectangular=true;
    for(int i=0;i<d.output;++i) {
      auto* lo=isl_pw_aff_eval(isl_pw_aff_copy(lower[i]),isl_point_copy(point));
      auto* hi=isl_pw_aff_eval(isl_pw_aff_copy(upper[i]),isl_point_copy(point));
      if(lo && hi && (isl_val_is_nan(lo)==isl_bool_true || isl_val_is_nan(hi)==isl_bool_true)) {
        isl_val_free(lo);isl_val_free(hi);isl_point_free(point);return result;
      }
      result.box.emplace_back(integer(lo),integer(hi));
    }
    isl_point_free(point);result.empty=false;return result;
  }
  if(d.general_bounds.domain.valid && d.general_membership.valid) {
    auto arguments=values;arguments.insert(arguments.end(),source.begin(),source.end());
    bool nonempty=false;std::vector<long> bounds;
    if(d.general_bounds.Eval(arguments,nonempty,bounds)) {
      OracleImage result;if(!nonempty)return result;
      // Large sparse hulls stay on ISL's exact point enumerator.
      __int128 volume=1;
      for(int i=0;i<d.output && volume<=16384;++i)volume*=__int128(bounds[2*i+1])-bounds[2*i]+1;
      if(volume>=0 && volume<=16384) {
        arguments.resize(d.parameters+d.input+d.output);std::vector<long> coordinate(d.output);bool valid=true;
        std::function<void(int)> visit=[&](int dim) {
          if(!valid)return;
          if(dim==d.output) {
            long contains=0;valid=d.general_membership.Eval(arguments,contains);
            if(valid && contains)result.points.push_back(coordinate);return;
          }
          for(long x=bounds[2*dim];; ++x) {
            if(x>bounds[2*dim+1])break;
            arguments[d.parameters+d.input+dim]=coordinate[dim]=x;visit(dim+1);
            if(x==bounds[2*dim+1])break;
          }
        };visit(0);
        if(valid){result.empty=result.points.empty();return result;}
      }
    }
  }
  auto* fiber=isl_set_copy(d.theta_image);
  for(int i=0;i<d.input;++i)fiber=isl_set_fix_val(fiber,isl_dim_param,d.parameters+i,isl_val_int_from_si(SharedIslContext().raw(),source[i]));
  OracleImage result;
  if(isl_set_is_empty(fiber)==isl_bool_true){isl_set_free(fiber);return result;}
  result.empty=false;
  if(d.kind==OracleKind::Rectangular) {
    auto* point=isl_set_sample_point(isl_set_params(isl_set_copy(fiber)));result.rectangular=true;
    for(int i=0;i<d.output;++i)result.box.emplace_back(
      integer(isl_pw_aff_eval(isl_pw_aff_copy(d.lower[i]),isl_point_copy(point))),
      integer(isl_pw_aff_eval(isl_pw_aff_copy(d.upper[i]),isl_point_copy(point))));
    isl_point_free(point);
  } else {
    // General remains General: this proves only the current concrete fiber,
    // not every symbolic source. Enumerating its proven box avoids an ISL LP
    // scan per point; strided/disconnected fibers still use exact point scans.
    fiber=isl_set_coalesce(fiber);
    if(isl_set_is_box(fiber)==isl_bool_true) {
      auto* point=isl_set_sample_point(isl_set_params(isl_set_copy(fiber)));result.rectangular=true;
      for(int i=0;i<d.output;++i)result.box.emplace_back(
        integer(isl_pw_aff_eval(isl_set_dim_min(isl_set_copy(fiber),i),isl_point_copy(point))),
        integer(isl_pw_aff_eval(isl_set_dim_max(isl_set_copy(fiber),i),isl_point_copy(point))));
      isl_point_free(point);isl_set_free(fiber);return result;
    }
    struct State {OracleImage* result;int dimensions;} state{&result,d.output};
    auto status=isl_set_foreach_point(fiber,[](isl_point* point,void* opaque)->isl_stat {
      auto& state=*static_cast<State*>(opaque);std::vector<long> coordinate;
      for(int i=0;i<state.dimensions;++i)coordinate.push_back(integer(isl_point_get_coordinate_val(point,isl_dim_set,i)));
      isl_point_free(point);state.result->points.push_back(std::move(coordinate));return isl_stat_ok;
    },&state);
    if(status<0){isl_set_free(fiber);throw std::runtime_error("General fiber enumeration failed");}
  }
  isl_set_free(fiber);return result;
}
bool SymbolicOracle::IsAllBox(std::vector<std::pair<long,long>> const& box,ParamBinding const& theta) const {
  auto& d=*impl_;if(int(box.size())!=d.output)return false;
  auto* map=isl_map_copy(d.map);
  for(int i=0;i<d.parameters;++i){auto* name=isl_map_get_dim_name(map,isl_dim_param,i);
    auto found=theta.values.find(name?name:"");
    if(found!=theta.values.end())map=isl_map_fix_val(map,isl_dim_param,i,isl_val_int_from_si(SharedIslContext().raw(),found->second));
  }
  auto* expected=isl_set_universe(isl_space_range(isl_map_get_space(map)));
  for(int i=0;i<d.output;++i){expected=isl_set_lower_bound_si(expected,isl_dim_set,i,box[i].first);expected=isl_set_upper_bound_si(expected,isl_dim_set,i,box[i].second);}
  auto* product=isl_map_from_domain_and_range(isl_map_domain(isl_map_copy(map)),expected);
  auto eq=isl_map_is_equal(product,map);isl_map_free(product);isl_map_free(map);return eq==isl_bool_true;
}
OraclePair::OraclePair(std::string const& relation):structure(classify(relation)),forward(relation),reverse(reverseText(relation)) {}
}
