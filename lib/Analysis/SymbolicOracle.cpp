// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/SymbolicOracle.h>
#include <tilemega/Analysis/ISLContext.h>
#include <isl/aff.h>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/point.h>
#include <isl/val.h>
#include <chrono>
#include <limits>
#include <stdexcept>

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
  std::vector<isl_pw_aff*> lower,upper;
  int input=0,output=0,parameters=0;
  mutable std::uint64_t queries=0;mutable double ms=0;
  ~Impl(){for(auto* p:lower)isl_pw_aff_free(p);for(auto* p:upper)isl_pw_aff_free(p);isl_pw_multi_aff_free(unique);isl_set_free(theta_image);isl_set_free(image);isl_map_free(map);}
};
SymbolicOracle::SymbolicOracle(std::string const& text):impl_(std::make_shared<Impl>()) {
  auto& d=*impl_;d.text=text;d.map=isl_map_read_from_str(SharedIslContext().raw(),text.c_str());
  if(!d.map)throw std::invalid_argument("invalid Oracle map");
  d.input=isl_map_dim(d.map,isl_dim_in);d.output=isl_map_dim(d.map,isl_dim_out);d.parameters=isl_map_dim(d.map,isl_dim_param);
  auto* parameterized=isl_map_copy(d.map);
  for(int i=0;i<d.input;++i)parameterized=isl_map_set_dim_name(parameterized,isl_dim_in,i,("__tilemega_oracle_"+std::to_string(i)).c_str());
  parameterized=isl_map_move_dims(parameterized,isl_dim_param,d.parameters,isl_dim_in,0,d.input);
  d.image=isl_map_range(parameterized);
  if(isl_map_is_single_valued(d.map)==isl_bool_true) {
    d.unique=isl_pw_multi_aff_from_map(isl_map_copy(d.map));
    if(!d.unique)throw std::runtime_error("single-valued relation failed affine conversion");
    d.kind=OracleKind::Unique;
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
}
OracleKind SymbolicOracle::kind() const{return impl_->kind;}
std::string const& SymbolicOracle::relation() const{return impl_->text;}
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
    for(int i=0;i<d.parameters;++i)d.theta_image=isl_set_fix_val(d.theta_image,isl_dim_param,i,isl_val_int_from_si(SharedIslContext().raw(),values[i]));
    d.theta_image=isl_set_coalesce(d.theta_image);
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
