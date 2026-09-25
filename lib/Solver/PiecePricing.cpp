// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/PiecePricing.h>
#include <tilemega/Analysis/CouplingCache.h>
#include <functional>
#include <iomanip>
#include <sstream>
namespace tilemega::solver {
PiecePrices PriceBoundaryPieces(CostModel const& cost,DerivedTaskInput const& input,
    ModelTaskSemantics const& semantic,BackendTraits const& traits,Residency residency,
    ModelDescription const& model,int chunks,PiecePriceCache* cache) {
  if(!cost.options().regime_a || model.dtype!=ScalarType::kBF16)throw std::invalid_argument("boundary parts require regime A");
  auto theta=model.MetricBindings();auto const& cal=cost.target().CalibrationFor("bf16");
  std::ostringstream key;key<<analysis::SemanticSignature(semantic.op)<<std::hexfloat;
  key<<':'<<traits.tile_m<<':'<<traits.tile_n<<':'<<traits.tile_k<<':'<<traits.stages<<':'<<chunks<<':'<<residency.ctas_per_sm<<':'<<traits.threads<<':'<<traits.smem_bytes;
  // Scalar combiners have no collective BackendTraits geometry. Their
  // local reduction extent and ownership still change with their own split.
  key<<':'<<input.work.task_reduce_extent.ToString()<<':'<<input.work.task_count.ToString();
  for(auto const& tile:input.task.tile)key<<':'<<tile.ToIslText();
  for(auto const& [name,value]:std::map<std::string,long>(theta.values.begin(),theta.values.end()))key<<':'<<name<<'='<<value;
  key<<":regime_a:"<<cost.options().physical_traffic<<cost.options().stage_latency<<cost.options().physical_fixed<<cost.options().sdcm_above_knee;
  double stream=input.no_producer_read_bytes?input.stream_bytes:model.LiveFootprintBytes();
  if(stream<=cal.l2_knee_bytes)key<<":resident:"<<model.LiveFootprintBytes();else key<<":stream:"<<stream<<':'<<input.produced_live_bytes;
  // Resource calibration is immutable during a search. Include its values
  // so an explicitly reused cache cannot alias a different target profile.
  key<<':'<<cal.dram_gbps<<':'<<cal.l2_gbps<<':'<<cal.task_body.latency_scale<<':'<<cal.task_body.stage_rate_bytes_per_ns;
  if(cache){auto found=cache->entries.find(key.str());if(found!=cache->entries.end()){++cache->hits;return found->second;}++cache->misses;}
  struct Axis {std::string name;std::vector<std::pair<long,long>> parts;};std::vector<Axis> axes;
  long tasks=input.work.task_count.Eval(theta);
  if(input.scalar_access)axes.push_back({"q",{{0,tasks}}});
  else for(std::size_t i=0;i<input.task.output.axes.size();++i)if(input.task.IsTiled(i)) {
    auto const& a=input.task.output.axes[i];long extent=a.extent.Eval(theta,theta),tile=input.task.tile[i].Eval(theta,theta);
    Axis axis;axis.name=a.name;if(extent/tile)axis.parts.push_back({0,extent/tile});if(extent%tile)axis.parts.push_back({extent/tile,extent/tile+1});axes.push_back(std::move(axis));
  }
  PiecePrices result;std::vector<std::pair<long,long>> bounds(axes.size());
  auto make_domain=[&]{std::string tuple,where;for(std::size_t i=0;i<axes.size();++i){if(i){tuple+=",";where+=" and ";}tuple+=axes[i].name;where+=std::to_string(bounds[i].first)+"<="+axes[i].name+"<"+std::to_string(bounds[i].second);}return analysis::CouplingRelation::FromIslText("{ ["+tuple+"] -> ["+tuple+"]"+(where.empty()?"":" : "+where)+" }");};
  std::vector<analysis::QuasiPolynomial const*> quantities{&input.work.read_elements,&input.work.write_elements};
  if(traits.stages>0){quantities.push_back(&input.work.nominal_read_elements);quantities.push_back(&input.work.nominal_write_elements);quantities.push_back(&input.work.nominal_task_reduce_extent);}
  if(input.physical_read_bytes)quantities.push_back(&*input.physical_read_bytes);
  if(input.no_producer_read_bytes)quantities.push_back(&*input.no_producer_read_bytes);
  if(input.external_write_bytes)quantities.push_back(&*input.external_write_bytes);
  // PriceParts depends on coordinates only through these access quantities.
  // Causal rows in different heads remain separate pieces but share arithmetic.
  std::map<std::vector<long>,TaskPriceParts> equal_prices;
  auto append=[&](analysis::CouplingRelation const& domain,analysis::ParamBinding const& point){
    PricePiece p;p.domain=domain;p.count=domain.ImageCard();p.representative=point;
    std::vector<long> values;for(auto q:quantities)values.push_back(q->BindCoordinates(point).Eval(theta));
    auto found=equal_prices.find(values);
    if(found==equal_prices.end())found=equal_prices.emplace(std::move(values),cost.PriceParts(input,traits,residency,model,chunks,point,residency.ctas_per_sm)).first;
    p.parts=found->second;
    result.total_isolated_ns+=p.count.Eval(theta)*IsolatedNs(p.parts,cal.dram_gbps/(cost.target().res.num_sms*residency.ctas_per_sm));result.pieces.push_back(std::move(p));
  };
  auto partition=[&]{
    auto domain=make_domain();analysis::ParamBinding point;for(std::size_t i=0;i<axes.size();++i)point.Bind(axes[i].name,bounds[i].first);
    bool constant=true;for(auto q:quantities) {
      auto value=q->BindCoordinates(point).Eval(theta);
      if(!q->SumAlong(domain).SemanticallyEqual(domain.Card().Scale(value),theta)){constant=false;break;}
    }
    if(constant){append(domain,point);return;}
    // Causal fibers (and scalar edge chunks) are exact singleton pieces;
    // this fallback is reported, never hidden behind a representative price.
    result.coordinate_varying=true;auto original=bounds;
    std::function<void(std::size_t)> singles=[&](std::size_t a){if(a==axes.size()){analysis::ParamBinding p;for(std::size_t i=0;i<axes.size();++i)p.Bind(axes[i].name,bounds[i].first);append(make_domain(),p);return;}for(long q=original[a].first;q<original[a].second;++q){bounds[a]={q,q+1};singles(a+1);}};
    singles(0);bounds=std::move(original);
  };
  std::function<void(std::size_t)> split=[&](std::size_t a){if(a==axes.size()){partition();return;}for(auto b:axes[a].parts){bounds[a]=b;split(a+1);}};
  if(tasks>0)split(0);long covered=0;for(auto const& p:result.pieces)covered+=p.count.Eval(theta);
  if(covered!=tasks)throw std::runtime_error("boundary pricing does not partition the complete task space");
  if(cache)cache->entries.emplace(key.str(),result);return result;
}
} // namespace tilemega::solver
