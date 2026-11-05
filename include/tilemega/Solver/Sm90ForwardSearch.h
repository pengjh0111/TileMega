// SPDX-License-Identifier: BSD-3-Clause
#pragma once
// Measured per-card forward costs and independent stage search.
#include <tilemega/Solver/Sm90ForwardCoordinates.h>
#include <tilemega/Solver/Sm90NativeConstants.h>
#include <tilemega/Solver/ForwardGeometry.h>
#include <tilemega/Target/TargetSpec.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <map>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::solver {

struct Sm90ForwardMissing:std::invalid_argument {using std::invalid_argument::invalid_argument;};
using Sm90ForwardSelector=std::map<std::string,std::string>;
inline long long ForwardCeil(long long a,long long b) {
  if(a<=0 || b<=0)throw std::invalid_argument("forward geometry extents must be positive");
  return (a+b-1)/b;
}
inline long long ForwardInteger(json::Value const& value,std::string const& name,bool zero=false) {
  double x=value.AsNumber(name.c_str());
  if(!std::isfinite(x) || x<(zero?0:1) || x>double(std::numeric_limits<long long>::max()) || std::floor(x)!=x)
    throw std::invalid_argument("invalid forward geometry integer: "+name);
  return static_cast<long long>(x);
}
inline long long ForwardField(json::Value const& geometry,char const* name,bool zero=false) {
  auto const* value=geometry.Find(name);
  if(!value)throw Sm90ForwardMissing(std::string("missing plan-derived dnn1_geometry.")+name);
  return ForwardInteger(*value,name,zero);
}
inline std::string ForwardString(json::Value const& geometry,char const* name) {
  auto const* value=geometry.Find(name);
  if(!value)throw Sm90ForwardMissing(std::string("missing plan-derived dnn1_geometry.")+name);
  return value->AsString(name);
}

// Resource/layout and budget checks share the parent ForwardGeometry helper.
struct Sm90ForwardConstant {
  double ns=0;std::string source;bool unstable=false;
  json::Value ToJson() const {
    return json::Object{{"value",ns},{"source",source},{"unstable",unstable}};
  }
};

class Sm90ForwardConstants {
 public:
  // Actual collector schema: sm90_forward_constants.by_gpu[gpu][metric][key].
  // Never blend cards: GPU0's externally locked clock differs from GPUs5/7.
  static Sm90ForwardConstants FromFile(std::string file,int gpu) {
    if(gpu!=0 && gpu!=5 && gpu!=7)throw std::invalid_argument("forward constants GPU must be 0, 5 or 7");
    Sm90ForwardConstants c;c.file_=std::move(file);c.gpu_=gpu;c.target_=json::ParseFile(c.file_);
    auto const* section=c.target_.Find("sm90_forward_constants");
    if(!section)throw Sm90ForwardMissing(c.file_+": missing sm90_forward_constants; no measured forward table");
    c.section_=*section;
    double schema=section->At("schema").AsNumber("forward constants schema");
    if(schema!=1 && schema!=2 && schema!=3)
      throw std::invalid_argument("unsupported forward constants schema");
    c.storage_schema_=int(schema);
    if(schema==3)c.ExpandDictionary();
    if(schema>=2)c.ExpandCompact();
    c.variant_=section->At("variant").AsString("forward constants variant");
    if(auto const* fixture=section->Find("synthetic_cpu_fixture"))
      if(fixture->AsBool("synthetic fixture") && std::getenv("DNN1_CPU_CONTRACT_TEST")==nullptr)
        throw std::invalid_argument("synthetic fixture is not calibrated constants");
    c.file_sha256_=HashFile(c.file_);
    return c;
  }
  static Sm90ForwardConstants FromEnvironment() {
    auto* path=std::getenv("TILEMEGA_FORWARD_CONSTANTS_FILE");
    auto* gpu=std::getenv("TILEMEGA_DEVICE_INDEX");
    if(!path || !*path || !gpu || !*gpu)
      throw Sm90ForwardMissing("set TILEMEGA_FORWARD_CONSTANTS_FILE to the actual --solve target JSON and TILEMEGA_DEVICE_INDEX");
    std::string card=gpu;
    if(card!="0" && card!="5" && card!="7")
      throw std::invalid_argument("forward constant selection requires physical GPU0/5/7");
    return FromFile(path,std::stoi(card));
  }
  static Sm90ForwardSelector ParseKey(std::string const& key) {
    Sm90ForwardSelector result;std::istringstream fields(key);std::string field;
    while(std::getline(fields,field,';')) {
      auto equals=field.find('=');
      if(equals==std::string::npos || equals==0 || equals+1==field.size() ||
          !result.emplace(field.substr(0,equals),field.substr(equals+1)).second)
        throw std::invalid_argument("malformed or duplicate forward measurement key: "+key);
    }
    return result;
  }
  Sm90ForwardConstant Get(std::string const& metric,Sm90ForwardSelector const& selector,
                          std::string const& unit) const {
    auto const* cards=section_.Find("by_gpu");
    auto const* card=cards?cards->Find(std::to_string(gpu_)):nullptr;
    auto const* table=card?card->Find(metric):nullptr;
    std::string query;
    for(auto const& [k,v]:selector)query+=(query.empty()?"":";")+k+"="+v;
    if(!table)throw Sm90ForwardMissing(file_+": missing GPU"+std::to_string(gpu_)+" "+metric+"["+query+"]");
    json::Value const* match=nullptr;std::string matched_key;
    for(auto const& [key,row]:table->AsObject("forward measured table")) {
      auto fields=ParseKey(key);bool matches=true;
      // Historical DW keys without a gate describe the ungated primitive.
      // A gated measurement is a different operation, not an extra match
      // for a geometry-only selector. Keep raw keys and medians unchanged.
      if(metric=="dw_tap_ns" && !selector.count("gate") && fields.count("gate"))continue;
      for(auto const& [k,v]:selector) {
        auto found=fields.find(k);
        if(found==fields.end() || found->second!=v){matches=false;break;}
      }
      if(matches) {
        if(match)throw Sm90ForwardMissing("ambiguous exact-geometry constant; refine selector: "+metric+"["+query+"]");
        match=&row;matched_key=key;
      }
    }
    if(!match)throw Sm90ForwardMissing(file_+": no measured "+metric+"["+query+"]; no interpolation or zero fill");
    double value=match->At("median").AsNumber("measured median");
    if(!std::isfinite(value) || !(value>0) || match->At("processes").AsNumber("fresh processes")!=5 ||
        match->At("unit").AsString("constant unit")!=unit)
      throw std::invalid_argument("invalid/insufficient measured constant: "+metric+"["+matched_key+"]");
    auto* unstable=match->Find("unstable");
    std::string source=file_+"#sm90_forward_constants.";
    if(storage_schema_>=2)
      source+="median_by_gpu."+std::to_string(gpu_)+"["+
          std::to_string(compact_indices_.at({metric,matched_key}))+ "] ("+metric+"["+matched_key+"])";
    else source+="by_gpu."+std::to_string(gpu_)+"."+metric+"["+matched_key+"]";
    if(storage_schema_>=2) {
      auto found=compact_provenance_.find(compact_indices_.at({metric,matched_key}));
      if(found!=compact_provenance_.end())source+="; "+found->second;
    }
    return {value,source,
      unstable?unstable->AsBool("unstable"):false};
  }
  // Historical reference is only the original N1 measured eta*P_sm product.
  // This never creates eta, refits a rate, or writes native.forward.
  Sm90ForwardConstant HistoricalWgmma(int n) const {
    auto const* ref=section_.Find("historical_reference");
    if(!ref)throw Sm90ForwardMissing("no recorded historical N1 t_wgmma source");
    auto const& rows=ref->At("t_wgmma").AsArray("historical WGMMA rows");
    for(auto const& row:rows)if(row.At("key").AsString("historical key")=="n="+std::to_string(n)) {
      double value=std::stod(row.At("median").AsString("historical median"));
      if(!(value>0) || row.At("processes").AsString("historical processes")!="5" ||
          row.At("unit").AsString("historical unit")!="flop_per_ns")
        throw std::invalid_argument("invalid historical N1 WGMMA source");
      return {value,ref->At("source").AsString("historical source")+"#t_wgmma[n="+std::to_string(n)+"]",false};
    }
    throw Sm90ForwardMissing("missing historical t_wgmma N"+std::to_string(n));
  }
  json::Value const& Target()const{return target_;}
  json::Value const& Section()const{return section_;}
  std::string const& File()const{return file_;}
  json::Value Receipt() const {
    json::Value result=json::Object{{"file",file_},{"sha256",file_sha256_},{"gpu",gpu_},{"variant",variant_},
      {"storage_schema",storage_schema_},
      {"method",section_.At("method")},{"header_sha256",section_.At("header_sha256")}};
    if(auto const* supplements=section_.Find("supplements"))result.Set("supplements",*supplements);
    return result;
  }
 private:
  // Schema3 shares metric/unit/selector field names; values stay exact strings.
  // Reconstruct schema2 keys without changing key order, median doubles, or IDs.
  void ExpandDictionary() {
    auto const& layouts=section_.At("key_layouts").AsArray("selector layouts");
    if(layouts.empty())throw std::invalid_argument("empty selector layouts");
    json::Array expanded;
    for(auto const& value:section_.At("keys").AsArray("dictionary keys")) {
      auto const& row=value.AsArray("dictionary key");
      if(row.empty())throw std::invalid_argument("empty dictionary key");
      auto index=ForwardInteger(row[0],"selector layout index",true);
      if(std::size_t(index)>=layouts.size())throw std::invalid_argument("selector layout index out of range");
      auto const& layout=layouts[index].AsArray("selector layout");
      if(layout.size()!=3)throw std::invalid_argument("selector layout needs metric/unit/fields");
      auto const& names=layout[2].AsArray("selector fields");
      if(names.empty() || row.size()!=names.size()+1)
        throw std::invalid_argument("dictionary key has wrong field count");
      std::string key;
      for(std::size_t i=0;i<names.size();++i) {
        auto const& name=names[i].AsString("selector field");
        auto const& field=row[i+1].AsString("selector exact value");
        if(name.empty() || name.find_first_of(";=")!=std::string::npos ||
            field.empty() || field.find(';')!=std::string::npos)
          throw std::invalid_argument("invalid selector dictionary field");
        key+=(i?";":"")+name+"="+field;
      }
      ParseKey(key); // Also reject duplicate fields before normal compact checks.
      expanded.emplace_back(json::Array{layout[0],std::move(key),layout[1]});
    }
    section_.Set("keys",std::move(expanded));
  }
  void ExpandCompact() {
    auto const& keys=section_.At("keys").AsArray("compact constant keys");
    auto const& cards=section_.At("median_by_gpu").AsObject("compact card medians");
    if(section_.At("processes").AsNumber("fresh processes")!=5 || keys.empty() || cards.size()!=3)
      throw std::invalid_argument("compact constants require five processes and all three cards");
    struct Key {std::string metric,key,unit;};std::vector<Key> parsed;
    for(std::size_t i=0;i<keys.size();++i) {
      auto const& row=keys[i].AsArray("compact constant key");
      if(row.size()!=3)throw std::invalid_argument("compact key needs metric/key/unit");
      Key k{row[0].AsString("metric"),row[1].AsString("key"),row[2].AsString("unit")};
      ParseKey(k.key);
      if(k.metric.empty() || k.unit.empty() || !compact_indices_.emplace(std::make_pair(k.metric,k.key),i).second)
        throw std::invalid_argument("empty or duplicate compact constant key");
      parsed.push_back(std::move(k));
    }
    if(auto const* supplements=section_.Find("supplements"))
      for(auto const& extra:supplements->AsArray("supplemental calibrations")) {
        auto provenance="calibration_variant="+extra.At("variant").AsString("supplement variant")+
            "; collection_receipt_sha256="+extra.At("raw_sources").At("collection_receipt.json").At("sha256").AsString("supplement receipt SHA");
        for(auto const& index:extra.At("key_indices").AsArray("supplement key indices")) {
          auto i=ForwardInteger(index,"supplement key index",true);
          if(std::size_t(i)>=parsed.size() || !compact_provenance_.emplace(std::size_t(i),provenance).second)
            throw std::invalid_argument("invalid or duplicate supplement constant index");
        }
      }
    std::map<int,std::map<std::pair<std::string,std::string>,bool>> unstable;
    for(auto const& row:section_.At("unstable").AsArray("unstable constants")) {
      int gpu=int(ForwardInteger(row.At("gpu"),"unstable GPU",true));
      auto key=std::make_pair(row.At("metric").AsString("unstable metric"),row.At("key").AsString("unstable key"));
      if((gpu!=0 && gpu!=5 && gpu!=7) || !compact_indices_.count(key))
        throw std::invalid_argument("unknown unstable compact constant");
      unstable[gpu][key]=true;
    }
    json::Object expanded;std::map<std::string,bool> seen;
    for(auto const& [gpu,values]:cards) {
      if((gpu!="0" && gpu!="5" && gpu!="7") || !seen.emplace(gpu,true).second)
        throw std::invalid_argument("invalid or duplicate compact GPU");
      auto const& medians=values.AsArray("compact medians");
      if(medians.size()!=parsed.size())throw std::invalid_argument("compact median count differs from keys");
      std::map<std::string,json::Object> metrics;
      for(std::size_t i=0;i<parsed.size();++i) {
        auto const& key=parsed[i];double value=medians[i].AsNumber("measured median");
        if(!std::isfinite(value) || value<=0)throw std::invalid_argument("invalid compact measured median");
        metrics[key.metric].emplace_back(key.key,json::Object{{"median",value},{"processes",5},
            {"unit",key.unit},{"unstable",unstable[std::stoi(gpu)][{key.metric,key.key}]}});
      }
      json::Object card;for(auto& [metric,rows]:metrics)card.emplace_back(metric,std::move(rows));
      expanded.emplace_back(gpu,std::move(card));
    }
    section_.Set("by_gpu",std::move(expanded));
  }
  static std::string HashFile(std::string const& path) {
    std::string quoted="'";for(char c:path)quoted+=c=='\''?"'\\''":std::string(1,c);quoted+="'";
    auto* pipe=popen(("sha256sum -- "+quoted).c_str(),"r");
    if(!pipe)throw std::runtime_error("cannot hash constants file");
    char buffer[65]{};auto* got=fgets(buffer,sizeof(buffer),pipe);int status=pclose(pipe);
    if(!got || status || std::string(buffer).size()!=64)
      throw std::runtime_error("constants file hash failed");
    return buffer;
  }
  int gpu_=0,storage_schema_=1;std::string file_,file_sha256_,variant_;json::Value target_,section_;
  std::map<std::pair<std::string,std::string>,std::size_t> compact_indices_;
  std::map<std::size_t,std::string> compact_provenance_;
};

struct Sm90ForwardPrediction {
  struct Stage {std::string body;double ns=0;long long tasks=0;json::Value terms;};
  std::vector<Stage> stages;double total_ns=0,legacy_total_ns=0;
  json::Value constants_receipt,search_receipt;
  json::Value ToJson() const {
    json::Array rows;
    for(auto const& s:stages)rows.emplace_back(json::Object{{"body",s.body},{"tasks",double(s.tasks)},
      {"stage_ns",s.ns},{"terms",s.terms}});
    return json::Object{{"forward_ns",total_ns},{"legacy_forward_ns",legacy_total_ns},
      {"stages",std::move(rows)},{"forward_constants",constants_receipt},
      {"search_receipt",search_receipt}};
  }
};

class Sm90ForwardCostModel {
 public:
  // Keep the existing native_forward.cpp two-argument public constructor.
  // The allowed Python/script harness binds the actual --solve target JSON
  // through TILEMEGA_FORWARD_CONSTANTS_FILE; TargetSpec::ToJson loses extras.
  Sm90ForwardCostModel(TargetSpec target,Sm90NativeConstants constants)
      :target_(std::move(target)),native_(std::move(constants)),
       forward_(Sm90ForwardConstants::FromEnvironment()) {
    if(target_.arch_tag!="sm_90" || target_.res.num_sms<=0)
      throw std::invalid_argument("forward model needs a complete H800 target");
    if(forward_.Target().At("sm90_native_constants").Dump(0)!=native_.section.Dump(0))
      throw std::invalid_argument("environment forward JSON and actual --solve native constants differ");
    auto const& resources=forward_.Target().At("resources");
    if(resources.At("num_sms").AsNumber("constant target SMs")!=target_.res.num_sms ||
        resources.At("max_dynamic_smem_per_cta").AsNumber("constant target shared")!=target_.res.max_dynamic_smem_per_cta)
      throw std::invalid_argument("environment forward JSON and actual --solve resources differ");
  }
  int Grid(int residency)const {
    if(residency!=1 && residency!=2)throw std::invalid_argument("forward residency must be 1 or 2");
    return target_.res.num_sms*residency;
  }
  static long long Tasks(Sm90ForwardShape const& s,Sm90ForwardStage const& c) {
    auto const& g=s.dnn1_geometry;
    switch(s.body) {
      case Sm90ForwardBody::kGemm:case Sm90ForwardBody::kConv:
        return ForwardCeil(s.m,c.tile_m)*ForwardCeil(s.n,c.tile_n)*c.split_k;
      case Sm90ForwardBody::kDepthwise:
        return ForwardField(g,"batch")*ForwardCeil(ForwardField(g,"output_height"),c.tile_h)*
            ForwardCeil(ForwardField(g,"output_width"),c.tile_w)*ForwardCeil(s.channels,c.channel_block);
      case Sm90ForwardBody::kDwPw:case Sm90ForwardBody::kInvertedResidual:
        return ForwardField(g,"batch")*ForwardCeil(ForwardField(g,"output_height"),c.tile_h)*
            ForwardCeil(ForwardField(g,"output_width"),c.tile_w)*ForwardCeil(s.out_channels,c.tile_n)*c.split_k;
      case Sm90ForwardBody::kEncoderAttention: {
        auto seq=ForwardField(g,"seq");
        if(seq%c.tile_m)throw std::invalid_argument("attention query rows must divide sequence");
        return ForwardField(g,"batch")*ForwardField(g,"heads")*(seq/c.tile_m);
      }
      case Sm90ForwardBody::kLayout:
        return ForwardField(g,"batch")*ForwardCeil(ForwardField(g,"width")*ForwardField(g,"height"),128);
      case Sm90ForwardBody::kPool:
        return ForwardCeil(ForwardField(g,"batch")*ForwardField(g,"output_width")*
            ForwardField(g,"output_height")*s.channels,128);
      case Sm90ForwardBody::kGlobalPool:
        return ForwardField(g,"batch")*ForwardCeil(s.channels,128);
      case Sm90ForwardBody::kGemv:return s.m*ForwardCeil(s.n,c.tile_n);
      case Sm90ForwardBody::kLayerNorm:case Sm90ForwardBody::kEmbedding:
        return ForwardCeil(ForwardField(g,"rows"),4);
    }
    throw std::invalid_argument("unknown forward body");
  }
  double Stage(Sm90ForwardShape const& s,Sm90ForwardStage const& c,int residency,
               json::Value* terms=nullptr) const;
  double LegacyStage(Sm90ForwardShape const& s,Sm90ForwardStage const& c,int residency,
                     json::Value* terms=nullptr) const;
  Sm90ForwardPrediction Predict(std::vector<Sm90ForwardShape> const& shapes,
                               Sm90ForwardCoordinates const& c) const {
    if(shapes.size()!=c.stages.size())throw std::invalid_argument("coordinate stage count differs");
    Sm90ForwardPrediction result;result.constants_receipt=forward_.Receipt();
    for(std::size_t i=0;i<shapes.size();++i) {
      Sm90ForwardPrediction::Stage p{kSm90ForwardBodies[unsigned(shapes[i].body)],0,
          Tasks(shapes[i],c.stages[i]),{}};
      p.ns=Stage(shapes[i],c.stages[i],c.residency,&p.terms);
      json::Value legacy_terms;
      double legacy=LegacyStage(shapes[i],c.stages[i],c.residency,&legacy_terms);
      p.terms.Set("legacy_diagnostic",legacy_terms);result.legacy_total_ns+=legacy;
      result.total_ns+=p.ns;result.stages.push_back(std::move(p));
    }
    return result;
  }
  Sm90NativeConstants const& Constants()const{return native_;}
  Sm90ForwardConstants const& ForwardConstants()const{return forward_;}
 private:
  double Epilogue(Sm90ForwardShape const&,Sm90ForwardStage const&,Sm90ForwardResourceUsage const&,
                  json::Array&)const;
  double BodyCost(Sm90ForwardShape const&,Sm90ForwardStage const&,Sm90ForwardResourceUsage const&,
                  json::Array&)const;
  struct TaskWork {long long async_bytes,dram_bytes,elements;};
  static TaskWork TaskBytes(Sm90ForwardShape const&,Sm90ForwardStage const&);
  TargetSpec target_;Sm90NativeConstants native_;Sm90ForwardConstants forward_;
};

struct Sm90ForwardSearchSpace {
  std::vector<Sm90ForwardShape> shapes;TargetSpec target;
  int Grid(int residency)const{return target.res.num_sms*residency;}
  std::vector<Sm90ForwardStage> Candidates(std::size_t index,int residency)const;
};
struct Sm90ForwardCandidate {
  Sm90ForwardCoordinates coordinates;Sm90ForwardPrediction prediction;
  std::string origin;json::Value search_receipt;
};
struct Sm90ForwardSearchResult {
  std::vector<Sm90ForwardCandidate> ranked;long long evaluated=0;
  double elapsed_ms=0;json::Value receipt;
};
Sm90ForwardSearchResult SearchSm90Forward(Sm90ForwardSearchSpace const&,
    Sm90ForwardCostModel const&,int top=4);

}  // namespace tilemega::solver
