// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Support/Json.h>
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::solver {
// Composite, observed software endpoints. This reader never turns a steady
// period into an unmeasured finite span or chooses a neighboring geometry.
// The current aggregation is a typical CTA diagnostic, not a critical-stage
// measurement. New flow calibrations must remain pending independent refit.
class Sm90DenseJointRingProfiles {
 public:
  static std::vector<std::string> const& Dimensions() {
    static std::vector<std::string> const keys={"batch","tile_n","tile_k","slots",
        "warps_n","warps_k","register_stages","grid","active_ctas","compiled_residency","input_k",
        "tasks_per_cta","prefix_bytes","prefetch_l2_bytes","dynamic_shared_bytes",
        "workspace_offset"};
    return keys;
  }
  Sm90DenseJointRingProfiles(json::Value const& value,std::string const& expected_producer_header) {
    if(value.At("schema").AsNumber("joint-ring schema")!=1 ||
       value.At("endpoint_model").AsString("joint-ring endpoint model")!=
           "producer_call_to_finalize_observed_v1")
      throw std::invalid_argument("unsupported native joint-ring profile schema/endpoints");
    binding_=value.At("source_binding");
    for(auto name:{"source_digest","producer_header_sha256","probe_sha256",
        "binary_sha256","collector_sha256"}) {
      auto const& digest=binding_.At(name).AsString(name);
      if(digest.size()!=64 || digest.find_first_not_of("0123456789abcdef")!=std::string::npos)
        throw std::invalid_argument("invalid native joint-ring source binding");
    }
    if(binding_.At("producer_header_sha256").AsString("joint-ring producer header")!=expected_producer_header)
      throw std::invalid_argument("native joint-ring producer header differs from calibration");
    std::set<std::string> ids,coordinates;
    for(auto const& row:value.At("profiles").AsArray("joint-ring profiles")) {
      Entry entry;entry.row=row;
      auto id=row.At("profile_id").AsString("joint-ring profile id");
      if(id.empty() || !ids.insert(id).second)
        throw std::invalid_argument("duplicate/empty native joint-ring profile id");
      std::string coordinate;
      for(auto const& name:Dimensions()) {
        double n=row.At(name).AsNumber(name.c_str());
        if(!std::isfinite(n) || n!=std::floor(n) || n<0 || n>2147483647 ||
            (n==0 && name!="prefix_bytes" && name!="prefetch_l2_bytes" && name!="workspace_offset"))
          throw std::invalid_argument("invalid native joint-ring dimension: "+name);
        entry.dimensions[name]=n;coordinate+='|'+name+'='+std::to_string(n);
      }
      auto const& d=entry.dimensions;
      entry.cache=row.At("cache_state").AsString("joint-ring cache state");
      auto evidence=row.At("evidence").AsString("joint-ring evidence");
      double processes=Number(row,"processes",false);
      if(entry.cache!="four_shape_replay_no_scrub" ||
         row.At("mma").AsString("joint-ring MMA")!="mma_sync" ||
         row.At("aggregation").AsString("joint-ring aggregation")!="per_cta_sample_median" ||
         (evidence!="verified" && evidence!="synthetic_cpu_fixture") ||
         processes<5 || processes!=std::floor(processes) ||
         row.At("continuous_task_drain").AsBool("joint-ring task drain") ||
         d.at("tasks_per_cta")!=1 || d.at("prefix_bytes") || d.at("prefetch_l2_bytes") ||
         d.at("compiled_residency")!=1 ||
         d.at("active_ctas")>d.at("grid") || d.at("warps_n")*d.at("warps_k")!=4 ||
         d.at("input_k")<d.at("tile_k") || std::fmod(d.at("input_k"),d.at("tile_k")) ||
         (d.at("slots")<2 || d.at("slots")>8) ||
         (d.at("batch")!=1 && d.at("batch")!=2 && d.at("batch")!=4 &&
             d.at("batch")!=8 && d.at("batch")!=16) ||
         (d.at("warps_n")!=1 && d.at("warps_n")!=2 && d.at("warps_n")!=4) ||
         std::fmod(d.at("tile_n"),8*d.at("warps_n")) || std::fmod(d.at("tile_k"),64) ||
         (d.at("register_stages")!=1 && d.at("register_stages")!=2 && d.at("register_stages")!=4) ||
         d.at("register_stages")>d.at("tile_k")/(16*d.at("warps_k")) ||
         std::fmod(d.at("workspace_offset"),1024) ||
         d.at("dynamic_shared_bytes")<d.at("workspace_offset")+1024+
             2*d.at("slots")*d.at("tile_k")*(16+d.at("tile_n"))+64*d.at("warps_k")*d.at("tile_n")+128)
        throw std::invalid_argument("unsupported native joint-ring profile scope/evidence");
      coordinate+='|'+entry.cache;
      if(!coordinates.insert(coordinate).second)
        throw std::invalid_argument("duplicate native joint-ring profile coordinates");
      double total=Number(row,"issue_to_finalize_ns",false);
      double first=Number(row,"first_full_ns",false);
      double main=Number(row,"main_span_ns",true);
      double tail=Number(row,"finalize_tail_ns",true);
      if(first>total || main>total || tail>total)
        throw std::invalid_argument("invalid native joint-ring diagnostic span");
      // These are marginal medians of paired intervals. Their sum need not
      // equal the paired total median; only the total is an additive price.
      auto const& prefix=row.At("finite_prefix").AsArray("joint-ring finite prefix");
      int count=std::min(int(d.at("slots"))+1,int(d.at("input_k")/d.at("tile_k")));
      if(int(prefix.size())!=count)
        throw std::invalid_argument("incomplete native joint-ring cold-fill prefix");
      double prior=0;
      for(int q=0;q<count;++q) {
        auto const& point=prefix[q];
        double full=Number(point,"full_ns",false),release=Number(point,"release_ns",false);
        if(point.At("sequence").AsNumber("joint-ring sequence")!=q || full<prior ||
            release<full || release>total || (!q && full!=first))
          throw std::invalid_argument("nonmonotone native joint-ring cold-fill prefix");
        prior=release;
      }
      entry.row.Set("source_binding",binding_);entries_.push_back(std::move(entry));
    }
  }
  json::Value const* Find(std::map<std::string,double> const& dimensions,
      std::string const& cache,std::string const& mma="mma_sync") const {
    for(auto const& entry:entries_)
      if(entry.dimensions==dimensions && entry.cache==cache && mma=="mma_sync")return &entry.row;
    return nullptr;
  }
 private:
  struct Entry {
    std::map<std::string,double> dimensions;
    std::string cache;
    json::Value row;
  };
  std::vector<Entry> entries_;
  json::Value binding_;
  static double Number(json::Value const& value,char const* name,bool zero) {
    double n=value.At(name).AsNumber(name);
    if(!std::isfinite(n) || (zero?n<0:n<=0))
      throw std::invalid_argument(std::string("invalid native joint-ring interval: ")+name);
    return n;
  }
};
}  // namespace tilemega::solver
