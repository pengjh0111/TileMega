// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <tilemega/Support/Json.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace tilemega::solver {

struct MoeRoutingPoint {
  std::uint32_t tokens=0,experts=0,top_k=0;
  std::uint64_t windows=0;
  double expected_distinct_experts=0;
  std::vector<std::map<std::uint32_t,std::uint64_t>> tokens_per_expert;
  std::map<std::uint32_t,std::map<std::uint32_t,std::uint64_t>> group_blocks;
  std::map<std::uint32_t,std::vector<std::uint64_t>> virtual_rows;
  std::map<std::uint32_t,std::vector<std::map<std::uint32_t,std::uint64_t>>> virtual_row_histograms;

  std::uint64_t SlotCapacity() const {return std::uint64_t(tokens)*top_k;}
  std::uint64_t GroupCapacity(std::uint32_t block_rows) const {
    if(!block_rows)throw std::invalid_argument("zero MoE binding block size");
    auto slots=SlotCapacity();
    return slots/block_rows+(slots%block_rows!=0)+std::min<std::uint64_t>(experts,slots);
  }
  double ExpectedGroupBlocks(std::uint32_t block_rows) const {
    if(!block_rows || !windows)throw std::invalid_argument("invalid MoE routing point");
    long double blocks=0;
    for(auto const& histogram:tokens_per_expert)
      for(auto const& [rows,count]:histogram)
        blocks+=(std::uint64_t(rows)/block_rows+(rows%block_rows!=0))*
                static_cast<long double>(count);
    return double(blocks/windows);
  }
  double ActiveVirtualProbability(std::uint32_t block_rows,std::uint64_t virtual_id) const {
    if(virtual_id>=GroupCapacity(block_rows))throw std::out_of_range("virtual MoE id exceeds capacity");
    auto found=group_blocks.find(block_rows);
    if(found==group_blocks.end() || !windows)
      throw std::out_of_range("unprofiled MoE binding block size");
    std::uint64_t active=0;
    for(auto const& [blocks,count]:found->second)if(blocks>virtual_id)active+=count;
    return double(active)/windows;
  }
  double ExpectedVirtualRows(std::uint32_t block_rows,std::uint64_t virtual_id) const {
    auto found=virtual_rows.find(block_rows);
    if(found==virtual_rows.end() || virtual_id>=found->second.size() || !windows)
      throw std::out_of_range("unprofiled MoE virtual row coordinate");
    return double(found->second[virtual_id])/windows;
  }
  // BM and the GEMM row tile are independent. A mean alone cannot price the
  // probability that a later row subtile is nonempty.
  double ActiveVirtualSubtileProbability(std::uint32_t block_rows,std::uint64_t virtual_id,
                                        std::uint32_t row_begin) const {
    auto const& histogram=VirtualRowHistogram(block_rows,virtual_id);
    if(row_begin>=block_rows)throw std::out_of_range("virtual MoE row subtile exceeds binding block");
    std::uint64_t active=0;
    for(auto const& [rows,count]:histogram)if(rows>row_begin)active+=count;
    return double(active)/windows;
  }
  double ExpectedVirtualSubtileRows(std::uint32_t block_rows,std::uint64_t virtual_id,
                                   std::uint32_t row_begin,std::uint32_t row_count) const {
    auto const& histogram=VirtualRowHistogram(block_rows,virtual_id);
    if(row_begin>=block_rows || !row_count)
      throw std::out_of_range("invalid virtual MoE row subtile");
    long double total=0;
    for(auto const& [rows,count]:histogram)if(rows>row_begin)
      total+=std::min(row_count,rows-row_begin)*static_cast<long double>(count);
    return double(total/windows);
  }
  std::map<std::uint32_t,std::uint64_t> const& VirtualRowHistogram(
      std::uint32_t block_rows,std::uint64_t virtual_id) const {
    auto found=virtual_row_histograms.find(block_rows);
    if(found==virtual_row_histograms.end() || virtual_id>=found->second.size() || !windows)
      throw std::out_of_range("unprofiled virtual MoE row distribution");
    return found->second[virtual_id];
  }
  // Unique-weight traffic is a DRAM lower bound for either binding policy.
  // Slot reads may reuse those weights in cache, so are a separate work quantity.
  double UniqueExpertWeightBytes(std::uint64_t bytes_per_expert) const {
    if(!bytes_per_expert)throw std::invalid_argument("zero expert weight size");
    return expected_distinct_experts*bytes_per_expert;
  }
  double SlotWeightReadBytes(std::uint64_t bytes_per_expert) const {
    if(!bytes_per_expert)throw std::invalid_argument("zero expert weight size");
    return double(SlotCapacity())*bytes_per_expert;
  }
  std::uint64_t GatheredAReadBytes(std::uint32_t width,std::uint32_t element_bytes) const {
    auto elements=SlotCapacity();
    for(auto factor:{width,element_bytes}) {
      if(!factor || elements>std::numeric_limits<std::uint64_t>::max()/factor)
        throw std::overflow_error("MoE gathered operand byte count is invalid");
      elements*=factor;
    }
    return elements;
  }
};

struct MoeRoutingProfile {
  std::string profile_id;
  std::uint32_t experts=0,top_k=0;
  std::vector<std::map<std::uint32_t,MoeRoutingPoint>> layers;

  MoeRoutingPoint const& At(std::uint32_t layer,std::uint32_t tokens) const {
    if(layer>=layers.size())throw std::out_of_range("unprofiled MoE layer");
    auto found=layers[layer].find(tokens);
    if(found==layers[layer].end())throw std::out_of_range("unprofiled MoE token count");
    return found->second;
  }
  static MoeRoutingProfile Read(json::Value const& value,std::uint32_t layer_count,
                               std::uint32_t expected_experts,std::uint32_t expected_top_k) {
    auto integer=[](json::Value const& v) {
      double number=v.AsNumber("MoE profile integer");
      if(!std::isfinite(number) || number<0 || number>9007199254740991. || std::floor(number)!=number)
        throw std::invalid_argument("MoE profile count is not an exact nonnegative integer");
      return std::uint64_t(number);
    };
    auto key=[](std::string const& text) {
      if(text.empty() || (text.size()>1 && text[0]=='0') ||
         std::any_of(text.begin(),text.end(),[](char c){return c<'0' || c>'9';}))
        throw std::invalid_argument("noncanonical MoE histogram coordinate");
      auto n=std::stoull(text);
      if(n>std::numeric_limits<std::uint32_t>::max())
        throw std::invalid_argument("MoE histogram coordinate overflows");
      return std::uint32_t(n);
    };
    if(value.At("schema").AsString("schema")!="tilemega.dm1.routing.profile.v1" ||
       value.At("evidence").AsString("evidence")!="verified")
      throw std::invalid_argument("unsupported or unverified MoE routing profile");
    auto const& sampling=value.At("sampling");
    if(!layer_count || !expected_experts || !expected_top_k || expected_top_k>expected_experts ||
       integer(sampling.At("layers"))!=layer_count || integer(sampling.At("experts"))!=expected_experts ||
       integer(sampling.At("top_k"))!=expected_top_k)
      throw std::invalid_argument("routing profile differs from model configuration");
    MoeRoutingProfile result;
    result.profile_id=value.At("profile_id").AsString("profile_id");
    if(result.profile_id.size()!=64 || std::any_of(result.profile_id.begin(),result.profile_id.end(),
        [](char c){return !(c>='0' && c<='9') && !(c>='a' && c<='f');}))
      throw std::invalid_argument("invalid MoE profile identity");
    result.experts=expected_experts;result.top_k=expected_top_k;result.layers.resize(layer_count);
    auto const& layers=value.At("layers").AsArray("layers");
    if(layers.size()!=layer_count)throw std::invalid_argument("incomplete MoE routing layers");
    std::vector<bool> seen(layer_count);
    for(auto const& layer:layers) {
      auto index=integer(layer.At("layer"));
      if(index>=layer_count || seen[index])throw std::invalid_argument("duplicate or invalid MoE routing layer");
      seen[index]=true;
      for(auto const& [token_key,coordinate]:layer.At("coordinates").AsObject("coordinates")) {
        MoeRoutingPoint point;point.tokens=key(token_key);point.experts=expected_experts;point.top_k=expected_top_k;
        point.windows=integer(coordinate.At("windows"));
        if(!point.tokens || !point.windows || integer(coordinate.At("tokens"))!=point.tokens ||
           integer(coordinate.At("assignments_per_window"))!=point.SlotCapacity())
          throw std::invalid_argument("invalid MoE routing coordinate");
        auto histogram=[&](json::Value const& h,std::uint32_t maximum,std::uint32_t minimum=0) {
          std::map<std::uint32_t,std::uint64_t> counts;std::uint64_t total=0;
          for(auto const& [label,v]:h.AsObject("histogram")) {
            auto coordinate=key(label);auto count=integer(v);
            if(coordinate<minimum || coordinate>maximum || !count ||
               count>point.windows-total || !counts.emplace(coordinate,count).second)
              throw std::invalid_argument("invalid MoE histogram bin");
            total+=count;
          }
          if(total!=point.windows)throw std::invalid_argument("MoE histogram omits windows");
          return counts;
        };
        auto distinct=histogram(coordinate.At("distinct_experts_histogram"),
            std::uint32_t(std::min<std::uint64_t>(expected_experts,point.SlotCapacity())),expected_top_k);
        long double distinct_sum=0;
        for(auto const& [n,count]:distinct)distinct_sum+=n*static_cast<long double>(count);
        point.expected_distinct_experts=coordinate.At("expected_distinct_experts").AsNumber("distinct mean");
        if(!std::isfinite(point.expected_distinct_experts) ||
           std::abs(point.expected_distinct_experts-double(distinct_sum/point.windows))>1e-10)
          throw std::invalid_argument("MoE distinct mean differs from histogram");
        auto const& per_expert=coordinate.At("tokens_per_expert_histograms").AsArray("expert histograms");
        if(per_expert.size()!=expected_experts)throw std::invalid_argument("MoE profile omits experts");
        long double assignments=0,active_experts=0;
        for(auto const& h:per_expert) {
          auto counts=histogram(h,point.tokens);
          for(auto const& [n,count]:counts) {
            assignments+=n*static_cast<long double>(count);
            if(n)active_experts+=count;
          }
          point.tokens_per_expert.push_back(std::move(counts));
        }
        if(assignments!=point.SlotCapacity()*static_cast<long double>(point.windows) || active_experts!=distinct_sum)
          throw std::invalid_argument("MoE expert histograms violate assignment conservation");
        if(auto groups=coordinate.Find("group_blocks_histograms")) {
          for(auto const& [label,bins]:groups->AsObject("group block histograms")) {
            auto block_rows=key(label);
            if(!block_rows)throw std::invalid_argument("zero MoE binding block size");
            auto maximum=std::min<std::uint64_t>(point.SlotCapacity(),point.GroupCapacity(block_rows));
            if(maximum>std::numeric_limits<std::uint32_t>::max())
              throw std::overflow_error("MoE group profile exceeds runtime task capacity");
            auto minimum=point.SlotCapacity()/block_rows+(point.SlotCapacity()%block_rows!=0);
            auto counts=histogram(bins,std::uint32_t(maximum),std::uint32_t(minimum));
            long double joint=0,marginal=0;
            for(auto const& [blocks,count]:counts)joint+=blocks*static_cast<long double>(count);
            for(auto const& h:point.tokens_per_expert)
              for(auto const& [rows,count]:h)
                marginal+=(std::uint64_t(rows)/block_rows+(rows%block_rows!=0))*
                    static_cast<long double>(count);
            if(joint!=marginal || !point.group_blocks.emplace(block_rows,std::move(counts)).second)
              throw std::invalid_argument("MoE joint block distribution differs from expert marginals");
          }
        }
        if(auto totals=coordinate.Find("virtual_rows_totals")) {
          for(auto const& [label,values]:totals->AsObject("virtual row totals")) {
            auto b=key(label);auto groups=point.group_blocks.find(b);
            if(groups==point.group_blocks.end())
              throw std::invalid_argument("virtual row totals lack prefix activity");
            auto const& array=values.AsArray("virtual rows");
            if(array.size()!=point.GroupCapacity(b))
              throw std::invalid_argument("virtual row totals differ from capacity");
            std::vector<std::uint64_t> rows;long double sum=0;
            for(std::size_t v=0;v<array.size();++v) {
              auto count=integer(array[v]);std::uint64_t active=0;
              for(auto const& [blocks,windows]:groups->second)if(blocks>v)active+=windows;
              if(count<active || count>b*static_cast<long double>(active))
                throw std::invalid_argument("virtual rows differ from prefix activity");
              sum+=count;rows.push_back(count);
            }
            if(sum!=point.SlotCapacity()*static_cast<long double>(point.windows) ||
               !point.virtual_rows.emplace(b,std::move(rows)).second)
              throw std::invalid_argument("virtual rows violate assignment conservation");
          }
        }
        if(auto distributions=coordinate.Find("virtual_rows_histograms")) {
          for(auto const& [label,values]:distributions->AsObject("virtual row distributions")) {
            auto b=key(label);auto totals=point.virtual_rows.find(b);
            if(totals==point.virtual_rows.end())
              throw std::invalid_argument("virtual row distribution lacks its mean");
            auto const& array=values.AsArray("virtual row distributions");
            if(array.size()!=point.GroupCapacity(b))
              throw std::invalid_argument("virtual row distribution differs from capacity");
            std::vector<std::map<std::uint32_t,std::uint64_t>> histograms;
            for(std::size_t v=0;v<array.size();++v) {
              std::map<std::uint32_t,std::uint64_t> bins;std::uint64_t active=0;long double sum=0;
              for(auto const& [row_label,frequency]:array[v].AsObject("virtual row histogram")) {
                auto rows=key(row_label);auto count=integer(frequency);
                if(!rows || rows>b || !count || count>point.windows-active || !bins.emplace(rows,count).second)
                  throw std::invalid_argument("invalid virtual row histogram bin");
                active+=count;sum+=rows*static_cast<long double>(count);
              }
              std::uint64_t expected_active=0;
              for(auto const& [blocks,windows]:point.group_blocks.at(b))if(blocks>v)expected_active+=windows;
              if(active!=expected_active || sum!=totals->second[v])
                throw std::invalid_argument("virtual row distribution differs from activity or mean");
              histograms.push_back(std::move(bins));
            }
            if(!point.virtual_row_histograms.emplace(b,std::move(histograms)).second)
              throw std::invalid_argument("duplicate virtual row distribution block size");
          }
        }
        if(!result.layers[index].emplace(point.tokens,std::move(point)).second)
          throw std::invalid_argument("duplicate MoE token coordinate");
      }
      if(result.layers[index].empty())throw std::invalid_argument("MoE layer has no routing observations");
    }
    return result;
  }
};

}  // namespace tilemega::solver
