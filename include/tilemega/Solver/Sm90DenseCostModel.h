// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/Sm90DenseCoordinates.h>
#include <tilemega/Target/TargetSpec.h>
#include <map>

namespace tilemega::solver {
struct Sm90DenseWorkload {
  std::array<std::array<int,2>,5> shapes;
  int batch=0,layers=0,head_dim=0,queries=0;
  bool qk_norm=false;
};
struct Sm90DensePrediction {
  struct Stage {
    std::string kind,key,protocol_key;
    int instances=1,tasks=0,waves=0;
    double ns=0,barrier=0,startup=0,read_startup=0,initialization=0,dram=0,stream=0,flow=0,
        activation=0,compute=0,epilogue=0,overlap=0,issue=0,
        overlap_limit=0,preissue_service=0,late_service=0,l2_service=0,issue_count=0,issue_service=0,
        receive_count=0,receive=0;
  };
  std::vector<Stage> stages;
  double step_ns=0;
  json::Value ToJson() const;
};

// Reads only the new native section. Legacy target serialization and pricing
// never see these fits; missing measurements or unfit bodies are hard errors.
class Sm90DenseCostModel {
 public:
  Sm90DenseCostModel(TargetSpec target,json::Value const& calibration);
  Sm90DensePrediction Predict(Sm90DenseWorkload const&,Sm90DenseCoordinates const&,
                             int past=575) const;
  Sm90DensePrediction Request(Sm90DenseWorkload const&,Sm90DenseCoordinates const&) const;
 private:
  TargetSpec target_;
  std::map<std::string,std::vector<json::Value>> observations_;
  std::map<std::string,json::Value> fits_;
  mutable std::map<std::string,double> quantities_;
  double dram_bw_=0;
  double Quantity(std::string const& metric,std::map<std::string,double> const& dimensions,
                  std::map<std::string,std::string> const& filters={}) const;
  double MainloopNs(Sm90DenseGemm const&) const;
  double Coefficient(std::string const& body,char const* name) const;
};
}  // namespace tilemega::solver
