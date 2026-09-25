// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/HopCurve.h>
#include <memory>
namespace tilemega::solver {
struct FlowPiece {int count=0;TaskPriceParts parts;};
struct FlowSpace {
  std::string name,category;
  int count=0,order=0;
  double rank_ns=0;
  std::vector<FlowPiece> pieces;
  std::vector<int> piece_of_task;
};
struct FlowRelease {int producer=0,consumer=0,kappa=1;bool all_producer=false,colocated=false;
  std::shared_ptr<std::vector<std::pair<int,int>> const> sorted; // (coarsened maximum predecessor, consumer)
};
struct FlowProblem {
  std::vector<FlowSpace> spaces;
  std::vector<FlowRelease> edges;
  int workers=0;
  double dram_gbps=0,publication_ns=0,consumer_wait_ns=0,hop_ns=0,dram_floor_ns=0,floor_ns=0;
  bool all_external_miss=false;
  bool inflight_dram=false;
  std::vector<double> inflight_curve_bytes,inflight_curve_gbps;
  std::vector<double> cta_stream_curve_bytes,cta_stream_curve_gbps;
};
struct FlowOptions {bool no_sync=false,no_fixed=false,infinite_workers=false,no_external=false;};
struct FlowSpaceResult {double first_start=-1,last_end=0,wait_ns=0,fixed_ns=0,mainloop_ns=0,publication_ns=0;int last_edge=-1;};
struct FlowChainLink {int space=-1,task=-1,edge=-1;double start_ns=0,end_ns=0,wait_ns=0,fixed_ns=0,mainloop_ns=0,publication_ns=0,hop_ns=0;};
struct FlowResult {
  double makespan_ns=0,delivered_bytes=0;
  std::vector<FlowSpaceResult> spaces;
  std::vector<int> critical_chain;
  std::vector<FlowChainLink> critical_links;
};
struct FlowDecomposition {
  FlowResult original;
  double no_sync=0,no_fixed=0,infinite=0,no_external=0;
  double synchronization=0,fixed=0,contention=0,chain=0,pg_upper_bound=0;
};
int CoarsenRelease(int maximum,int producer_tasks,int kappa);
void SetFlowCalibration(FlowProblem& problem,TargetSpec const& target,ScalarType dtype,HopCurve const& hop);
FlowResult EvaluateFlow(FlowProblem const& problem,FlowOptions const& options={});
FlowDecomposition DecomposeFlow(FlowProblem const& problem);
} // namespace tilemega::solver
