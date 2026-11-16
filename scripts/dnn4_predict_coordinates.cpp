// SPDX-License-Identifier: BSD-3-Clause
// Inspect the existing model at measured coordinates without running search.
#include <tilemega/Solver/Sm90ForwardPlan.h>
#include <tilemega/Solver/Sm90ForwardSearch.h>
#include <cstdlib>
#include <fstream>
#include <iostream>

int main(int argc,char** argv)try {
  if(argc!=7)throw std::invalid_argument("root target export coordinates batch output");
  using namespace tilemega;
  using namespace tilemega::solver;
  auto target=TargetSpec::FromJson(argv[2]);
  auto native=Sm90NativeConstants::FromTarget(json::ParseFile(argv[2]),argv[1]);
  auto coordinates=Sm90ForwardCoordinates::FromJson(json::ParseFile(argv[4]));
  if(coordinates.abi!=Sm90ForwardAbi::kF3)
    throw std::invalid_argument("DNN4 coordinate inspection requires F3");
  auto plan=BuildSm90ForwardPlan(argv[3],std::stoi(argv[5]),false,Sm90ForwardAbi::kF3);
  auto shapes=Sm90ForwardShapesFromPlan(plan,Sm90ForwardAbi::kF3);
  Sm90ForwardCostModel cost(target,native,Sm90ForwardAbi::kF3);
  auto prediction=cost.Predict(shapes,coordinates);
  std::ofstream output(argv[6]);
  if(!output)throw std::runtime_error("cannot open prediction output");
  output<<prediction.ToJson().Dump(2)<<'\n';
  std::cout<<"stages="<<prediction.stages.size()<<" predicted_ns="<<prediction.total_ns<<'\n';
}catch(std::exception const& error){std::cerr<<error.what()<<'\n';return 1;}
