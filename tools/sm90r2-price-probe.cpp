// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/Sm90DenseCostModel.h>
#include <tilemega/Solver/Sm90DenseSearchSpace.h>
#include <fstream>
#include <iostream>

int main(int argc,char** argv) {try {
  if(argc!=4)throw std::invalid_argument("usage: sm90r2_price_probe TARGET WORKLOAD COORDINATES");
  using namespace tilemega;using namespace tilemega::solver;
  auto target=TargetSpec::FromJson(argv[1]);auto cal=json::ParseFile(argv[1]).At("sm90_dense_decode");
  Sm90DenseCostModel model(target,cal);auto work=json::ParseFile(argv[2]);Sm90DenseWorkload w;
  auto integer=[&](char const* key) {return int(work.At(key).AsNumber(key));};
  w.batch=integer("batch");w.layers=integer("layers");w.head_dim=integer("head_dim");
  w.queries=integer("queries");w.qk_norm=work.At("qk_norm").AsBool("qk_norm");
  auto const& shapes=work.At("shapes").AsArray("shapes");
  if(shapes.size()!=5)throw std::invalid_argument("native workload requires five GEMM families");
  for(unsigned i=0;i<5;++i)for(unsigned j=0;j<2;++j)
    w.shapes[i][j]=int(shapes[i].AsArray("shape").at(j).AsNumber("extent"));
  Sm90DenseSearchSpace space{w.shapes,w.batch,w.head_dim,w.queries,target};
  auto values=json::ParseFile(argv[3]);json::Array rows;
  if(values.kind()!=json::Value::Kind::kArray)values=json::Array{values};
  for(auto const& value:values.AsArray("native coordinate cases")) {
    auto c=Sm90DenseCoordinates::FromJson(value);
    if(!space.Legal(c))throw std::invalid_argument("illegal native diagnostic coordinate");
    rows.emplace_back(json::Object{{"coordinates",c.ToJson()},
        {"request",model.Request(w,c).ToJson()},{"past575",model.Predict(w,c,575).ToJson()}});
  }
  std::cout<<json::Value(json::Object{{"diagnostic_only",true},
      {"fit_status",cal.At("fit_status")},{"predictions",rows}}).Dump(2)<<'\n';return 0;
} catch(std::exception const& e) {std::cerr<<e.what()<<'\n';return 2;}}
