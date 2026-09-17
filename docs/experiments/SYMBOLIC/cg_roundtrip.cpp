// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Dialect/CouplingGraph/ParametricPlacementIO.h>
#include <mlir/Parser/Parser.h>
#include <llvm/Support/raw_ostream.h>
#include <fstream>
#include <filesystem>
#include <sstream>
#include <iostream>
#include <optional>
#include <array>
#include <set>
#include <memory>
#include <isl/map.h>
#include <isl/set.h>
#include <isl/point.h>
#include <isl/val.h>
using namespace tilemega;
static analysis::CouplingRelation read(std::filesystem::path p){std::ifstream f(p);if(!f)throw std::invalid_argument("missing "+p.string());std::stringstream s;s<<f.rdbuf();return analysis::CouplingRelation::FromIslText(s.str());}
using Map=std::unique_ptr<isl_map,decltype(&isl_map_free)>;
static auto points(isl_map* map,int seq,int grid) {
  auto bound=isl_map_copy(map);
  for(auto entry:std::vector<std::pair<char const*,int>>{{"S",seq},{"G",grid}}){
    int index=isl_map_find_dim_by_name(bound,isl_dim_param,entry.first);
    if(index<0){isl_map_free(bound);throw std::invalid_argument("missing parameter");}
    bound=isl_map_fix_si(bound,isl_dim_param,index,entry.second);
  }
  bound=isl_map_project_out(bound,isl_dim_param,0,isl_map_dim(bound,isl_dim_param));
  auto set=isl_map_wrap(bound);std::set<std::array<long,4>> result;
  auto point=[](isl_point* p,void* data)->isl_stat {
    std::array<long,4> v{};for(int i=0;i<4;++i){auto x=isl_point_get_coordinate_val(p,isl_dim_set,i);v[i]=isl_val_get_num_si(x);isl_val_free(x);}
    isl_point_free(p);static_cast<std::set<std::array<long,4>>*>(data)->insert(v);return isl_stat_ok;
  };
  struct Sink { decltype(point)* callback; decltype(result)* output; } sink{&point,&result};
  auto component=[](isl_basic_set* b,void* data)->isl_stat {
    auto& s=*static_cast<Sink*>(data);auto set=isl_set_from_basic_set(b);
    auto status=isl_set_foreach_point(set,*s.callback,s.output);isl_set_free(set);return status;
  };
  auto status=isl_set_foreach_basic_set(set,component,&sink);isl_set_free(set);
  if(status!=isl_stat_ok)throw std::runtime_error("symbolic point enumeration failed");return result;
}
int main(int argc,char** argv)try{
 if(argc!=4)throw std::invalid_argument("cg_roundtrip INPUT_CG PROOF_DIR OUT_CG");
 std::cout<<std::unitbuf;analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();
 auto m=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);if(!m)throw std::invalid_argument("invalid CG");
 auto resident=m->getOperation()->getAttrOfType<mlir::IntegerAttr>("tilemega.solved_grid");if(!resident)throw std::invalid_argument("missing residency witness");
 std::vector<mlir::Attribute> attrs;std::filesystem::path root=argv[2];
 for(std::string family:{"legacy_grid_stride","rotate","band","wavefront"}){
  std::optional<solver::ParametricPlacement> united;
  for(int grid:{256,340}){
   auto stem=family+"_g"+std::to_string(grid);
   solver::ParametricPlacement p{read(root/(stem+".tasks.isl")),read(root/(stem+".dependencies.isl")),read(root/(stem+".pi_sigma.isl")),read(root/(stem+".rank.isl")),analysis::CouplingRelation::FromIslText("[S,G] -> { [] -> [G,"+std::to_string(resident.getInt())+"] : 1<=S<=128 and G="+std::to_string(grid)+" }"),family};
   if(!united)united=p;else{united->tasks=united->tasks.Union(p.tasks);united->dependencies=united->dependencies.Union(p.dependencies);united->pi_sigma=united->pi_sigma.Union(p.pi_sigma);united->rank=united->rank.Union(p.rank);united->grid_limit=united->grid_limit.Union(p.grid_limit);}
  }
  attrs.push_back(dialect::ParametricPlacementAttr(&context,*united));
 }
 mlir::Builder b(&context);m->getOperation()->setAttr("tilemega.parametric_alternatives",b.getArrayAttr(attrs));
 std::string text;llvm::raw_string_ostream stream(text);m->print(stream);stream.flush();std::ofstream(argv[3])<<text<<'\n';
 auto back=mlir::parseSourceString<mlir::ModuleOp>(text,&context);if(!back)throw std::invalid_argument("CG did not round trip");
 auto copied=back->getOperation()->getAttrOfType<mlir::ArrayAttr>("tilemega.parametric_alternatives");
 for(std::size_t i=0;i<attrs.size();++i){
  auto a=dialect::ReadParametricPlacement(llvm::cast<mlir::DictionaryAttr>(attrs[i]));auto p=dialect::ReadParametricPlacement(llvm::cast<mlir::DictionaryAttr>(copied[i]));
  Map am(isl_map_read_from_str(isl.raw(),a.pi_sigma.ToString().c_str()),&isl_map_free),pm(isl_map_read_from_str(isl.raw(),p.pi_sigma.ToString().c_str()),&isl_map_free);
  if(!am || !pm)throw std::runtime_error("invalid round-trip map");
  for(int g:{256,340})for(int s:{1,32,64,96,128}){
   analysis::ParamBinding theta;theta.Bind("S",s).Bind("G",g);
   auto expected=points(am.get(),s,g),actual=points(pm.get(),s,g);
   if(expected!=actual)throw std::runtime_error("round trip changed symbolic evaluation");
   std::cout<<"CG_SYMBOLIC_ROUNDTRIP family="<<p.family<<" grid="<<g<<" seq="<<s<<" nodes="<<actual.size()<<" identical=1\n";
  }
 }
}catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
