// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/DramFloor.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Solver/ModelDescription.h>
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Target/TargetSpec.h>
#include <mlir/Parser/Parser.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <llvm/Support/raw_ostream.h>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iomanip>
using namespace tilemega;
int main(int argc,char** argv) {
 try {
  if(argc!=8)throw std::invalid_argument("usage: tilemega-dram-floor CG target seq past fixture output.mlir tensors.tsv");
  analysis::IslContext isl;mlir::MLIRContext context;context.getOrLoadDialect<dialect::CGDialect>();context.getOrLoadDialect<dialect::ExecDialect>();
  auto module=mlir::parseSourceFile<mlir::ModuleOp>(argv[1],&context);if(!module)throw std::runtime_error("cannot parse CG");
  int seq=std::stoi(argv[3]),past=std::stoi(argv[4]);auto target=TargetSpec::FromJson(argv[2]);
  auto model=solver::ModelDescription::FromCouplingGraph(*module,{seq,past,seq+past},"floor");
  auto theta=model.MetricBindings();auto plan=(*module)->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if(!plan)throw std::runtime_error("missing buffer identities");
  std::map<std::string,std::string> files,sources;
  for(auto v:plan.getAs<mlir::ArrayAttr>("buffers")) {
    auto d=llvm::cast<mlir::DictionaryAttr>(v);auto name=d.getAs<mlir::StringAttr>("name").getValue().str();
    files[name]=d.getAs<mlir::StringAttr>("file").getValue().str();sources[name]=d.getAs<mlir::StringAttr>("source").getValue().str();
  }
  analysis::DramFloorOptions options;auto cal=target.CalibrationFor(model.dtype==solver::ScalarType::kBF16?"bf16":"f32");
  options.dram_gbps=cal.dram_gbps;options.tc_gflops=model.dtype==solver::ScalarType::kBF16?cal.tc_bf16_gflops:cal.cuda_fp32_gflops;
  options.outputs=model.exported_tensors;
  analysis::SemanticGraph semantics;std::set<std::string> names;
  for(auto const& task:model.task_semantics)if(names.insert(task.op.name).second)semantics.ops.push_back(task.op);
  for(auto const& op:semantics.ops)for(auto const& operand:op.operands) {
    bool indirect=false;for(auto const& index:operand.map.results)indirect|=index.kind==analysis::IndexResult::Kind::kDataDependent;
    if(!indirect)continue;
    // The covered gather ABI carries an independent integer index tensor.
    // Its values, rather than the work-model row-zero placeholder, define
    // this audit's exact physical image (duplicates disappear under union).
    if(op.operands.size()!=2 || operand.tensor.axes.size()!=2)throw std::runtime_error("unsupported indirect read binding");
    auto const& ids=op.operands.front().tensor;int bits=64;
    if(auto b=plan.getAs<mlir::IntegerAttr>("token_id_bits"))bits=b.getInt();
    options.element_bytes[ids.name]=bits/8;
    std::ifstream input(std::filesystem::path(argv[5])/files.at(ids.name),std::ios::binary);
    if(!input)throw std::runtime_error("cannot bind indirect input "+ids.name);
    long n=ids.axes.front().extent.Eval(theta,{}),width=operand.tensor.axes[1].extent.Eval(theta,{}),vocab=operand.tensor.axes[0].extent.Eval(theta,{});
    std::set<long> rows;for(long i=0;i<n;++i){std::int64_t x=0;if(bits==64)input.read(reinterpret_cast<char*>(&x),8);else{std::int32_t y=0;input.read(reinterpret_cast<char*>(&y),4);x=y;}
      if(!input || x<0 || x>=vocab)throw std::runtime_error("invalid indirect fixture index");rows.insert(x);}
    std::string rel="{ [] -> [row,col] : 0<=col<"+std::to_string(width)+" and (";bool first=true;
    for(long row:rows){if(!first)rel+=" or ";first=false;rel+="row="+std::to_string(row);}rel+=") }";
    options.indirect_read_images[operand.tensor.name]=analysis::CouplingRelation::FromIslText(rel);
    std::cout<<"INDIRECT tensor="<<operand.tensor.name<<" tokens="<<n<<" unique_rows="<<rows.size()<<" width="<<width<<'\n';
  }
  auto floor=analysis::DeriveDramFloor(semantics,options);auto value=floor.Evaluate(theta);
  mlir::Builder b(&context);(*module)->setAttr("tmexec.dram_floor",b.getDictionaryAttr({
    b.getNamedAttr("dram_ns",b.getStringAttr(floor.dram_ns.ToString())),b.getNamedAttr("compute_ns",b.getStringAttr(floor.compute_ns.ToString())),
    b.getNamedAttr("dram_value_ns",b.getF64FloatAttr(value.dram_ns)),b.getNamedAttr("compute_value_ns",b.getF64FloatAttr(value.compute_ns)),
    b.getNamedAttr("floor_value_ns",b.getF64FloatAttr(value.floor_ns)),b.getNamedAttr("indirect_input_bound",b.getBoolAttr(!options.indirect_read_images.empty()))}));
  std::error_code error;llvm::raw_fd_ostream out(argv[6],error);if(error)throw std::runtime_error(error.message());module->print(out);
  std::ofstream detail(argv[7]);detail<<"tensor\tsource\tread_bytes\twrite_bytes\tread_polynomial\twrite_polynomial\n";
  double weights=0;for(auto const& [name,t]:floor.tensors){auto reads=t.read_bytes.Eval(theta),writes=t.write_bytes.Eval(theta);if(sources[name]=="weight")weights+=reads;
    detail<<name<<'\t'<<sources[name]<<'\t'<<reads<<'\t'<<writes<<'\t'<<t.read_bytes.ToString()<<'\t'<<t.write_bytes.ToString()<<'\n';}
  std::cout<<std::setprecision(17)<<"DRAM_FLOOR weight_bytes="<<weights<<" read_bytes="<<value.read_bytes<<" write_bytes="<<value.write_bytes<<" flops="<<value.flops<<" dram_ns="<<value.dram_ns<<" compute_ns="<<value.compute_ns<<" floor_ns="<<value.floor_ns<<'\n';
 }catch(std::exception const& e){std::cerr<<e.what()<<'\n';return 1;}
}
