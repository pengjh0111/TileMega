// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Solver/ModelDramFloor.h>
#include <tilemega/Solver/MoeDramBindings.h>
#include <mlir/IR/Builders.h>
#include <mlir/IR/BuiltinOps.h>
#include <filesystem>
#include <fstream>
#include <iostream>
namespace tilemega::solver {
analysis::DramFloor DeriveModelDramFloor(mlir::ModuleOp module,ModelDescription const& model,
    TargetSpec const& target,std::string const& fixture,
    MoeRoutingProfile const* profile,unsigned first_profile_layer) {
  auto theta=model.MetricBindings();auto plan=module->getAttrOfType<mlir::DictionaryAttr>("tilemega.model_plan");
  if(!plan)throw std::runtime_error("missing buffer identities");
  std::map<std::string,std::string> files,sources;
  for(auto v:plan.getAs<mlir::ArrayAttr>("buffers")) {
    auto d=llvm::cast<mlir::DictionaryAttr>(v);auto name=d.getAs<mlir::StringAttr>("name").getValue().str();
    files[name]=d.getAs<mlir::StringAttr>("file").getValue().str();sources[name]=d.getAs<mlir::StringAttr>("source").getValue().str();
  }
  analysis::DramFloorOptions options;auto cal=target.CalibrationFor(model.dtype==solver::ScalarType::kBF16?"bf16":"f32");
  options.dram_gbps=cal.dram_gbps;options.tc_gflops=model.dtype==solver::ScalarType::kBF16?cal.tc_bf16_gflops:cal.cuda_fp32_gflops;
  options.outputs=model.exported_tensors;
  if(model.dm) {options.element_bytes=model.buffer_element_bytes;options.infer_leaf_outputs=false;}
  analysis::SemanticGraph semantics;std::set<std::string> names;
  bool const serving = static_cast<bool>(module->getAttr("tilemega.serving"));
  auto token_axis=plan.getAs<mlir::BoolAttr>("forward_token_axis");
  bool const forward_tokens=model.dm && token_axis && token_axis.getValue();
  std::string batch_symbol;
  long serving_seq = 0;
  if (serving) {
    auto roles = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.dimension_roles");
    auto info = module->getAttrOfType<mlir::DictionaryAttr>("tilemega.serving");
    if (!roles || !info || (!forward_tokens && !roles.getAs<mlir::StringAttr>("batch")) ||
        !info.getAs<mlir::IntegerAttr>("seq"))
      throw std::runtime_error("serving floor requires batch and fixed seq");
    if(!forward_tokens)batch_symbol = roles.getAs<mlir::StringAttr>("batch").getValue().str();
    serving_seq = info.getAs<mlir::IntegerAttr>("seq").getInt();
  }
  for(auto const& task:model.task_semantics)if(names.insert(task.op.name).second)semantics.ops.push_back(task.op);
  if(model.dm)BindMoeDramInputs(module,model,options,profile,first_profile_layer);
  for(auto const& op:semantics.ops)for(auto const& operand:op.operands) {
    bool indirect=false;for(auto const& index:operand.map.results)indirect|=index.kind==analysis::IndexResult::Kind::kDataDependent;
    if(!indirect)continue;
    if(model.dm) {
      if(options.binding_internal.count(operand.tensor.name) ||
          options.indirect_read_images.count(operand.tensor.name) ||
          options.expected_indirect_reads.count(operand.tensor.name))continue;
      if((op.arithmetic!="embedding" && op.arithmetic!="embedding_sum") || operand.tensor.axes.size()!=2)
        throw std::runtime_error("unsupported DM indirect read: "+op.name+" / "+operand.tensor.name);
      auto width=operand.tensor.axes[1].extent.Eval(theta,{});
      if(width<=0 || operand.tensor.axes[0].extent.Eval(theta,{})<=0)
        throw std::invalid_argument("embedding table must have positive dimensions");
      // Nonempty valid ids touch at least one row. This is an explicit lower
      // bound, never a fabricated row-zero image or an empirical expectation.
      options.expected_indirect_reads[operand.tensor.name]={analysis::QuasiPolynomial::Constant(width),
          "inferred: nonempty validated embedding ids touch at least one row","lower_bound"};
      continue;
    }
    // The covered gather ABI carries an independent integer index tensor.
    // Its values, rather than the work-model row-zero placeholder, define
    // this audit's exact physical image (duplicates disappear under union).
    if(op.operands.size()!=2 || operand.tensor.axes.size()!=2)throw std::runtime_error("unsupported indirect read binding");
    auto const& ids=op.operands.front().tensor;int bits=64;
    if(auto b=plan.getAs<mlir::IntegerAttr>("token_id_bits"))bits=b.getInt();
    options.element_bytes[ids.name]=bits/8;
    if (serving) {
      // A data-dependent gather reads no more than one embedding row per
      // input token.  The abstract rows below deliberately make that an
      // upper bound without loading a fixture; tied lm_head reads the entire
      // table and makes the union exact for the anchored serving models.
      long width = operand.tensor.axes[1].extent.Eval(theta,{});
      std::string relation = (forward_tokens?std::string{}:"[" + batch_symbol + "] -> ")+
          "{ [] -> [row,col] : 0 <= row < " + std::to_string(serving_seq) +
          (forward_tokens?std::string{}:"*" + batch_symbol)+
          " and 0 <= col < " + std::to_string(width) + " }";
      options.indirect_read_images[operand.tensor.name] =
          analysis::CouplingRelation::FromIslText(relation);
      std::cout << "INDIRECT_UPPER_BOUND tensor=" << operand.tensor.name
                << " seq=" << serving_seq << " batch=" << batch_symbol << '\n';
      continue;
    }
    std::ifstream input(std::filesystem::path(fixture)/files.at(ids.name),std::ios::binary);
    if(!input)throw std::runtime_error("cannot bind indirect input "+ids.name);
    long n=1;for(auto const& axis:ids.axes)n*=axis.extent.Eval(theta,{});long width=operand.tensor.axes[1].extent.Eval(theta,{}),vocab=operand.tensor.axes[0].extent.Eval(theta,{});
    std::set<long> rows;for(long i=0;i<n;++i){std::int64_t x=0;if(bits==64)input.read(reinterpret_cast<char*>(&x),8);else{std::int32_t y=0;input.read(reinterpret_cast<char*>(&y),4);x=y;}
      if(!input || x<0 || x>=vocab)throw std::runtime_error("invalid indirect fixture index");rows.insert(x);}
    std::string rel="{ [] -> [row,col] : 0<=col<"+std::to_string(width)+" and (";bool first=true;
    for(long row:rows){if(!first)rel+=" or ";first=false;rel+="row="+std::to_string(row);}rel+=") }";
    options.indirect_read_images[operand.tensor.name]=analysis::CouplingRelation::FromIslText(rel);
    std::cout<<"INDIRECT tensor="<<operand.tensor.name<<" tokens="<<n<<" unique_rows="<<rows.size()<<" width="<<width<<'\n';
  }
  auto floor=model.dm?analysis::DeriveDramFloor(semantics,options,theta):
      analysis::DeriveDramFloor(semantics,options);auto value=floor.Evaluate(theta);
  mlir::Builder b(module.getContext());module->setAttr("tmexec.dram_floor",b.getDictionaryAttr({
    b.getNamedAttr("dram_ns",b.getStringAttr(floor.dram_ns.ToString())),b.getNamedAttr("compute_ns",b.getStringAttr(floor.compute_ns.ToString())),
    b.getNamedAttr("dram_value_ns",b.getF64FloatAttr(value.dram_ns)),b.getNamedAttr("compute_value_ns",b.getF64FloatAttr(value.compute_ns)),
    b.getNamedAttr("floor_value_ns",b.getF64FloatAttr(value.floor_ns)),b.getNamedAttr("indirect_input_bound",b.getBoolAttr(!options.indirect_read_images.empty()))}));
  if(model.dm) {
    std::vector<mlir::Attribute> cardinalities,bindings;
    for(auto const& [name,tensor]:floor.tensors) {
      if(tensor.expected_read_elements)cardinalities.push_back(b.getDictionaryAttr({
          b.getNamedAttr("tensor",b.getStringAttr(name)),
          b.getNamedAttr("kind",b.getStringAttr(tensor.cardinality_kind)),
          b.getNamedAttr("elements",b.getStringAttr(tensor.expected_read_elements->ToString())),
          b.getNamedAttr("source",b.getStringAttr(tensor.expectation_source))}));
      if(tensor.binding_producer)bindings.push_back(b.getDictionaryAttr({
          b.getNamedAttr("tensor",b.getStringAttr(name)),
          b.getNamedAttr("source",b.getStringAttr(tensor.binding_producer->source)),
          b.getNamedAttr("capacity_envelope",b.getStringAttr(tensor.binding_producer->envelope.ToString()))}));
    }
    mlir::NamedAttrList metadata(module->getAttrOfType<mlir::DictionaryAttr>("tmexec.dram_floor"));
    metadata.set("cardinalities",b.getArrayAttr(cardinalities));
    metadata.set("binding_provenance",b.getArrayAttr(bindings));
    module->setAttr("tmexec.dram_floor",metadata.getDictionary(module.getContext()));
  }
  return floor;
}
}
