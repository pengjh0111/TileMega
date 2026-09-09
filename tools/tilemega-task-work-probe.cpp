// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/TaskWork.h>
#include <tilemega/Analysis/TaskInstantiation.h>
#include <tilemega/Analysis/ISLContext.h>
#include <tilemega/Frontend/ExportBridge.h>
#include <tilemega/Frontend/SemanticLifting.h>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) try {
  using namespace tilemega::analysis;
  IslContext context;
  auto fixed=[](long n){return ClosedForm::Constant(n);};
  auto s=ClosedForm::Symbol("S");
  SemanticOp semantic;
  semantic.name="gemm";
  semantic.domain={{"m",s},{"n",fixed(512)},{"k",fixed(512)}};
  semantic.result_map.results={IndexResult::Dim("m"),IndexResult::Dim("n")};
  OperatorNode task;
  task.name="gemm"; task.kind=OperatorKind::kMatmul;
  task.output={"C",{{"m",s},{"n",fixed(512)}}};
  task.tile={fixed(128),fixed(128)};
  task.operands={
    {"",{"A",{{"m",s},{"k",fixed(512)}}},
       {OperandAxisMap::Indexed(0),OperandAxisMap::FullRange()}},
    {"",{"B",{{"n",fixed(512)},{"k",fixed(512)}}},
       {OperandAxisMap::Indexed(1),OperandAxisMap::FullRange()}}};
  SemanticOperand a,b;
  a.tensor=task.operands[0].tensor;
  a.map.results={IndexResult::Dim("m"),IndexResult::Dim("k")};
  b.tensor=task.operands[1].tensor;
  b.map.results={IndexResult::Dim("n"),IndexResult::Dim("k")};
  semantic.operands={a,b};
  auto work=DeriveTaskWork(semantic,task,{});
  std::cout << "seq\ttask_count\tphysical_bytes_per_iter\tnominal_bytes_per_iter\tlegacy_mainloop_bytes\tphysical_equal\n";
  int differences=0;
  for (int seq:{4,128,512}) {
    ParamBinding theta; theta.Bind("S",seq).Bind("m",0).Bind("n",0);
    long count=work.task_count.Eval(theta);
    if (count!=((seq+127)/128)*4 || work.reduce_extent.Eval(theta)!=512)
      throw std::runtime_error("task count or output-index reduction gate");
    double physical=work.read_elements.Eval(theta)*2.0/32;
    double nominal=work.nominal_read_elements.Eval(theta)*2.0/32;
    double legacy=2.0*16*(128+128);
    if (nominal!=legacy) throw std::runtime_error("nominal tile differs from legacy");
    differences+=physical!=legacy;
    std::cout<<seq<<'\t'<<count<<'\t'<<physical<<'\t'<<nominal<<'\t'<<legacy<<'\t'<<(physical==legacy)<<'\n';
  }
  // This diagnostic must expose the incompatible gate, not silently relabel
  // a nominal allocated tile as a physical global read.
  if (differences!=1) throw std::runtime_error("physical-tail counterexample absent");
  std::cerr<<"A3_DUAL_DOMAIN differences="<<differences
           <<" nominal_gate=PASS physical_counterexample=preserved production_gate=pending\n";
  task.operands.push_back(task.operands.front());
  semantic.operands.push_back(semantic.operands.front());
  auto repeated = DeriveTaskWork(semantic,task,{});
  ParamBinding tail; tail.Bind("S",4).Bind("m",0).Bind("n",0);
  if (repeated.read_elements.Eval(tail)!=work.read_elements.Eval(tail))
    throw std::runtime_error("repeated operand was double-counted");
  semantic.operands.pop_back();
  task.operands.pop_back();
  semantic.kind=OperatorKind::kMatmul;
  semantic.result=task.output;
  semantic.reduction={"k","add","gemm.partial","gemm.combine",true,{"m","n"}};
  SemanticGraph semantics; semantics.ops={semantic};
  int split_cells=0;
  for (int chunks:{1,2,4,8,16}) {
    Granularity g;
    g.Tile("gemm","m",fixed(128)).Tile("gemm","n",fixed(128));
    if (chunks>1) g.Split("gemm",fixed(512/chunks));
    auto graph=Instantiate(semantics,g);
    auto partial=DeriveTaskWork(semantic,*graph.Find("gemm"),{});
    for (int seq:{1,4,128,512,2048}) {
      ParamBinding theta; theta.Bind("S",seq).Bind("m",0).Bind("n",0).Bind("j",0);
      if (partial.reduce_extent.Eval(theta)!=512 ||
          partial.task_reduce_extent.Eval(theta)!=512/chunks ||
          partial.nominal_read_elements.Eval(theta)!=(128+128)*(512/chunks) ||
          partial.task_count.Eval(theta)!=((seq+127)/128)*4*chunks)
        throw std::runtime_error("split partial local work differs from access span");
      ++split_cells;
    }
  }
  std::cerr << "A3_LOCAL_REDUCTION split_seq_cells=" << split_cells << " status=PASS\n";
  auto nonaligned=semantics;
  nonaligned.ops[0].domain.back().extent=fixed(1536);
  for (auto& operand:nonaligned.ops[0].operands) operand.tensor.axes.back().extent=fixed(1536);
  Granularity split_g;
  split_g.Tile("gemm","m",fixed(128)).Tile("gemm","n",fixed(128)).Split("gemm",fixed(96));
  auto split_graph=Instantiate(nonaligned,split_g);
  auto unpadded=DeriveTaskWork(nonaligned.ops[0],*split_graph.Find("gemm"),{});
  TaskWorkOptions inner;
  inner.reduction_tiles.emplace("k",fixed(64));
  auto padded=DeriveTaskWork(nonaligned.ops[0],*split_graph.Find("gemm"),{},inner);
  ParamBinding point; point.Bind("S",4).Bind("m",0).Bind("n",0).Bind("j",0);
  if (unpadded.nominal_read_elements.Eval(point)!=256*96 ||
      padded.nominal_read_elements.Eval(point)!=256*128 ||
      padded.task_reduce_extent.Eval(point)!=96 ||
      padded.nominal_task_reduce_extent.Eval(point)!=128 ||
      padded.read_elements.Eval(point)!=unpadded.read_elements.Eval(point))
    throw std::runtime_error("inner padding changed physical reads or lost issued work");
  std::cerr << "A3_INNER_PADDING K=1536 split=16 tile_k=64 physical_chunk=96 nominal_chunk=128"
            << " physical_reads_unchanged=1 nominal_bytes_per_iter="
            << padded.nominal_read_elements.Eval(point)*2/2 << " status=PASS\n";
  int rejected=0;
  auto reject = [&](auto action) {
    auto before=context.ReferenceCount();
    try { action(); } catch (std::exception const&) { ++rejected; }
    if (context.ReferenceCount()!=before) throw std::runtime_error("error path leaked isl references");
  };
  auto bad=task;
  bad.operands.front().tensor.name.clear();
  reject([&]{ (void)DeriveTaskWork(semantic,bad,{}); });
  bad=task;
  bad.operands.push_back(bad.operands.front());
  bad.operands.back().tensor.layout_id="incompatible";
  reject([&]{ (void)DeriveTaskWork(semantic,bad,{}); });
  TaskWorkOptions invalid_inner;
  invalid_inner.reduction_tiles.emplace("m",fixed(64));
  reject([&]{ (void)DeriveTaskWork(semantic,task,{},invalid_inner); });
  invalid_inner.reduction_tiles.clear();
  invalid_inner.reduction_tiles.emplace("k",fixed(0));
  reject([&]{ (void)DeriveTaskWork(semantic,task,{},invalid_inner); });
  auto bad_semantic=semantic;
  bad_semantic.operands.front().map.results.back().terms.front().coefficient=fixed(2);
  reject([&]{ (void)DeriveTaskWork(bad_semantic,task,{}); });
  if (rejected!=5) throw std::runtime_error("TaskWork validation failed to reject invalid input");
  std::cerr<<"A3_READ_UNION repeated_operand=PASS error_branches="<<rejected<<"\n";
  for (int input=1; input<argc; ++input) {
    using namespace tilemega::frontend;
    auto bridge=ReadExportBridge(argv[input]);
    auto plan=BuildModelPlan(bridge.nodes,bridge.inputs,bridge.outputs);
    auto lifted=LiftSemantics(plan,{});
    auto granularity=LaunchGranularity(lifted,plan,{});
    auto graph=Instantiate(lifted.sem,granularity);
    int checked=0, norms=0;
    for (auto const& op:lifted.sem.ops) {
      if (op.arithmetic=="rmsnorm") {
        auto const* node=graph.Find(op.name);
        if (!node || op.operands.size()!=2)
          throw std::runtime_error("normalization scale read is missing");
        auto derived=DeriveTaskWork(op,*node,{});
        long width=op.result.axes.back().extent.Eval({},{});
        for (int seq:{1,4,128,512,2048}) {
          ParamBinding theta; theta.Bind("S",seq).Bind("past",3);
          for (auto const& coordinate:node->Coordinates()) theta.Bind(coordinate,0);
          if (derived.task_count.Eval(theta)!=seq || derived.read_elements.Eval(theta)!=2*width ||
              derived.write_elements.Eval(theta)!=width || derived.task_reduce_extent.Eval(theta)!=width)
            throw std::runtime_error("normalization access-derived work mismatch");
          ++norms;
        }
      }
      if (op.kind!=OperatorKind::kMatmul) continue;
      auto const* node=graph.Find(op.name);
      if (!node || op.operands.size()!=2)
        throw std::runtime_error("production GEMM read set is incomplete");
      auto derived=DeriveTaskWork(op,*node,{});
      ParamBinding fixed_dims;
      long n=node->output.axes[1].extent.Eval(fixed_dims,fixed_dims);
      long k=op.domain.back().extent.Eval(fixed_dims,fixed_dims);
      long tm=node->tile[0].Eval(fixed_dims,fixed_dims);
      long tn=node->tile[1].Eval(fixed_dims,fixed_dims);
      for (int seq:{1,4,128,512,2048}) {
        ParamBinding theta; theta.Bind("S",seq).Bind("past",3);
        for (auto const& coordinate:node->Coordinates()) theta.Bind(coordinate,0);
        long expected=((seq+tm-1)/tm)*((n+tn-1)/tn);
        if (derived.task_count.Eval(theta)!=expected ||
            derived.nominal_read_elements.Eval(theta)!=k*(tm+tn))
          throw std::runtime_error("production GEMM count/nominal gate: "+op.name);
        ++checked;
      }
    }
    if (!checked) throw std::runtime_error("export contains no checked GEMM");
    std::cerr<<"A3_PRODUCTION_EXPORT file="<<argv[input]<<" count_nominal_cells="<<checked
             <<" status=PASS cost_integration=pending\n";
    if (!norms) throw std::runtime_error("export contains no checked normalization");
    std::cerr << "A3_NORMALIZATION_EXPORT file=" << argv[input] << " cells=" << norms
              << " status=PASS scale_read=present\n";
  }
} catch(std::exception const& e) { std::cerr<<e.what()<<'\n'; return 2; }
