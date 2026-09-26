// SPDX-License-Identifier: BSD-3-Clause
// The serving plan has fixed S and a symbolic batch/past.  Packed QKV is
// represented as (token, KV group, column within group); this preserves the
// group rectangle in the CG instead of flattening it into an all-to-all edge.
#include <tilemega/Frontend/SemanticLifting.h>

#include <algorithm>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace tilemega::frontend {
namespace {
using namespace analysis;
constexpr auto missing = std::numeric_limits<std::uint32_t>::max();
ClosedForm C(long v) { return ClosedForm::Constant(v); }
TensorAxis Axis(std::string name, ClosedForm extent,
                ClosedForm origin = ClosedForm::Constant(0)) {
  TensorAxis a;
  a.name = std::move(name);
  a.extent = std::move(extent);
  a.origin = std::move(origin);
  return a;
}
TensorSpace Tensor(std::string name, std::vector<TensorAxis> axes) {
  TensorSpace result;
  result.name = std::move(name);
  result.axes = std::move(axes);
  return result;
}
IterationDim Parallel(std::string name, ClosedForm extent) {
  IterationDim d;
  d.name = std::move(name);
  d.extent = std::move(extent);
  return d;
}
IterationDim Reduce(std::string name, ClosedForm extent) {
  auto d = Parallel(std::move(name), std::move(extent));
  d.type = IteratorType::kReduction;
  return d;
}
IndexResult Id(std::string name) { return IndexResult::Dim(std::move(name)); }
IndexResult Aff(std::vector<IndexResult::Term> terms,
                ClosedForm offset = C(0)) {
  return IndexResult::Affine(std::move(terms), std::move(offset));
}
SemanticOperand Input(std::string producer, TensorSpace space,
                      std::vector<IndexResult> map) {
  SemanticOperand input;
  input.producer = std::move(producer);
  input.tensor = std::move(space);
  input.map.results = std::move(map);
  return input;
}
SemanticOp Output(std::string name, OperatorKind kind,
                  std::vector<IterationDim> domain, TensorSpace space,
                  std::vector<IndexResult> result_map,
                  std::vector<SemanticOperand> inputs) {
  SemanticOp op;
  op.name = std::move(name);
  op.kind = kind;
  op.domain = std::move(domain);
  op.result = std::move(space);
  op.result_map.results = std::move(result_map);
  op.result_effect.kind = EffectKind::kWrite;
  op.operands = std::move(inputs);
  return op;
}
}  // namespace

LiftedModel LiftServingSemantics(ModelPlan const& plan,
                                 LiftOptions const& options) {
  if (!plan.serving || !options.serving || options.batch_symbol.empty() ||
      plan.serving_seq != options.static_seq)
    throw std::invalid_argument("serving semantics require structural dimension roles");
  LiftedModel result;
  result.has_plan = true;
  result.written.assign(plan.buffers.size(), 0);
  const auto B = ClosedForm::Symbol(options.batch_symbol);
  const auto S = C(plan.serving_seq);
  const auto M = B * S;
  const auto past = options.past_symbol.empty()
      ? C(0) : ClosedForm::Symbol(options.past_symbol);
  const auto cap = C(plan.serving_capacity);
  std::unordered_map<std::uint32_t, std::string> writer;
  std::unordered_map<std::uint32_t, TensorSpace> written;
  auto name = [&](std::uint32_t id) -> std::string {
    if (id == missing || id >= plan.buffers.size())
      throw std::invalid_argument("serving stage has an unknown operand");
    return plan.buffers[id].name;
  };
  auto producer = [&](std::uint32_t id) -> std::string {
    auto found = writer.find(id);
    return found == writer.end() ? std::string() : found->second;
  };
  auto space = [&](std::uint32_t id, std::vector<TensorAxis> axes) {
    auto found = written.find(id);
    return found == written.end() ? Tensor(name(id), std::move(axes))
                                  : found->second;
  };
  auto record = [&](std::size_t i, SemanticOp op, OpRole role,
                    std::uint32_t output, std::string arithmetic) {
    op.dtype = ScalarType::kBF16;
    op.arithmetic = std::move(arithmetic);
    writer[output] = op.name;
    written[output] = op.result;
    result.written[output] = 1;
    result.ops.push_back({op.name, role, OwnershipKind::kTilePerBlock,
                          static_cast<int>(i), 0, plan.stages[i].representative});
    result.sem.ops.push_back(std::move(op));
  };
  for (std::size_t i = 0; i < plan.stages.size(); ++i) {
    auto const& stage = plan.stages[i];
    const auto op_name = "serving.s" + std::to_string(i);
    if (stage.kind == PlanTaskKind::kEmbedding) {
      auto token = Tensor(name(stage.operands[0]),
                          {Axis("b", B), Axis("pos", cap)});
      // m mod S = m - S*floor(m/S); the batch index is floor(m/S).
      auto token_index = Aff({{"m", C(1), C(1)},
                              {"m", C(-plan.serving_seq), S}}, past);
      auto id_read = Input(producer(stage.operands[0]), token,
                           {Aff({{"m", C(1), S}}), token_index});
      auto table = space(stage.operands[1],
                         {Axis("v", C(stage.extent)), Axis("h", C(stage.width))});
      auto table_read = Input(producer(stage.operands[1]), table,
                              {IndexResult::DataDependent(), Id("h")});
      auto op = Output(op_name, OperatorKind::kPointwise,
          {Parallel("m", M), Parallel("h", C(stage.width))},
          Tensor(name(stage.operands[2]),
                 {Axis("m", M), Axis("h", C(stage.width))}),
          {Id("m"), Id("h")}, {id_read, table_read});
      // The actual table row is data dependent.  One row per token is an
      // exact cardinality upper bound; the tied lm_head reads the whole table.
      op.element_reads = {{token, id_read.map, {}},
                          {table, {{Aff({}), Id("h")}}, {}}};
      record(i, std::move(op), OpRole::kEmbedding, stage.operands[2], "embedding");
      continue;
    }
    if (stage.kind == PlanTaskKind::kRMSNorm) {
      auto rows = stage.batch_rows ? B : M;
      const auto H = C(stage.width);
      auto row = stage.batch_rows
          ? Aff({{"m", C(stage.row_stride), C(1)}}, C(stage.row_offset))
          : Id("m");
      auto x = Input(producer(stage.operands[0]),
                     space(stage.operands[0], {Axis("m", M), Axis("h", H)}),
                     {row, Id("r")});
      auto w = Input(producer(stage.operands[1]),
                     space(stage.operands[1], {Axis("h", H)}), {Id("h")});
      auto op = Output(op_name, OperatorKind::kReduction,
          {Parallel("m", rows), Parallel("h", H), Reduce("r", H)},
          Tensor(name(stage.operands[2]), {Axis("m", rows), Axis("h", H)}),
          {Id("m"), Id("h")}, {x, w});
      record(i, std::move(op), OpRole::kNorm, stage.operands[2], "rmsnorm");
      continue;
    }
    if (stage.kind == PlanTaskKind::kGemm) {
      auto const& gemm = plan.gemms.at(stage.gemm);
      const auto rows = stage.batch_rows ? B : M;
      const auto N = C(gemm.n), K = C(gemm.k);
      std::vector<IterationDim> domain;
      TensorSpace dst;
      std::vector<IndexResult> dst_map;
      OpRole role = OpRole::kProjection;
      std::string arithmetic = "gemm";
      std::vector<SemanticOperand> reads;
      if (gemm.epilogue == PlanGemm::Epilogue::kSwiGLU) {
        role = OpRole::kServingSwiGlu;
        arithmetic = "swiglu_gemm";
        domain = {Parallel("m", rows), Parallel("i", C(gemm.n / 2)), Reduce("k", K)};
        dst = Tensor(name(gemm.d), {Axis("m", rows), Axis("i", C(gemm.n / 2))});
        dst_map = {Id("m"), Id("i")};
        auto input = space(gemm.a, {Axis("m", rows), Axis("k", K)});
        reads.push_back(Input(producer(gemm.a), input, {Id("m"), Id("k")}));
        auto weight = Tensor(name(gemm.b), {Axis("n", N), Axis("k", K)});
        auto gate = Aff({{"i", C(1), C(1)},
                         {"i", C(gemm.interleave_u), C(gemm.interleave_u)} });
        // gate row = i + u*floor(i/u), up row = gate + u.
        reads.push_back(Input("", weight, {gate, Id("k")}));
        reads.push_back(Input("", weight,
                              {Aff(gate.terms, C(gemm.interleave_u)), Id("k")}));
      } else if (gemm.epilogue == PlanGemm::Epilogue::kArgmaxPartial) {
        role = OpRole::kServingArgmaxPartial;
        arithmetic = "argmax_gemm";
        // A plan's partial reduction tile is fixed before its maps are built.
        if (!gemm.partial_tile_n || gemm.partial_tile_n % 32)
          throw std::invalid_argument("argmax partial requires a bound N tile");
        const auto tile_n = C(gemm.partial_tile_n);
        const auto Nt = N.CeilDiv(tile_n);
        domain = {Parallel("m", rows), Parallel("tile", Nt),
                  Reduce("n", tile_n), Reduce("k", K)};
        dst = Tensor(name(gemm.d), {Axis("m", rows), Axis("tile", Nt)});
        dst_map = {Id("m"), Id("tile")};
        reads.push_back(Input(producer(gemm.a),
            space(gemm.a, {Axis("m", rows), Axis("k", K)}), {Id("m"), Id("k")}));
        reads.push_back(Input("", Tensor(name(gemm.b),
            {Axis("n", N), Axis("k", K)}),
            {Aff({{"tile", tile_n, C(1)}, {"n", C(1), C(1)}}), Id("k")}));
      } else if (plan.buffers.at(gemm.b).pack_json.find(
                     "qkv_group_interleave") != std::string::npos) {
        role = OpRole::kServingQkv;
        // The packed rows are grouped, and a tile cannot cross a KV group.
        std::uint32_t groups = 0;
        for (std::size_t next = i + 1; next < plan.stages.size(); ++next) {
          if (plan.stages[next].kind == PlanTaskKind::kFusedAttention &&
              plan.stages[next].operands[0] == gemm.d) {
            groups = plan.stages[next].extent;
            break;
          }
        }
        if (!groups || gemm.n % groups)
          throw std::invalid_argument("packed QKV has no matching grouped attention");
        const auto G = C(groups), U = C(gemm.n / groups);
        domain = {Parallel("m", rows), Parallel("g", G),
                  Parallel("u", U), Reduce("k", K)};
        dst = Tensor(name(gemm.d),
                     {Axis("m", rows), Axis("g", G), Axis("u", U)});
        dst_map = {Id("m"), Id("g"), Id("u")};
        reads.push_back(Input(producer(gemm.a),
            space(gemm.a, {Axis("m", rows), Axis("k", K)}), {Id("m"), Id("k")}));
        reads.push_back(Input("", Tensor(name(gemm.b),
            {Axis("g", G), Axis("u", U), Axis("k", K)}),
            {Id("g"), Id("u"), Id("k")}));
      } else {
        domain = {Parallel("m", rows), Parallel("n", N), Reduce("k", K)};
        dst = Tensor(name(gemm.d), {Axis("m", rows), Axis("n", N)});
        dst_map = {Id("m"), Id("n")};
        auto input = space(gemm.a, {Axis("m", rows), Axis("k", K)});
        std::vector<IndexResult> input_map;
        if (input.axes.size() == 4)
          input_map = {Id("m"), IndexResult::FullRange(),
                       IndexResult::FullRange(), IndexResult::FullRange()};
        else input_map = {Id("m"), Id("k")};
        reads.push_back(Input(producer(gemm.a), input, input_map));
        reads.push_back(Input("", Tensor(name(gemm.b),
            {Axis("n", N), Axis("k", K)}), {Id("n"), Id("k")}));
        if (gemm.epilogue == PlanGemm::Epilogue::kResidual)
          reads.push_back(Input(producer(gemm.c),
              space(gemm.c, {Axis("m", rows), Axis("n", N)}),
              {Id("m"), Id("n")}));
      }
      auto op = Output(op_name, OperatorKind::kMatmul, std::move(domain),
                       std::move(dst), std::move(dst_map), std::move(reads));
      // The packed QKV/SwiGLU and vocabulary-partial maps couple output
      // tiles to specific weight rows. Rectangular operand projection would
      // forget those affine expressions and charge every task the full
      // weight tensor. Preserve the physical element reads for pricing and
      // the DRAM floor.
      bool affine_reads=true;
      for(auto const& operand:op.operands)
        for(auto const& index:operand.map.results)
          affine_reads &= index.kind==IndexResult::Kind::kAffine;
      if(affine_reads)
        for(auto const& operand:op.operands)
          op.element_reads.push_back({operand.tensor,operand.map,{}});
      if (role != OpRole::kServingArgmaxPartial) {
        op.reduction = {"k", "add", op.name + ".partial",
                        op.name + ".combine", true, {}};
      }
      record(i, std::move(op), role, gemm.d, arithmetic);
      continue;
    }
    if (stage.kind == PlanTaskKind::kFusedAttention) {
      const auto G = C(stage.extent), D = C(stage.width), Q = C(stage.group);
      const auto Ec = C(stage.attention_kv_block);
      const auto Wg = C((stage.group + 2) * stage.width);
      const auto blocks = C((plan.serving_capacity + stage.attention_kv_block - 1) /
                            stage.attention_kv_block);
      const bool decode = plan.serving_seq == 1;
      const bool split_kv = decode &&
          plan.serving_capacity > stage.attention_kv_block;
      std::vector<IterationDim> domain;
      TensorSpace output;
      std::vector<IndexResult> out_map;
      IndexResult bidx, gidx, qidx, didx, posidx, qkv_row;
      if (decode) {
        domain = {Parallel("b", B), Parallel("g", G)};
        if (split_kv) domain.push_back(Parallel("c", blocks));
        domain.push_back(Parallel("q", Q));
        domain.push_back(Parallel("d", D));
        domain.push_back(Reduce("z", Ec));
        domain.push_back(Reduce("u", Wg));
        // A single KV block writes context directly; only split-KV writes
        // partial/LSE.  Naming the wrong result drops the attention -> o_proj
        // edge and lets L2 consume unwritten context.
        if (!split_kv) {
          output = Tensor(name(stage.operands[7]),
              {Axis("m", B), Axis("g", G), Axis("q", Q), Axis("d", D)});
          out_map = {Id("b"), Id("g"), Id("q"), Id("d")};
        } else {
          output = Tensor(name(stage.operands[8]),
              {Axis("b", B), Axis("g", G), Axis("c", blocks),
               Axis("q", Q), Axis("d", D)});
          out_map = {Id("b"), Id("g"), Id("c"), Id("q"), Id("d")};
        }
        bidx = Id("b"); gidx = Id("g"); qidx = Id("q"); didx = Id("d");
        qkv_row = Aff({{"b", S, C(1)}, {"q", C(1), Q}});
        posidx = split_kv
            ? Aff({{"c", Ec, C(1)}, {"z", C(1), C(1)}})
            : Id("z");
      } else {
        domain = {Parallel("m", M), Parallel("g", G), Parallel("q", Q),
                  Parallel("d", D), Reduce("z", S), Reduce("u", Wg)};
        output = Tensor(name(stage.operands[7]),
            {Axis("m", M), Axis("g", G), Axis("q", Q), Axis("d", D)});
        out_map = {Id("m"), Id("g"), Id("q"), Id("d")};
        bidx = Aff({{"m", C(1), S}});
        gidx = Id("g"); qidx = Id("q"); didx = Id("d");
        qkv_row = Id("m");
        posidx = Id("z");
      }
      auto qkv = space(stage.operands[0],
          {Axis("m", M), Axis("g", G), Axis("u", Wg)});
      auto cache = [&](std::uint32_t id) {
        return Tensor(name(id), {Axis("b", B), Axis("g", G),
                                 Axis("pos", cap), Axis("d", D)});
      };
      auto cos = Tensor(name(stage.operands[3]),
                         {Axis("pos", cap), Axis("d", D)});
      auto sin = Tensor(name(stage.operands[4]),
                         {Axis("pos", cap), Axis("d", D)});
      auto token_pos = decode ? Aff({}, past)
          : Aff({{"m", C(1), C(1)},
                 {"m", C(-plan.serving_seq), S}}, past);
      auto rope_pos = token_pos;
      auto kv_bound = decode
          ? (split_kv
              ? Aff({{"c", C(-stage.attention_kv_block), C(1)},
                     {"z", C(-1), C(1)}}, past + S + C(-1))
              : Aff({{"z", C(-1), C(1)}}, past + S + C(-1)))
          : Aff({{"m", C(1), C(1)},
                 {"m", C(-plan.serving_seq), S},
                 {"z", C(-1), C(1)}}, past);
      auto qkv_read = Input(producer(stage.operands[0]), qkv,
                             {qkv_row, gidx, Id("u")});
      auto k_read = Input("", cache(stage.operands[1]),
                           {bidx, gidx, posidx, didx});
      auto v_read = Input("", cache(stage.operands[2]),
                           {bidx, gidx, posidx, didx});
      k_read.effect.state_object = "kv_cache";
      v_read.effect.state_object = "kv_cache";
      auto cos_read = Input("", cos, {rope_pos, didx});
      auto sin_read = Input("", sin, {rope_pos, didx});
      std::vector<SemanticOperand> operands =
          {qkv_read, k_read, v_read, cos_read, sin_read};
      if (stage.operands[5] != missing)
        operands.push_back(Input("", Tensor(name(stage.operands[5]),
                                {Axis("d", D)}), {didx}));
      if (stage.operands[6] != missing)
        operands.push_back(Input("", Tensor(name(stage.operands[6]),
                                {Axis("d", D)}), {didx}));
      auto op = Output(op_name, OperatorKind::kReduction, std::move(domain),
                       output, out_map, operands);
      op.element_reads.push_back({qkv, qkv_read.map, {}});
      op.element_reads.push_back({k_read.tensor, k_read.map, {kv_bound}});
      op.element_reads.push_back({v_read.tensor, v_read.map, {kv_bound}});
      op.element_reads.push_back({cos, cos_read.map, {}});
      op.element_reads.push_back({sin, sin_read.map, {}});
      for (std::size_t extra = 5; extra < operands.size(); ++extra)
        op.element_reads.push_back({operands[extra].tensor,
                                    operands[extra].map, {}});
      // A single fused task writes the next KV row as a state side effect.
      // On decode only the block containing `past` is its writer.
      auto new_pos = token_pos;
      std::vector<IndexResult> writer_predicates;
      if (split_kv) {
        writer_predicates.push_back(Aff({{"c", C(-stage.attention_kv_block), C(1)}}, past));
        writer_predicates.push_back(Aff({{"c", Ec, C(1)}},
                                         Ec + C(-1) + past * C(-1)));
      }
      for (auto id : {stage.operands[1], stage.operands[2]}) {
        ElementWrite write;
        write.tensor = cache(id);
        write.map.results = {bidx, gidx, new_pos, didx};
        write.nonnegative = writer_predicates;
        write.effect.kind = EffectKind::kReadWrite;
        write.effect.state_object = "kv_cache";
        op.additional_writes.push_back(std::move(write));
        result.written[id] = 1;
      }
      if (split_kv) {
        ElementWrite lse;
        lse.tensor = Tensor(name(stage.operands[9]),
            {Axis("b", B), Axis("g", G), Axis("c", blocks), Axis("q", Q)});
        lse.map.results = {Id("b"), Id("g"), Id("c"), Id("q")};
        op.additional_writes.push_back(std::move(lse));
        writer[stage.operands[9]] = op.name;
        written[stage.operands[9]] = op.additional_writes.back().tensor;
        result.written[stage.operands[9]] = 1;
      }
      record(i, std::move(op), OpRole::kServingFusedAttention,
             split_kv ? stage.operands[8] : stage.operands[7], "fused_attention");
      continue;
    }
    if (stage.kind == PlanTaskKind::kAttentionMerge) {
      const auto G = C(stage.extent), Q = C(stage.group), D = C(stage.width);
      const auto blocks = C((plan.serving_capacity + stage.attention_kv_block - 1) /
                            stage.attention_kv_block);
      auto po = space(stage.operands[0],
          {Axis("b", B), Axis("g", G), Axis("c", blocks),
           Axis("q", Q), Axis("d", D)});
      auto lse = space(stage.operands[1],
          {Axis("b", B), Axis("g", G), Axis("c", blocks), Axis("q", Q)});
      auto op = Output(op_name, OperatorKind::kReduction,
          {Parallel("m", B), Parallel("g", G), Parallel("q", Q),
           Parallel("d", D), Reduce("c", blocks)},
          Tensor(name(stage.operands[2]),
              {Axis("m", B), Axis("g", G), Axis("q", Q), Axis("d", D)}),
          {Id("m"), Id("g"), Id("q"), Id("d")},
          {Input(producer(stage.operands[0]), po,
                 {Id("m"), Id("g"), Id("c"), Id("q"), Id("d")}),
           // LSE is a side write of the same fused task.  The PO edge
           // already carries the full dependency; naming the side write as
           // a second producer would give W/R different ranks.
           Input("", lse,
                 {Id("m"), Id("g"), Id("c"), Id("q")})});
      record(i, std::move(op), OpRole::kServingMerge,
             stage.operands[2], "attention_merge");
      continue;
    }
    if (stage.kind == PlanTaskKind::kArgmaxReduce) {
      auto partial = space(stage.operands[0],
          {Axis("m", B), Axis("tile", C((stage.extent + 31) / 32))});
      auto op = Output(op_name, OperatorKind::kReduction,
          {Parallel("m", B), Reduce("tile", partial.axes[1].extent)},
          Tensor(name(stage.operands[2]),
              {Axis("b", B), Axis("pos", C(1), past + S)}),
          {Id("m"), Aff({})},
          {Input(producer(stage.operands[0]), partial,
                 {Id("m"), Id("tile")}),
           Input(producer(stage.operands[1]),
                 Tensor(name(stage.operands[1]), partial.axes),
                 {Id("m"), Id("tile")})});
      op.result_effect.state_object = "serving.tokens";
      record(i, std::move(op), OpRole::kServingArgmax,
             stage.operands[2], "argmax_reduce");
      continue;
    }
    // Fused attention and LSE merge require a physical KV element map.  They
    // are lifted below; no serving stage is silently dropped.
    throw std::invalid_argument("serving semantic lifting has no rule for stage " +
                                std::to_string(i));
  }
  return result;
}
}  // namespace tilemega::frontend
