// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Analysis/ISLContext.h>
// Independent CG-input versus archived generated-input equivalence gate.
#include <tilemega/Dialect/CouplingGraph/CGDialect.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <tilemega/Solver/CostModel.h>
#include <tilemega/Solver/ChainDP.h>
#include <mlir/IR/MLIRContext.h>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <regex>
#include <sstream>
#include <stdexcept>

using namespace tilemega::solver;
std::vector<std::string> Fields(std::string const& line) {
  std::istringstream input(line); std::string field;
  std::vector<std::string> result;
  while (std::getline(input, field, '\t')) result.push_back(field);
  return result;
}
bool Bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }
int main(int argc, char** argv) try {
  tilemega::analysis::IslContext isl_context;
  if (argc < 2 || argc > 3) throw std::runtime_error("usage: tilemega-parametric REPO [f32|bf16]");
  std::string root = argv[1];
  std::string dtype = argc == 3 ? argv[2] : "f32";
  if (dtype != "f32" && dtype != "bf16") throw std::invalid_argument("unknown gate dtype");
  bool const bf16 = dtype == "bf16";
  auto scalar = bf16 ? ScalarType::kBF16 : ScalarType::kF32;
  std::string const oracle = root + "/docs/experiments/ORACLE/" + (bf16 ? "raw_bf16/" : "raw/");
  auto target = tilemega::TargetSpec::FromJson(root + "/configs/targets/sm_89.json");
  CostModelOptions legacy_options;
  legacy_options.fp32_partials = bf16;
  CostModel cost(target, scalar, legacy_options);
  CostModelOptions partial_options;
  partial_options.fp32_partials = true;
  CostModel partial_cost(target, scalar, partial_options);
  mlir::MLIRContext context;
  context.getOrLoadDialect<tilemega::dialect::CGDialect>();
  int total = 0;
  std::ofstream dp_output(root + "/docs/experiments/PARAMETRIC/finite_dp" +
                          (bf16 ? "_bf16.tsv" : ".tsv"));
  if (!dp_output) throw std::runtime_error("cannot write finite DP report");
  dp_output << "model\tbegin\tend\tctas_per_sm\tgemm\ttile_m\ttile_n\ttile_k\tstages\tsplit\n";
  std::cout << "model\tconfig\tinput_equal\tlegacy_total\tbound_cg_total\n" << std::setprecision(17);
  for (std::string const name : {"gqa2", "mha4"}) {
    std::string const base = root + "/docs/experiments/" +
        (name == "gqa2" ? "E2E_GEN/raw/" : "P3_GENERALIZATION/raw/");
    auto cg = tilemega::frontend::TorchExportImporter{}.Import(
        bf16 ? oracle + "export/" + name + ".json" : base + "export_bridge.json", context);
    auto symbolic = ModelDescription::FromCouplingGraph(*cg, ModelDims::Symbolic("S", 3), name);
    tilemega::analysis::ParamBinding theta;
    theta.Bind("S", 4);
    auto concrete = symbolic.SubstituteParams(theta);
    auto legacy = ModelDescription::FromGeneratedCuda(
        bf16 ? oracle + "src/" + name + "_128x128x16s3k1.cu" :
        base + (name == "gqa2" ? "generated_e2e.cu" : "generated.cu"), {4, 3, 7}, name);
    if (legacy.dtype != scalar || concrete.dtype != scalar)
      throw std::runtime_error("gate input dtype does not match requested dtype");
    std::ifstream input(oracle + "screen_" + name + ".tsv");
    if (!input) throw std::runtime_error("missing ORACLE input");
    std::string line; std::getline(input, line);
    std::map<std::string, int> registers;
    std::ifstream register_input((bf16 ? oracle + "cost/" :
        root + "/docs/experiments/COST_MODEL/raw/") + "registers_" + name + ".tsv");
    if (!register_input) throw std::runtime_error("missing measured register table");
    while (std::getline(register_input, line)) {
      if (line.empty() || line.front() == '#') continue;
      auto row = Fields(line);
      registers.emplace(row.at(0), std::stoi(row.at(1)));
    }
    std::vector<DpCandidate> candidates;
    int count = 0, excluded = 0, recovered = 0;
    while (std::getline(input, line)) {
      auto row = Fields(line);
      if (row.size() < 13) throw std::runtime_error("malformed ORACLE row");
      if (row[8] != "PASS" && !bf16) { ++excluded; continue; }
      GemmConfig config{std::stoi(row[0]), std::stoi(row[1]), std::stoi(row[2]),
                        std::stoi(row[3]), std::stoi(row[4])};
      auto shape = row[0] + "x" + row[1] + "x" + row[2] + "s" + row[3];
      int smem = 0, ctas = 0;
      if (row[8] == "PASS") {
        smem = std::stoi(row[10]); ctas = std::stoi(row[11]);
      } else {
        std::ifstream replay(root + "/docs/experiments/BF16/runfail_audit/logs/" +
                             name + "_" + shape + "k" + row[4] + ".txt");
        if (!replay) throw std::runtime_error("missing failure classification evidence");
        std::string text((std::istreambuf_iterator<char>(replay)), {});
        std::smatch resource, mismatch;
        if (row[8] != "RUNFAIL" || text.find("RESULT status=MISMATCH") == std::string::npos ||
            !std::regex_search(text, resource, std::regex("E2E_RESOURCE [^\\n]*smem=([0-9]+)[^\\n]*ctas_per_sm=([0-9]+)")) ||
            !std::regex_search(text, mismatch, std::regex("l05_vs_l0_mismatch=([0-9]+)")) ||
            std::stoi(mismatch[1]) == 0)
          throw std::runtime_error("nonpass is not a classified numerical failure");
        smem = std::stoi(resource[1]); ctas = std::stoi(resource[2]); ++recovered;
      }
      candidates.push_back({config, registers.at(shape), bf16 ? smem :
          4*config.stages*config.tile_k*(config.tile_m+config.tile_n+2)});
      Residency const residency{ctas};
      auto a = cost.Evaluate(legacy, config, residency);
      auto b = cost.Evaluate(concrete, config, residency);
      auto p = partial_cost.Evaluate(legacy, config, residency);
      if (!bf16 && !(Bits(a.total_ns,p.total_ns) && Bits(a.gemm_ns,p.gemm_ns) &&
            Bits(a.combine_ns,p.combine_ns) && Bits(a.other_ns,p.other_ns) &&
            Bits(a.barrier_ns,p.barrier_ns) && a.stage_count == p.stage_count))
        throw std::runtime_error("FP32 partial-storage cost regression; stop");
      bool equal = Bits(a.total_ns,b.total_ns) && Bits(a.gemm_ns,b.gemm_ns) &&
          Bits(a.combine_ns,b.combine_ns) && Bits(a.other_ns,b.other_ns) &&
          Bits(a.barrier_ns,b.barrier_ns) && a.stage_count == b.stage_count;
      std::cout << name << '\t' << row[0] << 'x' << row[1] << 'x' << row[2]
                << 's' << row[3] << 'k' << row[4] << '\t' << equal << '\t'
                << a.total_ns << '\t' << b.total_ns << '\n';
      if (!equal) throw std::runtime_error("bitwise CG input gate failed; stop before DP changes");
      ++count;
    }
    if (count != (bf16 ? 770 : 1077)) throw std::runtime_error("incomplete archived configuration universe");
    total += count;
    std::cerr << "CG_INPUT model=" << name << " metrics=" << symbolic.coupling_metrics.edges.size()
              << " configs=" << count << " excluded_historical_nonpass=" << excluded
              << " recovered_numerical_failures=" << recovered << '\n';
    ChainDP dp(cost, candidates);
    auto same_solution = [](ChainDpSolution const& a, ChainDpSolution const& b) {
      if (a.feasible != b.feasible || a.residency.ctas_per_sm != b.residency.ctas_per_sm ||
          a.configs.size() != b.configs.size() || !Bits(a.cost.total_ns,b.cost.total_ns)) return false;
      for (std::size_t i = 0; i < a.configs.size(); ++i) {
        auto const& x = a.configs[i]; auto const& y = b.configs[i];
        if (x.tile_m != y.tile_m || x.tile_n != y.tile_n || x.tile_k != y.tile_k ||
            x.stages != y.stages || x.split_k != y.split_k) return false;
      }
      return true;
    };
    for (auto mode : {DpMode::kUniform, DpMode::kPerOperatorSplit, DpMode::kPerOperator}) {
      ChainDpOptions options; options.mode = mode;
      if (!same_solution(dp.Solve(legacy, options), dp.Solve(concrete, options)))
        throw std::runtime_error("CG input DP choice/bit gate failed; stop");
    }
    ChainDpOptions options; options.mode = DpMode::kUniform;
    auto finite = dp.SolveFiniteParameter(symbolic, {"S", 1, 16, {}}, options);
    for (auto const& piece : finite.pieces) {
      for (int seq = piece.begin; seq <= piece.end; ++seq) {
        auto independent = legacy; independent.dims = {seq, 3, seq+3};
        if (!same_solution(piece.points.at(seq-piece.begin), dp.Solve(independent, options)))
          throw std::runtime_error("finite DP concrete choice/bit gate failed; stop");
      }
      auto const& choice = piece.points.front();
      for (std::size_t gemm = 0; gemm < choice.configs.size(); ++gemm) {
        auto const& c = choice.configs[gemm];
        dp_output << name << '\t' << piece.begin << '\t' << piece.end << '\t'
          << choice.residency.ctas_per_sm << '\t' << gemm << '\t' << c.tile_m << '\t'
          << c.tile_n << '\t' << c.tile_k << '\t' << c.stages << '\t' << c.split_k << '\n';
      }
    }
    std::cerr << "FINITE_DP model=" << name << " domain=S:1..16 past=3 mode=uniform points="
              << finite.evaluated_points << " pieces=" << finite.pieces.size()
              << " all_concrete_choices_equal=1 input_modes_equal=3\n";
  }
  std::cerr << "INPUT_GATE dtype=" << dtype << " matched=" << total
            << " expected=" << (bf16 ? 1540 : 2154) << '\n';
  if (!bf16) std::cerr << "FP32_PARTIAL_COST_GATE matched=" << total << " expected=2154\n";
  return 0;
} catch (std::exception const& error) {
  std::cerr << "tilemega-parametric: " << error.what() << '\n'; return 2;
}
