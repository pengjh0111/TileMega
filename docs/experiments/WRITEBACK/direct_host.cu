// SPDX-License-Identifier: BSD-3-Clause
// The model, geometry and dependencies are identical; only the Plan carrier
// bypasses CG/codegen and receives the independently solved host table.
#define main r6_generated_main
#include "interval_closure/gqa2.cu"
#undef main
#include <fstream>
#include <sstream>
#include <stdexcept>

int main(int argc, char** argv) try {
  if (argc != 6)
    throw std::invalid_argument("direct_host FIXTURE SEQ PAST GRID SOLVED_TABLE");
  std::ifstream input(argv[5]);
  std::string line;
  if (!std::getline(input, line) || line != "node\tworker\tslot")
    throw std::invalid_argument("invalid direct solver table");
  std::vector<std::int32_t> workers, slots;
  while (std::getline(input, line)) {
    std::istringstream row(line);
    int node, worker, slot;
    if (!(row >> node >> worker >> slot) || node != int(workers.size()))
      throw std::invalid_argument("noncontiguous direct solver table");
    workers.push_back(worker);
    slots.push_back(slot);
  }
  tilemega::codegen::RuntimePlanDesc direct;
  direct.mode = 3;  // Public RuntimePlanDesc encoding for a materialized table.
  direct.eft_worker = workers.data();
  direct.eft_slot = slots.data();
  direct.eft_nodes = workers.size();
  direct.eft_seq = std::stoul(argv[2]);
  direct.eft_past = std::stoul(argv[3]);
  direct.eft_grid = std::stoul(argv[4]);
  auto variant = kRuntimeVariants[0];
  variant.plan = direct;  // No interval table or generated pi/sigma is read.
  auto model = kModel;
  model.runtime_variants = &variant;
  return tilemega::codegen::RunModel(model, argv[1]);
} catch (std::exception const& error) {
  std::fprintf(stderr, "%s\n", error.what());
  return 2;
}
