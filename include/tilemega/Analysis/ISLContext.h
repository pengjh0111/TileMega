// SPDX-License-Identifier: BSD-3-Clause
// Skeleton refs: §2 Presburger domains and §5.2 (Phase 3 stub).
#pragma once
#include <string>
#include <utility>
namespace tilemega::analysis {
class ISLContext {
 public:
  explicit ISLContext(std::string parameters = {}) : parameters_(std::move(parameters)) {}
  std::string const& parameters() const { return parameters_; }
 private:
  std::string parameters_;
};
}  // namespace tilemega::analysis
