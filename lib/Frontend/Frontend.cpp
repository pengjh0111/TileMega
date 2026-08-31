// SPDX-License-Identifier: BSD-3-Clause
#include <tilemega/Frontend/SymbolicShapeBridge.h>
#include <tilemega/Frontend/TorchExportImporter.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
namespace tilemega::frontend {
ImportedProgram TorchExportImporter::Import(std::string const& path) const {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot read exported program: " + path);
  return {{std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()}};
}
SymbolicShape SymbolicShapeBridge::Parse(std::string const& env) const {
  // TODO(P1.2): replace the lossless placeholder with ShapeEnv guard parsing.
  return {{env}, {}};
}
}  // namespace tilemega::frontend
