// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <tilemega/Support/Json.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Support/Allocator.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SHA256.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/StringSaver.h>
#include <cerrno>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace tilemega::commands::mx_decode::compile_receipt_v1 {
using json::Array;
using json::Object;
using json::Value;
inline std::string Hash(std::string const& path) {
  auto file = llvm::MemoryBuffer::getFile(path);
  if (!file) throw std::runtime_error("MX-5 receipt cannot read " + path);
  llvm::SHA256 hash; hash.update(file.get()->getBuffer());
  return llvm::toHex(hash.final(), true);
}
inline Value Record(std::string const& path) {
  return Object{{"path", std::filesystem::absolute(path).lexically_normal().string()}, {"sha256", Hash(path)}};
}
inline void WriteExclusive(std::string const& path, Value const& value) {
  int fd = ::open(path.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  if (fd < 0) throw std::runtime_error("MX-5 receipt refuses missing/unwritable/reused output: " + path);
  auto data = value.Dump() + "\n";
  std::size_t offset = 0;
  while (offset < data.size()) {
    auto n = ::write(fd, data.data() + offset, data.size() - offset);
    if (n < 0 && errno == EINTR) continue;
    if (n <= 0) { ::close(fd); throw std::runtime_error("MX-5 receipt write failed: " + path); }
    offset += std::size_t(n);
  }
  int result = ::fsync(fd); int closed = ::close(fd);
  if (result || closed) throw std::runtime_error("MX-5 receipt flush failed: " + path);
}
inline void RequireFresh(std::string const& out) {
  for (char const* suffix : {"", ".cu", ".build_command.txt", ".runtime_flags.txt", ".mx5_pin.json",
       ".source.json", ".source.json.patch", ".ptxas.log", ".compile_receipt.json", ".resources.json",
       ".resource_report.log", ".resource_failure.json", ".plan.json", ".identity.json"})
    if (std::filesystem::exists(out + suffix))
      throw std::invalid_argument("MX-5 resource attempt needs a fresh output path: " + out + suffix);
}
inline std::string CompilerPath(std::string const& name) {
  auto found = llvm::sys::findProgramByName(name);
  if (!found) throw std::runtime_error("MX-5 cannot resolve device compiler: " + name);
  return std::filesystem::absolute(*found).lexically_normal().string();
}
inline std::vector<std::string> Arguments(std::string const& command) {
  llvm::BumpPtrAllocator allocator; llvm::StringSaver saver(allocator);
  llvm::SmallVector<char const*, 32> tokens;
  llvm::cl::TokenizeGNUCommandLine(command, saver, tokens);
  std::vector<std::string> argv;
  for (auto token : tokens) argv.emplace_back(token);
  if (argv.empty() || !std::filesystem::path(argv[0]).is_absolute())
    throw std::invalid_argument("MX-5 device compiler must have an absolute executable path");
  return argv;
}
inline Object Inputs(std::string const& out, std::string const& pin,
                     std::string const& compiler, std::string const& producer) {
  return Object{{"pin", Record(pin)}, {"resolved_pin", Record(out + ".mx5_pin.json")},
    {"generated_source", Record(out + ".cu")}, {"command", Record(out + ".build_command.txt")},
    {"source_snapshot", Record(out + ".source.json")}, {"source_patch", Record(out + ".source.json.patch")},
    {"compiler", Record(compiler)}, {"producer", Record(producer)}};
}
// Execute the recorded argv directly; shell expansion cannot change the observed command.
inline int Run(std::vector<std::string> const& argv, std::string const& out, Object const& inputs) {
  if (argv.empty()) throw std::invalid_argument("MX-5 receipt has empty argv");
  std::vector<char*> args; for (auto const& arg : argv) args.push_back(const_cast<char*>(arg.c_str()));
  args.push_back(nullptr);
  int log = ::open((out + ".ptxas.log").c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
  if (log < 0) throw std::runtime_error("MX-5 receipt cannot create fresh compiler log");
  pid_t child = ::fork();
  if (child < 0) { ::close(log); throw std::runtime_error("MX-5 cannot fork device compiler"); }
  if (child == 0) {
    if (::dup2(log, STDOUT_FILENO) < 0 || ::dup2(log, STDERR_FILENO) < 0) ::_exit(126);
    ::close(log); ::execv(args[0], args.data()); ::_exit(127);
  }
  ::close(log); int status = 0;
  while (::waitpid(child, &status, 0) < 0) {
    if (errno == EINTR) continue;
    throw std::runtime_error("MX-5 cannot wait for device compiler");
  }
  int code = WIFEXITED(status) ? WEXITSTATUS(status) : WIFSIGNALED(status) ? 128 + WTERMSIG(status) : 125;
  Object files = inputs; files.emplace_back("log", Record(out + ".ptxas.log"));
  Value binary = std::filesystem::is_regular_file(out) ? Record(out) : Value();
  files.emplace_back("binary", binary);
  Array arguments; for (auto const& arg : argv) arguments.emplace_back(arg);
  Object receipt{{"schema_version", 1}, {"kind", "mx5_native_compile_receipt"},
      {"argv", arguments}, {"returncode", code}, {"wait_status", status},
      {"same_compile_mir", Hash(out + ".ptxas.log")}, {"files", files},
      {"binary_sha256", binary.IsNull() ? Value() : binary.At("sha256")}};
  Array changed;
  for (auto const& [name, ref] : inputs) {
    receipt.emplace_back(name + "_sha256", ref.At("sha256"));
    auto const& path = ref.At("path").AsString("input path");
    if (!std::filesystem::is_regular_file(path) || Hash(path) != ref.At("sha256").AsString("input hash"))
      changed.emplace_back(name);
  }
  receipt.emplace_back("log_sha256", Hash(out + ".ptxas.log"));
  receipt.emplace_back("changed_inputs", changed);
  WriteExclusive(out + ".compile_receipt.json", receipt);
  if (!changed.empty()) throw std::runtime_error("MX-5 compiler inputs changed during compile; receipt retained");
  return code;
}
}  // namespace tilemega::commands::mx_decode::compile_receipt_v1
