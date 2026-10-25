// SPDX-License-Identifier: BSD-3-Clause
// Host-only candidate enumeration. From the repository root, build with:
// c++ -std=c++17 -O2 -Iinclude experiments/mx5/domain.cpp -o runs/mx5/domain/domain
#include <tilemega/Backend/Mx/DecodeGeometryV2.h>

#include <charconv>
#include <climits>
#include <cstdio>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_set>

namespace mx = tilemega::backend::mx;

namespace {

struct Options {
  std::string kind;
  int n = 0, k = 0, batch = 0, threads = 576;
  std::optional<int> op, next_registers;
};

constexpr char kUsage[] =
    "usage: domain --kind gemm|head --n N --k K --batch B "
    "[--threads 512|576] [--r-next R] [--op 0|1|2]\n"
    "--op is required for gemm and omitted for head; threads defaults to 576.\n"
    "R_next and prefetch are emitted only when --r-next is explicitly supplied.\n";

int Integer(std::string_view text, std::string const& flag) {
  int value = 0;
  auto const parsed = std::from_chars(text.data(), text.data() + text.size(), value);
  if (parsed.ec != std::errc{} || parsed.ptr != text.data() + text.size() || value < 0)
    throw std::invalid_argument(flag + " requires a nonnegative integer");
  return value;
}

Options Parse(int argc, char** argv) {
  Options options;
  std::unordered_set<std::string> seen;
  for (int i = 1; i < argc; ++i) {
    std::string const flag = argv[i];
    if (!seen.insert(flag).second) throw std::invalid_argument("duplicate argument: " + flag);
    if (flag != "--kind" && flag != "--n" && flag != "--k" && flag != "--batch" &&
        flag != "--threads" && flag != "--r-next" && flag != "--op")
      throw std::invalid_argument("unknown argument: " + flag);
    if (++i == argc) throw std::invalid_argument("missing value for " + flag);
    if (flag == "--kind") options.kind = argv[i];
    else {
      int const value = Integer(argv[i], flag);
      if (flag == "--n") options.n = value;
      else if (flag == "--k") options.k = value;
      else if (flag == "--batch") options.batch = value;
      else if (flag == "--threads") options.threads = value;
      else if (flag == "--r-next") options.next_registers = value;
      else options.op = value;
    }
  }
  if (options.kind != "gemm" && options.kind != "head")
    throw std::invalid_argument("--kind must be gemm or head");
  if (options.n <= 0 || options.k <= 0 || options.batch <= 0)
    throw std::invalid_argument("--n, --k and --batch must be positive");
  if (options.threads != 512 && options.threads != 576)
    throw std::invalid_argument("--threads must be 512 or 576");
  if (options.kind == "gemm" && (!options.op || *options.op > 2))
    throw std::invalid_argument("gemm requires --op 0, 1 or 2");
  if (options.kind == "head" && options.op)
    throw std::invalid_argument("head does not take --op");
  // The production geometry uses 32-bit CTA*unit products.
  if (static_cast<long long>(options.n / 16) * (options.k / 32) > INT_MAX / 104)
    throw std::invalid_argument("shape exceeds production geometry's 32-bit index range");
  return options;
}

template <int TN, int WN, int Depth, mx::MxPart Part>
void Emit(Options const& options) {
  bool const head = options.kind == "head";
  if (head && (WN < 4 || Part != mx::MxPart::kTileRange)) return;
  auto const op = head ? mx::MxOpV2::kStore : static_cast<mx::MxOpV2>(*options.op);
  if (!mx::MxGemmLegalV2(options.n, options.k, options.batch, TN, WN, Depth, op, Part)) return;
  int const first = mx::MxFirstChunksV2<TN, WN, Part>(options.n, options.k, 104);
  std::cout << "{\"kind\":\"" << options.kind << "\",\"n\":" << options.n
            << ",\"k\":" << options.k << ",\"b\":" << options.batch
            << ",\"threads\":" << options.threads << ",\"tn\":" << TN
            << ",\"wn\":" << WN << ",\"depth\":" << Depth
            << ",\"part\":" << static_cast<int>(Part) << ",\"first_chunks\":" << first;
  if (!head) std::cout << ",\"op\":" << *options.op;
  if (options.next_registers) {
    int const width = mx::MxPrefetchWidthV2<TN, WN, Depth, Part>(
        options.n, options.k, 104, options.threads, *options.next_registers);
    std::cout << ",\"r_next\":" << *options.next_registers << ",\"prefetch\":" << width;
  }
  std::cout << "}\n";
}

template <int TN, int WN, int Depth>
void Parts(Options const& options) {
  Emit<TN, WN, Depth, mx::MxPart::kTileRange>(options);
  if constexpr (WN == 1) Emit<TN, WN, Depth, mx::MxPart::kStreamK>(options);
}

template <int TN, int WN>
void Depths(Options const& options) {
  Parts<TN, WN, 1>(options); Parts<TN, WN, 2>(options);
  Parts<TN, WN, 3>(options); Parts<TN, WN, 4>(options);
}

template <int TN>
void Waves(Options const& options) {
  Depths<TN, 1>(options); Depths<TN, 2>(options);
  Depths<TN, 4>(options); Depths<TN, 8>(options);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--help") {
    std::fputs(kUsage, stdout);
    return 0;
  }
  try {
    auto const options = Parse(argc, argv);
    Waves<16>(options); Waves<32>(options); Waves<64>(options); Waves<128>(options);
  } catch (std::exception const& error) {
    std::fprintf(stderr, "%s\n%s", error.what(), kUsage);
    return 2;
  }
}
