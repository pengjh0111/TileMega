"""Compile the generic host emitter and check emitted stage/next/barrier order; no GPU."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import textwrap
import unittest

ROOT = Path(__file__).resolve().parents[2]

PREFIX = r'''
#include <cassert>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#define __device__
#define __forceinline__ inline
std::vector<std::string> events;
struct Packet { int owner; };
struct Shape {};
struct BufferRow { char const* name; int dtype; std::uint64_t constant, per_batch; char const* recipe; int slot, layer; };
template<int Id> struct Body {
  using Prefetched = Packet;
  static Packet Prefetch(int operand, Shape) {
    assert(operand == Id); events.push_back("P" + std::to_string(Id)); return {Id};
  }
};
namespace tilemega { namespace backend { namespace mx {} }
namespace codegen { namespace mx {
template<class B, class Next>
auto RunStageV2(bool enabled, Packet incoming, int operand, Shape, char*, Next next) {
  assert(enabled && incoming.owner == B::IdValue && operand == B::IdValue);
  events.push_back("R" + std::to_string(B::IdValue)); return next();
}
} } }
template<int Id> struct TypedBody : Body<Id> { static constexpr int IdValue = Id; };
template<class P> void Check(int layers, int tail_count) {
  assert(P::kStageCount == 5 * layers + tail_count);
  int binding = 0;
  auto incoming = P::template EntryPrefetch<P>(binding);
  std::vector<std::string> expected{"P0"};
  for (int step = 0; step < 2; ++step) {
    int stage = 0;
    auto barrier = [&] {
      events.push_back("B" + std::to_string(stage));
      if (stage + 1 == P::kStageCount && step + 1 < 2) events.push_back("X-next-token-broadcast");
      ++stage;
    };
    auto run = [&](int stage_class) { events.push_back("D" + std::to_string(stage_class)); return true; };
    P::template RunStep<P>(binding, nullptr, incoming, barrier, run);
    assert(stage == P::kStageCount && incoming.owner == 0);
    int expected_stage = 0;
    auto edge = [&](int body, int next) {
      expected.push_back("D" + std::to_string(body));
      expected.push_back("R" + std::to_string(body));
      expected.push_back("P" + std::to_string(next));
      expected.push_back("B" + std::to_string(expected_stage));
      if (++expected_stage == 5 * layers + tail_count && step == 0)
        expected.push_back("X-next-token-broadcast");
    };
    for (int layer = 0; layer < layers; ++layer) {
      for (int body = 0; body < 4; ++body) edge(body, body + 1);
      edge(4, layer + 1 == layers ? 5 : 0);
    }
    for (int tail = 0; tail < tail_count; ++tail)
      edge(5 + tail, tail + 1 == tail_count ? 0 : 6 + tail);
  }
  assert(events == expected); events.clear();
}
'''

DRIVER = r'''
#include "StageProgram.h"
#include <cassert>
#include <fstream>
#include <stdexcept>
#include <string>
using namespace tilemega::codegen;
StageProgramSpec Spec(int layers, int tails, StageProgramKind kind) {
  StageProgramSpec s;
  s.kind = kind; s.namespace_name = "case_" + std::to_string(layers) + "_" + std::to_string(tails) + "_" + std::to_string(int(kind));
  s.plan_declarations = ""; s.loop.begin = "3"; s.loop.end = std::to_string(3 + layers);
  s.operand_parameters = "int& b"; s.run_parameters = "int& b, char* smem";
  s.entry_prefetch_operands = "0"; s.buffer_row_type = "BufferRow";
  s.external_buffers = {{"unused", "0", 0, 0, "", "0", 0}};
  for (int i = 0; i < 5; ++i) {
    auto name = "S" + std::to_string(i);
    StageProgramStage row{name, "TypedBody<" + std::to_string(i) + ">", "Shape", std::to_string(i), std::to_string(i),
                          i == 4 ? "S0" : "S" + std::to_string(i + 1), "", i};
    if (i == 4) { row.terminal_body = row.body_type; row.terminal_next = "T0"; row.terminal_prefetch_operands = "5"; }
    s.stages.push_back(row);
  }
  for (int i = 0; i < tails; ++i)
    s.tail_stages.push_back({"T" + std::to_string(i), "TypedBody<" + std::to_string(5 + i) + ">", "Shape",
        std::to_string(5 + i), std::to_string(5 + i), i + 1 == tails ? "S0" : "T" + std::to_string(i + 1), "", 5 + i});
  return s;
}
int main(int argc, char** argv) {
  assert(argc == 2); std::ofstream out(argv[1]);
  for (int kind = 0; kind < 3; ++kind) for (int layers : {1, 2, 28}) for (int tails : {1, 2})
    out << EmitStageProgram(Spec(layers, tails, StageProgramKind(kind)));
  auto invalid = Spec(1, 1, StageProgramKind::kDecode); invalid.stages.back().terminal_next = "missing";
  bool rejected = false; try { EmitStageProgram(invalid); } catch (std::invalid_argument const&) { rejected = true; }
  assert(rejected); out << "int main() {\n";
  for (int kind = 0; kind < 3; ++kind) for (int layers : {1, 2, 28}) for (int tails : {1, 2})
    out << "Check<case_" << layers << "_" << tails << "_" << kind << "::Plan>(" << layers << "," << tails << ");\n";
  out << "std::cout << \"PASS 18 generic emission cases, two steps each, exact next/barrier/broadcast order\\n\"; }\n";
}
'''


class StageProgramScopeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.compiler = shutil.which('g++') or shutil.which('c++')
        if cls.compiler is None:
            raise unittest.SkipTest('a host C++17 compiler is needed')
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        driver = cls.directory / 'emit.cpp'
        driver.write_text(DRIVER)
        emitter = cls.directory / 'emit'
        subprocess.run([cls.compiler, '-std=c++17', '-O2', '-UNDEBUG', '-I' + str(ROOT / 'lib/Codegen'),
                        str(driver), str(ROOT / 'lib/Codegen/StageProgram.cpp'), '-o', str(emitter)], check=True, capture_output=True, text=True)
        cls.generated = cls.directory / 'generated.cpp'
        subprocess.run([str(emitter), str(cls.generated)], check=True, capture_output=True, text=True)
        cls.source = cls.generated.read_text()
        cls.generated.write_text(PREFIX + '\n' + cls.source)

    def test_emitted_protocol_executes_in_order(self):
        executable = self.directory / 'protocol'
        process = subprocess.run([self.compiler, '-std=c++17', '-O2', '-UNDEBUG', str(self.generated), '-o', str(executable)], capture_output=True, text=True)
        self.assertEqual(process.returncode, 0, process.stderr)
        result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
        self.assertIn('PASS 18 generic emission cases', result.stdout)

    def test_terminal_packet_is_local_and_consumed_before_break(self):
        self.assertNotIn('typename T0::Prefetched pf_T0;', self.source)
        self.assertEqual(self.source.count('auto pf_T0 = mx::RunStageV2<S4Last>'), 18)
        for piece in self.source.split('      if (')[1:]:
            branch = piece.split('        break;', 1)[0]
            self.assertIn('auto pf_T0 = ', branch)
            self.assertIn('mx::RunStageV2<T0>', branch)
            self.assertTrue(branch.rstrip().endswith('barrier();'))

    def test_interface_stays_generic_and_constexpr(self):
        self.assertNotIn('Head', self.source)
        self.assertNotIn('Qkv', self.source)
        self.assertIn('kProgramKind = 0', self.source)
        self.assertIn('kProgramKind = 1', self.source)
        self.assertIn('kProgramKind = 2', self.source)
        self.assertIn('mx::RunStageV2<T1>', self.source)
        self.assertNotIn('stage_descriptor', self.source)


if __name__ == '__main__':
    unittest.main(verbosity=2)
