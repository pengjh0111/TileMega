"""Execute emitted V3 layer/tail/token edges and audit release-before-prefetch."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from test_mx5_stage_program_scope import DRIVER, PREFIX, ROOT


class StageProgramV3Tests(unittest.TestCase):
    def test_arrival_precedes_every_prefetch_and_wait_closes_each_edge(self):
        compiler = shutil.which('g++') or shutil.which('c++')
        if compiler is None:
            self.skipTest('host C++17 compiler required')
        driver = DRIVER.replace('s.kind = kind;', 's.version = 3; s.kind = kind;')
        driver = driver.replace('auto invalid = Spec', '''
  auto bad_version = Spec(1, 1, StageProgramKind::kDecode);
  bad_version.version = 4;
  bool rejected_version = false;
  try { EmitStageProgram(bad_version); } catch (std::invalid_argument const&) { rejected_version = true; }
  assert(rejected_version);
  auto invalid = Spec''')
        prefix = PREFIX.replace('auto RunStageV2(bool enabled, Packet incoming, int operand, Shape, char*, Next next)',
            'auto RunStageV3(bool enabled, Packet incoming, int operand, Shape, char*, Next next, auto arrive)')
        prefix = prefix.replace('template<class B, class Next>', 'template<class B, class Next, class Arrive>')
        prefix = prefix.replace('Next next, auto arrive)', 'Next next, Arrive arrive)')
        prefix = prefix.replace('events.push_back("B"', 'events.push_back("W"')
        prefix = prefix.replace('P::template RunStep<P>(binding, nullptr, incoming, barrier, run);', '''
    auto arrive = [&] { events.push_back("A" + std::to_string(stage)); };
    P::template RunStep<P>(binding, nullptr, incoming, arrive, barrier, run);''')
        prefix = prefix.replace('expected.push_back("P" + std::to_string(next));', '''
      expected.push_back("A" + std::to_string(expected_stage));
      expected.push_back("P" + std::to_string(next));''')
        prefix = prefix.replace('expected.push_back("B"', 'expected.push_back("W"')
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            (directory/'emit.cpp').write_text(driver)
            subprocess.run([compiler, '-std=c++17', '-O2', '-UNDEBUG',
                '-I'+str(ROOT/'lib/Codegen'), str(directory/'emit.cpp'),
                str(ROOT/'lib/Codegen/StageProgram.cpp'), '-o', str(directory/'emit')],
                check=True, capture_output=True, text=True)
            subprocess.run([str(directory/'emit'), str(directory/'generated.cpp')], check=True)
            source = (directory/'generated.cpp').read_text()
            self.assertNotIn('barrier();', source)
            self.assertNotIn('RunStageV2', source)
            self.assertEqual(source.count('auto pf_T0 = mx::RunStageV3<S4Last>'), 18)
            (directory/'protocol.cpp').write_text(prefix + source)
            subprocess.run([compiler, '-std=c++17', '-O2', '-UNDEBUG', str(directory/'protocol.cpp'),
                            '-o', str(directory/'protocol')], check=True, capture_output=True, text=True)
            result = subprocess.run([str(directory/'protocol')], check=True, capture_output=True, text=True)
            self.assertIn('PASS 18 generic emission cases', result.stdout)


if __name__ == '__main__':
    unittest.main()
