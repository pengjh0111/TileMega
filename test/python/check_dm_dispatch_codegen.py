"""Compile emitted dispatch with runners that reject unreachable instantiations."""
import argparse
from pathlib import Path
import re
import subprocess


def function(source, name):
    start = source.index('__device__ inline ' + name)
    start = source.rfind('template<', 0, start)
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end].replace('__device__ ', '').replace(
        'asm volatile("trap;")', 'throw std::runtime_error("unbound")')


def check(source_path, compiler):
    source = source_path.read_text()
    bindings = source[source.index('#define TILEMEGA_DM_GEMM_BINDINGS 1'):]
    bindings = bindings[:bindings.index('} // namespace tilemega::codegen')]
    bindings = bindings[bindings.index('template<int Gemm, int Variant>'):]
    entries = re.findall(r'DmGemmBinding<(\d+), (\d+)>.*?roles = (\d+)u', bindings)
    masks = {(int(g), int(v)): int(role) for g, v, role in entries}
    count = len(re.findall(r'using DmSpec\d+ =', source))
    variants = {v for _, v in masks} | {max(v for _, v in masks) + 1}
    text = '#include <cstdint>\n#include <stdexcept>\n#include <cassert>\n'
    text += bindings + '\nint seen = -1;\n'
    for g in range(count):
        text += f'struct DmSpec{g} {{ static constexpr int id = {g}; }};\n'
    text += '''template<unsigned Mask> struct Runner {
      template<class Spec> void Run() const {
        static_assert((Mask >> Spec::id) & 1u, "unreachable epilogue instantiated");
        seen = Spec::id;
      }
    };
    template<unsigned Mask> struct FusedRunner {
      template<int Channels, class Chain, class Spec> void Run() const {
        static_assert((Mask >> Spec::id) & 1u, "unreachable fusion instantiated");
        seen = Spec::id;
      }
    };
    '''
    text += function(source, 'void DispatchDmEpilogue') + '\n'
    fused = 'bool DispatchDmFused(' in source
    fused_cases = []
    if fused:
        dispatch = function(source, 'bool DispatchDmFused')
        fused_cases = re.findall(r'if\(stage==(\d+)u\).*?DmPrimitiveChain(\d+), DmSpec(\d+)', dispatch)
        for _, primitive, _ in fused_cases:
            text += f'struct DmPrimitiveChain{primitive} {{}};\n'
        text += dispatch + '\n'
    text += 'int main() {\n'
    for v in sorted(variants):
        for combine, role in ((False, 1), (True, 2)):
            allowed = sum(1 << g for g in range(count) if masks.get((g, v), 0) & role)
            for g in range(count + 1):
                valid = g < count and bool(allowed & (1 << g))
                text += (f'seen=-1; try {{ DispatchDmEpilogue<{v}, {str(combine).lower()}>'
                         f'({g}, Runner<{allowed}>{{}}); assert({str(valid).lower()}); }}'
                         f' catch (std::runtime_error const&) {{ assert({str(not valid).lower()}); }}\n'
                         f'assert(seen == {g if valid else -1});\n')
        if fused:
            allowed = sum(1 << g for g in range(count) if masks.get((g, v), 0) & 4)
            for stage, _, g in fused_cases:
                valid = bool(allowed & (1 << int(g)))
                text += (f'seen=-1; assert(DispatchDmFused<{v}>({stage}, FusedRunner<{allowed}>{{}})'
                         f' == {str(valid).lower()}); assert(seen == {g if valid else -1});\n')
    text += '}\n'
    probe = source_path.with_suffix('.dispatch.cpp')
    probe.write_text(text)
    binary = probe.with_suffix('.out')
    subprocess.run([compiler, '-std=c++17', '-O0', str(probe), '-o', str(binary)], check=True)
    subprocess.run([str(binary)], check=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--unit', required=True)
    p.add_argument('--compiler', required=True)
    p.add_argument('--fixture', required=True)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    base = a.out / 'memory.cu'
    subprocess.run([a.unit, 'dnn_memory_codegen', '--emit', str(base)], check=True)
    fused = a.out / 'fused.cu'
    subprocess.run([a.unit, 'dnn_dwpw_fusion', a.fixture, str(fused)], check=True)
    for source in (base, Path(str(base) + '.mixed.cu'), Path(str(base) + '.union.cu'), fused):
        check(source, a.compiler)
    print('DM dispatch: direct/split/fused geometry bounds and sequence union PASS')


if __name__ == '__main__':
    main()
