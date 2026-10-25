"""CPU-only contract tests for the versioned MX5 GEMM reference."""
import ast
from decimal import Decimal, localcontext
from fractions import Fraction
import importlib.util
from pathlib import Path
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode


SOURCE = Path(__file__).resolve().parents[2] / 'experiments/mx5/stage_bench.py'
SPEC = importlib.util.spec_from_file_location('mx5_stage_bench_reference_test', SOURCE)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def baseline(x, w, residual, op, norm):
    """Freeze the original rounding contract on CPU, independently of the harness."""
    result = x.float() @ w.float().T
    if norm:
        result *= torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-6)
    result = result.bfloat16()
    if op == 1:
        return (result.float() + residual.float()).bfloat16()
    if op == 2:
        pair = result.reshape(x.shape[0], -1, 2, 16)
        gate, up = pair[:, :, 0].float(), pair[:, :, 1].float()
        return ((gate / (1 + (-gate).exp())).bfloat16().float() * up).bfloat16().reshape(x.shape[0], -1)
    return result


class Trace(TorchDispatchMode):
    def __init__(self):
        self.ops = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        if isinstance(result, torch.Tensor):
            self.ops.append((str(func), result.dtype, result.device.type))
        return result


class StageReferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        generator = torch.Generator().manual_seed(137)
        cls.x = (torch.randn(3, 64, generator=generator) * .1).bfloat16()
        cls.w = (torch.randn(64, 64, generator=generator) * .1).bfloat16()
        cls.residual = torch.randn(3, 64, generator=generator).bfloat16()

    def test_rounding_contract_all_ops_and_norm_modes(self):
        for op in (0, 1, 2):
            for norm in (False, True):
                with self.subTest(op=op, norm=norm):
                    actual = BENCH.cpu_fp32_reference(self.x, self.w, self.residual, op, norm, torch)
                    expected = baseline(self.x, self.w, self.residual, op, norm)
                    self.assertEqual(actual.dtype, torch.bfloat16)
                    self.assertEqual(actual.device.type, 'cpu')
                    self.assertTrue(torch.equal(actual.view(torch.int16), expected.view(torch.int16)))

    def test_matmul_rms_silu_are_cpu_fp32(self):
        trace = Trace()
        with trace:
            BENCH.cpu_fp32_reference(self.x, self.w, self.residual, 2, True, torch)
        for name in ('aten.mm.default', 'aten.mean.dim', 'aten.rsqrt.default', 'aten.exp.default'):
            observed = [(dtype, device) for op, dtype, device in trace.ops if op == name]
            self.assertTrue(observed, name)
            self.assertTrue(all(dtype == torch.float32 and device == 'cpu' for dtype, device in observed), name)
        self.assertTrue(all(device == 'cpu' for _, _, device in trace.ops))
        # GEMM result, SiLU result, and final product each round to BF16.
        self.assertEqual(sum(dtype == torch.bfloat16 for op, dtype, _ in trace.ops if op == 'aten._to_copy.default'), 3)

    def test_rejects_non_bf16_sources(self):
        with self.assertRaisesRegex(ValueError, 'CPU BF16'):
            BENCH.cpu_fp32_reference(self.x.float(), self.w, self.residual, 0, False, torch)

    def test_original_acceptance_constants_and_formula(self):
        expected = ast.parse('not t.all((out.float() - reference.float()).abs() <= .02 + .016 * reference.float().abs()).item()', mode='eval').body
        conditions = [node.test for node in ast.walk(ast.parse(SOURCE.read_text()))
                      if isinstance(node, ast.If) and any(isinstance(part, ast.Constant) and part.value == 'repeated-stage output mismatch' for part in ast.walk(node))]
        self.assertEqual(len(conditions), 1)
        self.assertEqual(ast.dump(conditions[0]), ast.dump(expected))

    def test_one_unconditional_versioned_oracle(self):
        main = next(node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == 'main')
        calls = [node for node in ast.walk(main) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'cpu_fp32_reference']
        self.assertEqual(len(calls), 1)
        self.assertEqual([node.id for node in calls[0].args], ['x_cpu', 'w_cpu', 'residual_cpu', 'op', 'norm', 't'])
        # Require the fixed oracle assignment directly in the per-key loop, outside any result-dependent branch.
        loop = next(node for node in ast.walk(main) if isinstance(node, ast.For) and isinstance(node.target, ast.Tuple))
        assignments = [node for node in loop.body if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == 'reference' for target in node.targets)]
        self.assertEqual(len(assignments), 1)
        assignment = assignments[0]
        self.assertIsInstance(assignment.value, ast.Call)
        self.assertEqual(assignment.value.func.attr, 'to')
        self.assertEqual(ast.unparse(assignment.value.args[0]), 'out.device')
        self.assertIs(assignment.value.func.value, calls[0])
        self.assertEqual(BENCH.REFERENCE_BACKEND, 'torch_cpu')
        self.assertEqual(BENCH.REFERENCE_PRECISION, 'float32')
        self.assertEqual(BENCH.REFERENCE_ORACLE_VERSION, 'gemm_cpu_fp32_v1')

    def test_small_bf16_midpoint_with_exact_decimal_proof(self):
        # All source numbers are exactly BF16. Dot=313/128 is the midpoint
        # between BF16 2.4375 and 2.453125; RMS epsilon moves it strictly below.
        x = torch.tensor([[1, 1]], dtype=torch.bfloat16)
        w = torch.zeros(32, 2, dtype=torch.bfloat16)
        w[0] = torch.tensor([1.21875, 1.2265625], dtype=torch.bfloat16)
        w[16] = torch.tensor([-2, -2.125], dtype=torch.bfloat16)
        residual = torch.zeros(1, 32, dtype=torch.bfloat16)
        dot = sum((Fraction(a) * Fraction(b) for a, b in zip(x[0].float().tolist(), w[0].float().tolist())), Fraction())
        ss = sum((Fraction(a)**2 for a in x[0].float().tolist()), Fraction())
        midpoint = Fraction(313, 128)
        variance = ss / 2 + Fraction(torch.tensor(1e-6, dtype=torch.float32).item())
        self.assertEqual(dot, midpoint)
        self.assertLess(dot * dot - midpoint * midpoint * variance, 0)
        with localcontext() as context:
            context.prec = 70
            def decimal(value):
                return Decimal(value.numerator) / Decimal(value.denominator)
            normalized_gate = decimal(dot) / decimal(variance).sqrt()
            normalized_up = Decimal('-4.125') / decimal(variance).sqrt()
            self.assertLess(normalized_gate, decimal(midpoint))
            self.assertLess(decimal(midpoint) - normalized_gate, Decimal('0.000002'))
            gate = torch.tensor(float(normalized_gate), dtype=torch.float64).bfloat16().item()
            up = torch.tensor(float(normalized_up), dtype=torch.float64).bfloat16().item()
            self.assertEqual(Fraction(gate), Fraction(39, 16))
            activation = Decimal(gate) / (1 + (-Decimal(gate)).exp())
            activation_bf16 = torch.tensor(float(activation), dtype=torch.float64).bfloat16().item()
            expected = torch.tensor(activation_bf16 * up, dtype=torch.float64).bfloat16()
        actual = BENCH.cpu_fp32_reference(x, w, residual, 2, True, torch)
        self.assertEqual(actual[0, 0].view(torch.int16).item(), expected.view(torch.int16).item())


if __name__ == '__main__':
    unittest.main(verbosity=2)
