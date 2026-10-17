import importlib.util
import pathlib
import unittest

path = pathlib.Path(__file__).resolve().parents[1] / 'scripts/r16_moe_pilot_screen.py'
spec = importlib.util.spec_from_file_location('screen', path)
screen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(screen)


class ScreenTests(unittest.TestCase):
    def setUp(self):
        self.inc = [dict(e2e_seconds=e, tpot_seconds=(e-2)/1023) for e in (308,309,310)]

    def test_clearly_slower(self):
        self.assertTrue(screen.evaluate(self.inc,[500,501,502],[3,3.1,3.2])['screened_out'])

    def test_one_fast_repeat_prevents_screen(self):
        self.assertFalse(screen.evaluate(self.inc,[330,500,501],[3,3,3])['screened_out'])

    def test_large_range_prevents_screen(self):
        self.assertFalse(screen.evaluate(self.inc,[350,351,800],[3,3,3])['screened_out'])

    def test_ttft_uncertainty_uses_upper_bound(self):
        self.assertFalse(screen.evaluate(self.inc,[350,351,352],[2,3,200])['screened_out'])

    def test_invalid_protocol(self):
        for pilot in ([500,501], [500,501,float('nan')]):
            with self.assertRaises(ValueError):screen.evaluate(self.inc,pilot,[3,3,3])
        with self.assertRaises(ValueError):screen.evaluate(self.inc,[500]*3,[3]*3,1.01)


if __name__ == '__main__':
    unittest.main()
