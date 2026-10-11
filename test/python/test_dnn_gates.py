import math
import unittest

from tilemega.dnn.gates import evaluate, summarize


class DnnGates(unittest.TestCase):
    def classification(self, **values):
        return dict(cosine=.9986, bf16_cosine=.999, tm_top1_match=.801,
                    bf16_top1_match=.8, **values)

    def test_bf16_relative_gate_and_one_sided_agreement(self):
        for model in ('resnet18', 'mbv1', 'mbv2'):
            metrics = self.classification()
            self.assertTrue(evaluate(model, metrics)['passed'])
            # TM can exceed BF16 agreement by any amount; it is not an absolute gap.
            metrics['tm_top1_match'] = .95
            self.assertTrue(evaluate(model, metrics)['passed'])
            metrics['tm_top1_match'] = .794
            self.assertFalse(evaluate(model, metrics)['checks']['top1_agreement'])
            metrics['tm_top1_match'] = .8
            metrics['cosine'] = .9984
            self.assertFalse(evaluate(model, metrics)['checks']['logits_cosine'])

    def test_bert_requires_both_mean_cosines(self):
        metrics = dict(hidden_cosine=.94, bf16_hidden_cosine=.95,
                       pooler_cosine=.9986, bf16_pooler_cosine=.999)
        self.assertTrue(evaluate('bert', metrics)['passed'])
        for key in ('hidden_cosine', 'pooler_cosine'):
            changed = dict(metrics, **{key: 0.9})
            self.assertFalse(evaluate('bert', changed)['passed'])

    def test_bert_hidden_mean_is_weighted_by_nonpadding_tokens(self):
        records = [dict(valid_length=1, hidden_cosine_sum=.9, bf16_hidden_cosine_sum=.95,
                        hidden_cosine_min=.9, bf16_hidden_cosine_min=.95,
                        tm_bf16_hidden_cosine_min=.96, pooler_cosine=.99, bf16_pooler_cosine=.995),
                   dict(valid_length=3, hidden_cosine_sum=2.4, bf16_hidden_cosine_sum=2.7,
                        hidden_cosine_min=.7, bf16_hidden_cosine_min=.85,
                        tm_bf16_hidden_cosine_min=.91, pooler_cosine=.98, bf16_pooler_cosine=.99)]
        metrics, legacy = summarize('bert', records)
        self.assertAlmostEqual(metrics['hidden_cosine'], .825)
        self.assertAlmostEqual(metrics['bf16_hidden_cosine'], .9125)
        self.assertAlmostEqual(metrics['pooler_cosine'], .985)
        self.assertEqual(metrics['hidden_cosine_min'], .7)
        self.assertEqual(metrics['nonpadding_tokens'], 4)
        self.assertFalse(legacy)

    def test_classification_reports_label_accuracy_separately(self):
        common = dict(cosine=.9995, bf16_cosine=.9995, tm_top1_match=True,
                      bf16_top1_match=True, bf16_top1_correct=True, fp32_top1_correct=True)
        records = [dict(common, tm_top1_correct=True), dict(common, tm_top1_correct=False)]
        metrics, legacy = summarize('resnet18', records)
        self.assertEqual(metrics['tm_top1_correct'], .5)
        self.assertEqual(metrics['tm_top1_match'], 1)
        self.assertTrue(legacy)
        self.assertTrue(evaluate('resnet18', metrics)['passed'])

    def test_nafnet_psnr_and_mse_are_independent(self):
        metrics = dict(tm_gt_psnr=32.0, bf16_gt_psnr=32.05,
                       tm_fp32_mse=1.4e-5, bf16_fp32_mse=1e-5)
        self.assertTrue(evaluate('nafnet', metrics)['passed'])
        self.assertFalse(evaluate('nafnet', dict(metrics, tm_gt_psnr=31.94))['passed'])
        self.assertFalse(evaluate('nafnet', dict(metrics, tm_fp32_mse=1.6e-5))['passed'])
        self.assertTrue(evaluate('nafnet', dict(metrics, tm_gt_psnr=34))['passed'])
        self.assertTrue(evaluate('nafnet', dict(metrics, tm_fp32_mse=0, bf16_fp32_mse=0))['passed'])
        self.assertFalse(evaluate('nafnet', dict(metrics, bf16_fp32_mse=0))['passed'])

    def test_invalid_metrics_cannot_pass(self):
        for value in (math.nan, math.inf, -math.inf):
            with self.assertRaises(ValueError):
                evaluate('mbv2', dict(self.classification(), cosine=value))
        with self.assertRaises(ValueError):
            evaluate('nafnet', dict(tm_gt_psnr=30, bf16_gt_psnr=30,
                                   tm_fp32_mse=-1, bf16_fp32_mse=1))
        with self.assertRaises(ValueError):
            evaluate('unsupported', {})


if __name__ == '__main__':
    unittest.main()
