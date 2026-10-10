import unittest
import torch
from tilemega.dnn.check_generated import output_metrics


class ModelMetrics(unittest.TestCase):
    def test_restoration_uses_relative_psnr(self):
        reference=torch.zeros(2,3,4,4)
        bf16=torch.full_like(reference,.01)
        actual=torch.full_like(reference,.011)
        result=output_metrics(actual,reference,bf16,restoration=True)
        self.assertLess(result['cosine_min'],.999)
        self.assertEqual(len(result['psnr_fp32']),2)
        with self.assertRaisesRegex(AssertionError,'PSNR'):
            output_metrics(torch.full_like(reference,.02),reference,bf16,restoration=True)

    def test_encoder_cosine_is_unchanged(self):
        reference=torch.tensor([[1.,0.]])
        with self.assertRaisesRegex(AssertionError,'cosine'):
            output_metrics(torch.tensor([[0.,1.]]),reference,reference)


if __name__=='__main__':unittest.main()
