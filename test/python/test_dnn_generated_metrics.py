import unittest
import torch
from tilemega.dnn.check_generated import output_metrics, intermediate_metrics


class ModelMetrics(unittest.TestCase):
    def test_intermediate_report_keeps_the_second_image_separate(self):
        fp32=torch.zeros(2,3,4,4)
        bf16=torch.full_like(fp32,.01)
        actual=bf16.clone();actual[1].fill_(1)
        result=intermediate_metrics(actual,fp32,bf16)
        self.assertEqual(result[0]['tm_bf16_mse'],0)
        self.assertGreater(result[1]['tm_bf16_mse'],.9)
        self.assertEqual(result[1]['tm_fp32_mse'],1)
        self.assertEqual([row['batch'] for row in result],[0,1])

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
