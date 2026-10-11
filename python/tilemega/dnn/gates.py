"""R15 DM-W1 acceptance rules, fixed before dataset execution."""
import math


AGREEMENT_MARGIN = 0.005
BF16_ERROR_FACTOR = 1.5
PSNR_MARGIN_DB = 0.1


def summarize(model, records):
    if not records:
        raise ValueError('dataset gate requires samples')
    mean = lambda key: sum(row[key] for row in records) / len(records)
    if model == 'bert':
        metrics = {key: min(row[key] for row in records) for key in
                   ('hidden_cosine_min', 'bf16_hidden_cosine_min', 'tm_bf16_hidden_cosine_min')}
        metrics.update(pooler_cosine_min=min(row['pooler_cosine'] for row in records),
                       bf16_pooler_cosine_min=min(row['bf16_pooler_cosine'] for row in records))
        legacy = metrics['hidden_cosine_min'] >= .999 and metrics['pooler_cosine_min'] >= .999
        tokens = sum(row['valid_length'] for row in records)
        if tokens <= 0:
            raise ValueError('BERT gate requires nonpadding tokens')
        metrics.update(hidden_cosine=sum(row['hidden_cosine_sum'] for row in records)/tokens,
            bf16_hidden_cosine=sum(row['bf16_hidden_cosine_sum'] for row in records)/tokens,
            pooler_cosine=mean('pooler_cosine'), bf16_pooler_cosine=mean('bf16_pooler_cosine'),
            nonpadding_tokens=tokens)
    elif model == 'nafnet':
        metrics = {key: mean(key) for key in ('tm_fp32_psnr', 'bf16_fp32_psnr',
                   'tm_gt_psnr', 'bf16_gt_psnr', 'tm_fp32_mse', 'bf16_fp32_mse')}
        legacy = (metrics['tm_fp32_psnr'] >= metrics['bf16_fp32_psnr'] - 1 and
                  abs(metrics['tm_gt_psnr'] - metrics['bf16_gt_psnr']) <= .05)
    elif model in ('resnet18', 'mbv1', 'mbv2'):
        metrics = {key: mean(key) for key in ('cosine', 'bf16_cosine', 'tm_top1_match',
                   'bf16_top1_match', 'tm_top1_correct', 'bf16_top1_correct', 'fp32_top1_correct')}
        legacy = metrics['cosine'] >= .999 and metrics['tm_top1_match'] >= metrics['bf16_top1_match'] - .002
    else:
        raise ValueError('no registered DNN gate for ' + model)
    return metrics, legacy


def evaluate(model, metrics):
    def number(key, *, infinite=False):
        value = float(metrics[key])
        if math.isnan(value) or (not infinite and not math.isfinite(value)):
            raise ValueError('nonfinite correctness metric: ' + key)
        return value

    def cosine(tm, bf16):
        return 1 - number(tm) <= BF16_ERROR_FACTOR * (1 - number(bf16))

    if model in ('resnet18', 'mbv1', 'mbv2'):
        tm, bf16 = number('tm_top1_match'), number('bf16_top1_match')
        if not (0 <= tm <= 1 and 0 <= bf16 <= 1):
            raise ValueError('top-1 agreement must be a fraction')
        checks = dict(top1_agreement=tm >= bf16 - AGREEMENT_MARGIN,
                      logits_cosine=cosine('cosine', 'bf16_cosine'))
    elif model == 'bert':
        checks = dict(hidden_cosine=cosine('hidden_cosine', 'bf16_hidden_cosine'),
                      pooler_cosine=cosine('pooler_cosine', 'bf16_pooler_cosine'))
    elif model == 'nafnet':
        tm, bf16 = number('tm_fp32_mse'), number('bf16_fp32_mse')
        if tm < 0 or bf16 < 0:
            raise ValueError('MSE must be nonnegative')
        checks = dict(psnr_gt=number('tm_gt_psnr', infinite=True) >=
                      number('bf16_gt_psnr', infinite=True) - PSNR_MARGIN_DB,
                      mse_fp32=tm <= BF16_ERROR_FACTOR * bf16)
    else:
        raise ValueError('no registered DNN gate for ' + model)
    return dict(protocol='R15-DM-W1-BF16', passed=all(checks.values()), checks=checks,
                thresholds=dict(agreement_margin=AGREEMENT_MARGIN,
                                bf16_error_factor=BF16_ERROR_FACTOR,
                                psnr_margin_db=PSNR_MARGIN_DB))
