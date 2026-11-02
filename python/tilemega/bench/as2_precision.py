"""Bind declared precision to the actual attention implementation tables."""
REPLAY = 'as2_llm_fp32_replay_bf16_casts_v1'
STREAMING = 'bf16_qkv_fp32_online_statistics_bf16_probability_v1'
CONTRACTS = {REPLAY: 1, STREAMING: 2}


def validate_plan_precision(precision, stages):
    if precision not in CONTRACTS:
        raise ValueError('Unsupported phase precision contract')
    attention = [s for s in stages if s[17] == 13]
    streaming = precision == STREAMING
    if streaming and not attention:
        raise ValueError('Streaming precision requires an actual attention domain')
    for s in attention:
        if not 0 <= s[31] <= 10 or (s[31] == 10) != streaming:
            raise ValueError('Attention table and precision contract differ')
        if streaming and (s[26] != 1088 or not s[23] or not s[25] or not s[27] or s[27] % s[25] or
                (s[24], s[25]) not in {(128, 2), (128, 3), (128, 4), (64, 4)}):
            raise ValueError('Unregistered streaming precision shape')
    return CONTRACTS[precision]
