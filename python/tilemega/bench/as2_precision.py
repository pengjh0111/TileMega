"""Bind declared precision to the actual attention implementation tables."""
REPLAY = 'as2_llm_fp32_replay_bf16_casts_v1'
STREAMING = 'bf16_qkv_fp32_online_statistics_bf16_probability_v1'
STREAMING_FP32 = 'bf16_qkv_split_scale_fp32_online_statistics_probability_v1'
CONTRACTS = {REPLAY: 1, STREAMING: 2, STREAMING_FP32: 3}


def validate_plan_precision(precision, stages):
    if precision not in CONTRACTS:
        raise ValueError('Unsupported phase precision contract')
    attention = [s for s in stages if s[17] == 13]
    streaming = precision in (STREAMING, STREAMING_FP32)
    if streaming and not attention:
        raise ValueError('Streaming precision requires an actual attention domain')
    for s in attention:
        if not 0 <= s[31] <= 13 or (s[31] in (10,11)) != streaming or (streaming and s[31] != CONTRACTS[precision]+8):
            raise ValueError('Attention table and precision contract differ')
        if s[31]==13 and (s[19]!=8 or s[23]!=8 or s[24]!=128 or s[25]!=2 or s[26]!=1088 or s[27]!=2):
            raise ValueError('Unregistered Qwen0.6 QK phase drain shape')
        if s[31]==12 and (s[24]!=128 or s[26]!=1088):
            raise ValueError('Valid-extent PV currently admits D128 only')
        if streaming and (s[26] != 1088 or not s[23] or not s[25] or not s[27] or s[27] % s[25] or
                (s[24], s[25]) not in {(128, 2), (128, 3), (128, 4), (64, 4)}):
            raise ValueError('Unregistered streaming precision shape')
    return CONTRACTS[precision]
