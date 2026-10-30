"""Capture V3 weights from its LDS-table diagnostic export, without global metadata."""
import pytest
from tilemega.moe.native_routing_check import routing_weight_dtype


def test_v2_and_v3_route_weight_sources_are_explicit():
    assert routing_weight_dtype(dict(buffers=[dict(name='mx.routing_weights',dtype='bf16')]))=='bf16'
    manifest=dict(program_version=3,threads=512,route_table_bytes=2576,
                  buffers=[dict(name='mx.route_weights',dtype='bf16')])
    assert routing_weight_dtype(manifest)=='bf16'
    manifest['route_table_bytes']=2832
    with pytest.raises(ValueError,match='BF16 LDS'):
        routing_weight_dtype(manifest)


def test_missing_diagnostic_weights_cannot_claim_a_contract():
    with pytest.raises(ValueError,match='absent'):
        routing_weight_dtype(dict(program_version=3,threads=512,route_table_bytes=2576,buffers=[]))
