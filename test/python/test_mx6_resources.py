"""Keep inlined code attribution conservative and tied to the measured image."""
import pytest

from tilemega.mx6_resources import dwarf_file_path, partition_ranges


def test_overlapping_different_stages_are_shared_not_double_counted():
    result=partition_ranges(64,[('qkv',8,32),('qkv',16,24),('attention',24,48)])
    assert result==dict(control=24,qkv=16,shared=8,attention=16)
    assert sum(result.values())==64


@pytest.mark.parametrize('lo,hi',[(-8,8),(8,72),(24,16),(8,8)])
def test_bad_pc_range_is_rejected(lo,hi):
    with pytest.raises(ValueError,match='outside'):
        partition_ranges(64,[('qkv',lo,hi)])


def test_unattributed_alignment_stays_control():
    assert partition_ranges(32,[])==dict(control=32)


def test_archived_trace_file_uses_dwarf_directory(tmp_path):
    root=tmp_path/'archive';source=root/'cp1/model/B1/plan.so.cu'
    assert dwarf_file_path('cp1/model/B1/plan.so.cu',str(root),str(root/'cp1_source'))==source
    assert dwarf_file_path('plan.so.cu',str(source.parent),str(root/'cp1_source'))==source
    assert dwarf_file_path(str(source),'ignored',str(root))==source
    assert dwarf_file_path('plan.so.cu','other',str(root))!=source
