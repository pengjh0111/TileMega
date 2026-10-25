"""CPU-only checks for actual device-code accounting and strict MX-5 limits."""
import struct
from pathlib import Path

import pytest

from tilemega.mx5_resources import device_elfs, inspect, kernel_instructions, text_sections


def elf_image(code=b'code'):
    names = b'\0.text\0.shstrtab\0'
    header = bytearray(64)
    header[:6] = b'\x7fELF\x02\x01'
    offset = len(header) + len(code) + len(names)
    struct.pack_into('<Q', header, 40, offset)
    struct.pack_into('<HHH', header, 58, 64, 3, 2)
    sections = bytes(64)
    sections += struct.pack('<IIQQQQIIQQ', 1, 1, 6, 0, 64, len(code), 0, 0, 16, 0)
    sections += struct.pack('<IIQQQQIIQQ', 7, 3, 0, 0, 64 + len(code), len(names), 0, 0, 1, 0)
    return bytes(header) + code + names + sections


def offload(elf):
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    target = b'maca-mxc-metaxgpu--xcore1002'
    offset = len(magic) + 8 + 24 + len(target)
    return b'host text bytes' + magic + struct.pack('<QQQQ', 1, offset, len(elf), len(target)) + target + elf


def log(mt=227, stack=0, shared=0):
    return f'''Function properties for _Zmx_decode_kernel_v2X
    {stack} bytes stack frame
Used {mt} MTregisters, 30 STregisters, {shared} bytes shared mem
# Machine code for function _Zmx_decode_kernel_v2X:
bb.0:
  successors: %bb.1
  renamable $r0 = LDG_B32 $r2, 0
  ARRIVE 4096
bb.1:
  MMA_16X16X16BF16 $r0, $r1
# End machine code for function _Zmx_decode_kernel_v2X.
'''


def test_counts_device_elf_text_separately_from_host_artifact(tmp_path):
    so, report = tmp_path / 'plan.so', tmp_path / 'plan.log'
    so.write_bytes(offload(elf_image(b'123456789')))
    report.write_text(log())
    result = inspect(so, report, 576)
    assert result['code_bytes'] == 9
    assert result['artifact_bytes'] == so.stat().st_size > 9
    assert result['kernel_instructions'] == {'_Zmx_decode_kernel_v2X': 3}
    host_stub = log().replace('_Zmx_decode_kernel_v2X', '_Z__device_stub__mx_decode_kernel_v2X')
    assert kernel_instructions(log() + host_stub) == {'_Zmx_decode_kernel_v2X': 3}
    assert result['violations'] == []
    assert all(value is None for value in result['body_instructions'].values())
    report.write_text(log(mt=228, stack=4, shared=16))
    assert len(inspect(so, report, 576)['violations']) == 3
    assert len(inspect(so, report, 512)['violations']) == 2
    report.write_text(log(mt=256))
    assert len(inspect(so, report, 512)['violations']) == 1


def test_rejects_missing_or_malformed_artifacts_and_mir():
    with pytest.raises(ValueError, match='offload'):
        device_elfs(b'host ELF only')
    with pytest.raises(ValueError, match='section table'):
        text_sections(elf_image()[:-1])
    with pytest.raises(ValueError, match='exactly one'):
        kernel_instructions('ordinary compiler messages')
    with pytest.raises(ValueError, match='duplicate'):
        kernel_instructions(log() + log())
    with pytest.raises(ValueError, match='incomplete'):
        kernel_instructions(log().split('# End machine code')[0])
