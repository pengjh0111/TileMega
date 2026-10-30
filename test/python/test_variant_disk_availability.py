from types import SimpleNamespace

from tilemega.build.variants import probe


def test_root_can_use_reserved_blocks(monkeypatch):
    monkeypatch.setattr(probe.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(probe.os, 'statvfs', lambda _: SimpleNamespace(f_bfree=4096, f_frsize=4096))
    monkeypatch.setattr(probe.shutil, 'disk_usage', lambda _: SimpleNamespace(free=0))
    assert probe.available_disk_bytes('/cache') == 16 * 1024**2


def test_unprivileged_compiler_cannot_use_reserved_blocks(monkeypatch):
    monkeypatch.setattr(probe.os, 'geteuid', lambda: 1000)
    monkeypatch.setattr(probe.shutil, 'disk_usage', lambda _: SimpleNamespace(free=4096))
    assert probe.available_disk_bytes('/cache') == 4096
