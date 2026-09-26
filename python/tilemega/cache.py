"""Content keys and atomic records shared by the end-to-end orchestration."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def key(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def export_key(model_config: Path, exporter: Path, phase: str, seq: int, capacity: int) -> str:
    return key(dict(config=file_sha(model_config), exporter=file_sha(exporter),
                    phase=phase, seq=seq, capacity=capacity))


def plan_key(export: str, stamps: dict, source: str, solver: dict, features: dict,
             batch: int, past_range: tuple[int, int]) -> str:
    return key(dict(export=export, target_stamps=stamps, source=source,
                    solver=solver, features=features, batch=batch, past_range=past_range))


def atomic_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def locked(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        yield


def valid_record(path: Path) -> bool:
    """A marker alone is insufficient: every recorded output must still match."""
    try:
        data = json.loads(path.read_text())
        return bool(data['outputs']) and all(
            Path(p).is_file() and file_sha(Path(p)) == sha
            for p, sha in data['outputs'].items())
    except (OSError, KeyError, ValueError):
        return False


def record_outputs(path: Path, outputs: list[Path], **metadata) -> None:
    atomic_json(path, dict(metadata, outputs={str(p.resolve()): file_sha(p) for p in outputs}))
