"""Freeze static weight recipes and allocation inputs for a native plan pair."""
import base64
import hashlib
import zlib

from tilemega.build.identity import digest

from .memory import allocation_metadata, memory_report
from .weights import prepare_weight_recipes


def binding(config, batch, capacity, identities, libraries):
    return dict(schema=1, config=digest(config), batch=batch, capacity=capacity,
                plans=[dict(binary=identity['binary_sha256'], manifest=identity['manifest_sha256'],
                            source=identity['source']['source_digest']) for identity in identities],
                abi=digest([[vars(buffer) for buffer in library.buffers] for library in libraries]))


def prepare(config, batch, manifests, identities, libraries):
    capacity = manifests[-1]['capacity']
    metadata = [allocation_metadata(manifest, library) for manifest, library in zip(manifests, libraries, strict=True)]
    return dict(binding=binding(config, batch, capacity, identities, libraries),
                allocation=memory_report(config, metadata, batch, capacity),
                weight_recipes=prepare_weight_recipes(*libraries))


def validate(record, config, batch, capacity, libraries):
    if record['binding'] != binding(config, batch, capacity, [lib.identity for lib in libraries], libraries):
        raise ValueError('frozen host preparation has a different plan, ABI or workload')


def encode_schedule(raw, binary_sha256):
    if len(raw) < 64 or not raw.startswith(b'TMSC\x01\x00\x00\x00') or len(raw) % 4:
        raise ValueError('unknown native schedule cache format')
    return dict(schema=1, prefill_sha256=binary_sha256, size=len(raw),
                sha256=hashlib.sha256(raw).hexdigest(),
                data=base64.b64encode(zlib.compress(raw)).decode('ascii'))


def prefill_schedule_bytes(record, library):
    snapshot = record.get('prefill_schedule')
    if snapshot is None:
        return None
    if (snapshot['schema'] != 1 or library is None or
            snapshot['prefill_sha256'] != library.identity['binary_sha256']):
        raise ValueError('prefill cache belongs to a different library or schema')
    size = snapshot['size']
    if not isinstance(size, int) or not 64 <= size <= 512*1024*1024 or size % 4:
        raise ValueError('invalid prefill cache size')
    decoder = zlib.decompressobj()
    raw = decoder.decompress(base64.b64decode(snapshot['data'], validate=True), size+1)
    if (len(raw) != size or not decoder.eof or decoder.unused_data or
            hashlib.sha256(raw).hexdigest() != snapshot['sha256']):
        raise ValueError('prefill schedule cache checksum mismatch')
    return raw
