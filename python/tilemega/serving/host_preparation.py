"""Freeze static weight recipes and allocation inputs for a native plan pair."""
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
