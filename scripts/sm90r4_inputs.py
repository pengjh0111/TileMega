"""Repository-owned model inputs: SM90D1 (original six) and SM90R4 (new models)."""
import lzma

from sm90d1_common import ROOT, read, sha

ROOTS = ('configs/serving/sm90d1/inputs', 'configs/serving/sm90r4/inputs')


def model_dir(model):
    for root in ROOTS:
        if (ROOT / root / model / 'config.json').is_file():
            return ROOT / root / model
    raise ValueError(f'no repository inputs for {model}')


def extract(model, phase, destination):
    """Decompress the packed export after checking both recorded hashes."""
    directory = model_dir(model)
    packed = directory / f'{phase}.export.json.xz'
    expected = read(directory.parent / 'manifest.json')[str(packed.relative_to(ROOT))]
    if sha(packed) != expected['compressed_sha256']:
        raise ValueError('repository export input changed')
    destination.write_bytes(lzma.decompress(packed.read_bytes()))
    if sha(destination) != expected['raw_sha256']:
        raise ValueError('decompressed export input changed')
    return destination
