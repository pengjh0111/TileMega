"""Isolate and fingerprint the MPK Ampere argmax row-barrier experiment."""
import hashlib
from pathlib import Path
import shutil
import tempfile

HEADER = Path('include/mirage/persistent_kernel/tasks/ampere/argmax.cuh')
ORIGINAL = '2eb7134795b09ecc9a9fb80842543704ba686870ea9f4e7c52d51e86f2eee232'
PATCHED = 'c89b4122248f437ab723cfe09efcb3589b1ce29cab86d38a7380b8356866ea74'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare_row_overlay(source, destination):
    if digest(source/HEADER) != ORIGINAL:
        raise ValueError('unexpected MPK Ampere argmax source')
    if not destination.exists():
        with tempfile.TemporaryDirectory(dir=destination.parent) as scratch:
            root = Path(scratch)/'overlay'
            # Copy the header tree so quoted includes in the runtime resolve
            # inside this cohort. All original source and demo files stay intact.
            shutil.copytree(source/'include', root/'include')
            (root/'deps').symlink_to((source/'deps').resolve(), target_is_directory=True)
            header = root/HEADER
            before, loops = header.read_text().split('    argmax_partial_kernel(', 1)
            end = '    }\n  }\n}'
            if loops.count(end) != 2:
                raise ValueError('unexpected MPK row loop shape')
            fence = ('    }\n    if constexpr (BATCH_SIZE > 1) {\n'
                     '      if (batch_idx + 1 < num_active_tokens) __syncthreads();\n'
                     '    }\n  }\n}')
            header.write_text(before + '    argmax_partial_kernel(' + loops.replace(end, fence))
            if digest(header) != PATCHED:
                raise ValueError('patched MPK header fingerprint differs')
            root.rename(destination)
    if digest(destination/HEADER) != PATCHED or (destination/'deps').resolve() != (source/'deps').resolve():
        raise ValueError('cohort MPK include root changed')
    return {'original_sha256': ORIGINAL, 'compiled_sha256': PATCHED, 'root': str(destination)}
