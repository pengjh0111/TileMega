"""Expand the frozen export for the CPU native CTG integration check."""
import lzma
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(sys.argv[2])
with tempfile.TemporaryDirectory(prefix='tilemega-native-ctg-') as directory:
    graph = Path(directory) / 'decode.export.json'
    graph.write_bytes(lzma.decompress((root / 'configs/serving/sm90d1/inputs/'
                                       'qwen3_1_7b/decode.export.json.xz').read_bytes()))
    raise SystemExit(subprocess.call([sys.argv[1], str(root / 'configs/targets/sm_90_h800.json'),
                                      str(root), str(graph)]))
