"""Exercise the rounded native MoE graph using a repository-owned export."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

root = Path(sys.argv[2])
sys.path.insert(0, str(root / 'scripts'))
from sm90r4_inputs import extract

with tempfile.TemporaryDirectory(prefix='tilemega-native-moe-ctg-') as temporary:
    directory = Path(temporary)
    graph = extract('qwen3_30b_a3b', 'decode', directory / 'decode.export.json')
    config = json.loads((root / 'configs/serving/sm90r5/ref_r4/qwen3_30b_a3b_b16.json').read_text())
    coordinates = directory / 'coordinates.json'
    coordinates.write_text(json.dumps(config.get('coordinates', config)))
    raise SystemExit(subprocess.call([sys.argv[1], str(root / 'configs/targets/sm_90_h800.json'),
        str(root), str(graph), str(coordinates), '16']))
