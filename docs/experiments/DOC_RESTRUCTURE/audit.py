from pathlib import Path
import hashlib, subprocess
root=Path(__file__).parents[2]
print('base',subprocess.check_output(['git','rev-parse','origin/tilemega'],cwd=root,text=True).strip())
for f in ['TileMega_skeleton.md','docs/STATUS.md','docs/TODO.md','docs/archive/TODO_v2.0.md','docs/FINDINGS.md']:
 p=root/f; print(f, p.stat().st_size)
