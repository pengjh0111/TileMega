from pathlib import Path
import subprocess
root=Path(__file__).parents[3]
print('base\t'+subprocess.check_output(['git','rev-parse','origin/tilemega'],cwd=root,text=True).strip())
for f in ['TileMega_skeleton.md','docs/STATUS.md','docs/TODO.md','docs/archive/TODO_v2.0.md','docs/FINDINGS.md']:
 p=root/f; print(f+'\t'+str(p.stat().st_size))
