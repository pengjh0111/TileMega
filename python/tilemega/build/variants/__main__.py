"""python -m tilemega.build.variants {probe,prewarm} [options]."""
import sys
from . import probe, prewarm
if len(sys.argv) < 2 or sys.argv[1] not in {'probe', 'prewarm'}:
    raise SystemExit('usage: python -m tilemega.build.variants {probe,prewarm} [options]')
command = sys.argv.pop(1)
raise SystemExit((probe if command == 'probe' else prewarm).main())
