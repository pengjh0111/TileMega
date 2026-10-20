"""AS-2 entry point for the shared, no-lock single-device job runner."""
from tilemega.bench.as1_run import main

if __name__ == '__main__':
    raise SystemExit(main(project='as2'))
