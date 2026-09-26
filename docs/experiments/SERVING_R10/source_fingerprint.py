#!/usr/bin/env python3
"""Compatibility import for archived R10 experiment scripts."""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'python'))
from tilemega.fingerprint import source_fingerprint

if __name__ == '__main__':
    print(source_fingerprint(ROOT))
