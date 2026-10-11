"""Fixed-geometry placeholder; accepts compiler arguments without using CUDA."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args, ignored = parser.parse_known_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'measurements.json').write_text(json.dumps({
        'modes': {'L1': {'mean_ms': 0}, 'L2': {'mean_ms': 0}},
        'placeholder': True, 'ignored': ignored}) + '\n')
    print('fixed-geometry placeholder (no GPU measurement)')


if __name__ == '__main__':
    main()
