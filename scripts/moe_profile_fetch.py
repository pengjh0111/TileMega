"""Download and verify pinned real weights without changing any system setting."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import time

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(16 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--transport', choices=['huggingface', 'modelscope'], default='huggingface')
    parser.add_argument('--direct', action='store_true', help='Use direct network for large weight transfers only')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    import requests
    plan = json.loads((ROOT / 'configs/moe_profile/workload.json').read_text())['model']
    repo, revision = plan['repo'], plan['revision']
    args.target.mkdir(parents=True, exist_ok=True)
    args.out.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    response = session.get(f'https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true', timeout=30)
    response.raise_for_status()
    tree = response.json()
    (args.out / 'tree.json').write_text(json.dumps(tree, indent=2) + '\n')
    wanted = [row for row in tree if row['path'].endswith('.safetensors') or row['path'] in (
        'config.json', 'generation_config.json', 'model.safetensors.index.json',
        'tokenizer.json', 'tokenizer_config.json', 'vocab.json', 'merges.txt', 'LICENSE')]
    weight_bytes = sum(row['size'] for row in wanted if row['path'].endswith('.safetensors'))
    missing = sum(row['size'] for row in wanted if not (args.target / row['path']).exists())
    if shutil.disk_usage(args.target).free < missing + 10 * 1024**3:
        raise ValueError('not enough target storage plus 10GiB headroom')
    mirror = None
    if args.transport == 'modelscope':
        response = session.get(f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true', timeout=30)
        response.raise_for_status()
        mirror = {row['Name']: row for row in response.json()['Data']['Files']}
        for row in wanted:
            if row['path'].endswith('.safetensors'):
                alternate = mirror[row['path']]
                if alternate['Size'] != row['size'] or alternate['Sha256'] != row['lfs']['oid']:
                    raise ValueError('mirror differs from pinned HF checkpoint')

    def download(row):
        name = row['path']
        path = args.target / name
        part = args.target / (name + '.download.part')
        source = (f'https://modelscope.cn/models/{repo}/resolve/master/{name}'
                  if mirror and name.endswith('.safetensors')
                  else f'https://huggingface.co/{repo}/resolve/{revision}/{name}')
        local = requests.Session()
        local.trust_env = not (args.direct and name.endswith('.safetensors'))
        if not path.exists():
            for attempt in range(10):
                try:
                    offset = part.stat().st_size if part.exists() else 0
                    if offset == row['size']:
                        break
                    if offset > row['size']:
                        raise ValueError('partial file longer than pinned source')
                    with local.get(source, headers={'Range': f'bytes={offset}-'},
                                   stream=True, timeout=(15, 60)) as result:
                        result.raise_for_status()
                        if result.status_code == 206:
                            if result.headers.get('Content-Range') != f"bytes {offset}-{row['size']-1}/{row['size']}":
                                raise ValueError('server returned a different byte range')
                        elif result.status_code != 200 or offset:
                            raise ValueError('server did not honor resume range')
                        with part.open('ab') as stream:
                            for data in result.iter_content(4 * 1024**2):
                                stream.write(data)
                    if part.stat().st_size == row['size']:
                        break
                except (requests.RequestException, OSError):
                    time.sleep(min(10, attempt + 1))
            if not part.exists() or part.stat().st_size != row['size']:
                raise ValueError(f'incomplete download: {name}')
            part.rename(path)
        if path.stat().st_size != row['size']:
            raise ValueError(f'wrong byte size: {name}')
        digest = sha_file(path)
        if 'lfs' in row:
            valid = digest == row['lfs']['oid']
        else:
            data = path.read_bytes()
            valid = hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == row['oid']
        if not valid:
            raise ValueError(f'pinned checksum mismatch: {name}')
        print(json.dumps(dict(file=name, status='verified', bytes=row['size'])), flush=True)
        return dict(path=name, bytes=row['size'], sha256=digest)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        files = list(pool.map(download, wanted))
    weights = [row for row in files if row['path'].endswith('.safetensors')]
    if len(weights) != 16 or sum(row['bytes'] for row in weights) != weight_bytes:
        raise ValueError('unexpected checkpoint shard layout')
    result = dict(passed=True, repo=repo, revision=revision, model_path=str(args.target.resolve()),
                  weights=weights, files=files, transport=args.transport, direct=args.direct,
                  finished=time.time(), scope='Real pretrained checkpoint;every shard LFS SHA256 verified.')
    (args.out / 'download_result.json').write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
