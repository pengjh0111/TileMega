"""Freeze real technical prose/code into exact-length, no-thinking prompts."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--source-revision', default='ebe2fb0ed9d56b9b1e1abdd57c8fa8fd6a835f4c')
    args = parser.parse_args()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    paths = subprocess.check_output(['git', 'ls-tree', '-r', '--name-only', args.source_revision,
                                    'docs', 'lib', 'include', 'python'], cwd=ROOT, text=True).splitlines()
    paths = [path for path in paths if not path.startswith('docs/results/')
             and Path(path).suffix in ('.md', '.cpp', '.h', '.cuh', '.py')]
    sources, documents = [], []
    for path in paths:
        data = subprocess.check_output(['git', 'show', f'{args.source_revision}:{path}'], cwd=ROOT)
        if len(data) < 500:
            continue
        text = data.decode('utf-8')
        ids = tokenizer.encode(text, add_special_tokens=False)
        sources.append(dict(path=path, bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                            token_count=len(ids)))
        documents.append(ids)
    pool = [token for document in documents for token in document]
    if len(pool) < 64 * 8192:
        raise ValueError('real corpus too short for unique long input windows')
    marker = 'TILEMEGA_CORPUS_INSERTION_20261116'
    template = tokenizer.apply_chat_template([
        dict(role='user', content='Read the following technical documentation or source code and '
             'explain its behavior and performance implications.\n\n' + marker)],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)
    before, after = template.split(marker)
    prefix = tokenizer.encode(before, add_special_tokens=False)
    suffix = tokenizer.encode(after, add_special_tokens=False)
    inputs, windows = {}, {}
    for length in (128, 2048, 8192):
        body_length = length - len(prefix) - len(suffix)
        if body_length <= 0:
            raise ValueError('chat template exceeds fixed input length')
        starts = [i * ((len(pool) - body_length) // 64) for i in range(64)]
        rows = [prefix + pool[start:start + body_length] + suffix for start in starts]
        assert all(len(row) == length for row in rows)
        assert len({tuple(row) for row in rows}) == 64
        inputs[str(length)] = rows
        windows[str(length)] = [dict(start=start, tokens=body_length) for start in starts]
    value = dict(schema=1, inputs=inputs)
    payload = json.dumps(value, separators=(',', ':')).encode()
    args.out.mkdir(parents=True, exist_ok=True)
    target = args.out / 'inputs.json.gz'
    target.write_bytes(gzip.compress(payload, mtime=0))
    manifest = dict(model='Qwen/Qwen3-30B-A3B',
                    model_revision='ad44e777bcd18fa416d9da3bd8f70d33ebb85d39',
                    tokenizer_files={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                     for path in args.model.iterdir()
                                     if path.name in ('tokenizer.json', 'tokenizer_config.json',
                                                      'vocab.json', 'merges.txt')},
                    source_revision=args.source_revision, sources=sources,
                    corpus_tokens=len(pool), windows=windows,
                    input_file_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                    thinking=False, chat_prefix_ids=prefix, chat_suffix_ids=suffix,
                    scope='Real public technical prose/source code from pinned repository files. '
                          'Distinct contiguous corpus windows;no random/repeated filler. '
                          'May cross file boundaries;technical workload only,not a representative '
                          'general-user routing/quality corpus.')
    (args.out / 'inputs_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(dict(path=str(target), bytes=target.stat().st_size,
                          source_files=len(sources), corpus_tokens=len(pool))))


if __name__ == '__main__':
    main()
