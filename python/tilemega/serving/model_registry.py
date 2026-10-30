"""Pinned model metadata shared by downloads and serving configuration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

ALIASES = {'llama': 'llama3_2_1b', 'qwen3': 'qwen3_1_7b',
           'qwen3_moe': 'qwen3_30b_a3b'}
ROOT = Path(__file__).resolve().parents[3]
DIMENSIONS = ('hidden_size', 'intermediate_size', 'num_hidden_layers',
              'num_attention_heads', 'num_key_value_heads', 'head_dim',
              'vocab_size', 'rope_theta', 'rope_scaling', 'num_experts',
              'num_experts_per_tok', 'moe_intermediate_size', 'norm_topk_prob',
              'decoder_sparse_step', 'mlp_only_layers')


def model_ids(repo_root: Path = ROOT) -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in
                        (Path(repo_root)/'configs/models').glob('*.json')))


def dimensions(config: dict) -> dict:
    result = {key: config.get(key) for key in DIMENSIONS}
    result['head_dim'] = config.get('head_dim',
                                    config['hidden_size']//config['num_attention_heads'])
    heads, kv_heads = config['num_attention_heads'], config['num_key_value_heads']
    if heads <= 0 or kv_heads <= 0 or heads % kv_heads or not 1 <= heads//kv_heads <= 8:
        raise ValueError('model requires an integral QPerKV in [1, 8]')
    result['qperkv'] = heads//kv_heads
    return result


def load_model(name: str, repo_root: Path = ROOT) -> dict:
    name = ALIASES.get(name, name)
    if not re.fullmatch(r'[a-z0-9_]+', name):
        raise ValueError(f'invalid model id {name!r}')
    path = Path(repo_root)/'configs/models'/f'{name}.json'
    if not path.is_file():
        raise ValueError(f'unknown model {name!r}')
    record = json.loads(path.read_text())
    if record.get('id') != name or not record.get('repo'):
        raise ValueError(f'invalid model registry record {path}')
    if not re.fullmatch('[0-9a-f]{40}', record.get('revision', '')):
        raise ValueError('model revision must be a fixed 40-character commit')
    if not re.fullmatch('[0-9a-f]{64}', record.get('config_sha256', '')):
        raise ValueError('model config must have a SHA256')
    if record.get('tokenizer_family') not in ('llama3', 'qwen', 'mistral', 'olmoe'):
        raise ValueError('unsupported tokenizer family')
    if not isinstance(record.get('tie_word_embeddings'), bool):
        raise ValueError('tie_word_embeddings must be explicit')
    if not isinstance(record.get('dimensions'), dict):
        raise ValueError('model dimensions must be explicit')
    return record


def verify_config(record: dict, path: Path) -> dict:
    raw = Path(path).read_bytes()
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != record['config_sha256']:
        raise ValueError(f'config SHA256 mismatch for {record["id"]}: {actual_sha}')
    config = json.loads(raw)
    if config.get('tie_word_embeddings', False) != record['tie_word_embeddings']:
        raise ValueError('config tie_word_embeddings differs from registry')
    if dimensions(config) != record['dimensions']:
        raise ValueError('config dimensions differ from registry')
    return config
