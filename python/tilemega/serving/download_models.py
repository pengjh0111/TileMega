"""Fetch pinned configs or serving checkpoints and audit their dimensions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from .model_registry import ALIASES, dimensions, load_model, model_ids, verify_config


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, destination: Path, repo_root: Path, *, config_only: bool = False) -> dict:
    record = load_model(name, repo_root)
    repo, revision = record['repo'], record['revision']
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    # Reject a mismatched config before downloading any large weight shards.
    config_path = destination/'config.json'
    if not (config_only and config_path.is_file()):
        from huggingface_hub import hf_hub_download
        config_path = Path(hf_hub_download(repo_id=repo, revision=revision,
                                          filename='config.json', local_dir=destination))
    actual = verify_config(record, config_path)
    result = {"model": record['id'], "source": repo, "revision": revision,
              "destination": str(destination), "config_differences": {},
              "config_sha256": record['config_sha256'],
              "tokenizer_family": record['tokenizer_family'],
              "tie_word_embeddings": record['tie_word_embeddings'],
              "dimensions": dimensions(actual)}
    if config_only:
        # Keep structural evidence separate from a verified weight manifest.
        result.update(structure_only=True, safetensors_sha256={})
        (destination/'tilemega_config_manifest.json').write_text(json.dumps(result, indent=2)+'\n')
        return result
    from huggingface_hub import snapshot_download
    snapshot_download(
        repo_id=repo, revision=revision, local_dir=destination,
        allow_patterns=['*.json', '*.safetensors', 'tokenizer.model',
                        '*.tiktoken', '*.txt'],
    )
    actual = verify_config(record, destination/'config.json')
    tensors = sorted(destination.glob("*.safetensors"))
    if not tensors:
        raise ValueError(f"no safetensors downloaded for {repo}")
    result.update(safetensors_sha256={p.name: sha256(p) for p in tensors},
                  dimensions=dimensions(actual))
    (destination / "tilemega_source_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main(argv=None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='all',
                        help='registered model id, llama/qwen3/qwen3_moe alias, or all')
    parser.add_argument("--model-root", type=Path, default=Path("/root/models"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--report", type=Path,
                        default=Path('runs/r15/model_sources.json'))
    parser.add_argument('--config-only', action='store_true',
                        help='verify only config.json; reuse valid local configs and never fetch weights')
    args = parser.parse_args(argv)
    names = [ALIASES.get(args.model, args.model)] if args.model != 'all' else model_ids(args.repo_root)
    result = {}
    for name in names:
        result[name] = fetch(name, args.model_root/name, args.repo_root,
                             config_only=args.config_only)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {"source": v["source"], "revision": v["revision"],
                          "files": len(v["safetensors_sha256"]),
                          "structure_only": v.get('structure_only', False),
                          "differences": v["config_differences"]}
                      for k, v in result.items()}))


if __name__ == "__main__":
    main()
