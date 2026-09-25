"""Fetch pinned serving checkpoints outside the repository and audit dimensions."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCES = {
    "llama": ["meta-llama/Llama-3.2-1B", "unsloth/Llama-3.2-1B"],
    "qwen3": ["Qwen/Qwen3-1.7B"],
}
CONFIG_SOURCES = {
    "llama": "docs/experiments/MODELS/sources/llama_config_public_copy.json",
    "qwen3": "docs/experiments/MODELS/sources/qwen_config.json",
}
DIMENSIONS = ("hidden_size", "intermediate_size", "num_hidden_layers",
              "num_attention_heads", "num_key_value_heads", "head_dim",
              "vocab_size", "rope_theta", "rope_scaling", "tie_word_embeddings")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def fetch(name: str, destination: Path, repo_root: Path) -> dict:
    from huggingface_hub import HfApi, snapshot_download
    errors = []
    for repo in SOURCES[name]:
        try:
            revision = HfApi().model_info(repo).sha
            snapshot_download(
                repo_id=repo, revision=revision, local_dir=destination,
                allow_patterns=["*.json", "*.safetensors", "tokenizer.model",
                                "*.tiktoken", "*.txt"],
            )
            break
        except Exception as exc:
            errors.append(f"{repo}: {type(exc).__name__}: {exc}")
    else:
        raise RuntimeError("all checkpoint sources failed: " + "; ".join(errors))
    actual = json.loads((destination / "config.json").read_text())
    expected = json.loads((repo_root / CONFIG_SOURCES[name]).read_text())
    differences = {key: {"checkpoint": actual.get(key), "recorded": expected.get(key)}
                   for key in DIMENSIONS if actual.get(key) != expected.get(key)}
    tensors = sorted(destination.glob("*.safetensors"))
    if not tensors:
        raise ValueError(f"no safetensors downloaded for {repo}")
    result = {"model": name, "source": repo, "revision": revision,
              "destination": str(destination), "config_differences": differences,
              "safetensors_sha256": {p.name: sha256(p) for p in tensors},
              "dimensions": {key: actual.get(key) for key in DIMENSIONS},
              "fallback_errors": errors}
    (destination / "tilemega_source_manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("llama", "qwen3", "all"), default="all")
    parser.add_argument("--model-root", type=Path, default=Path("/root/models"))
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--report", type=Path,
                        default=Path("docs/experiments/SERVING_R10/model_sources.json"))
    args = parser.parse_args()
    names = [args.model] if args.model != "all" else ["llama", "qwen3"]
    paths = {"llama": args.model_root / "llama3_2_1b",
             "qwen3": args.model_root / "qwen3_1_7b"}
    result = {}
    for name in names:
        paths[name].mkdir(parents=True, exist_ok=True)
        result[name] = fetch(name, paths[name], args.repo_root)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: {"source": v["source"], "revision": v["revision"],
                          "files": len(v["safetensors_sha256"]),
                          "differences": v["config_differences"]}
                      for k, v in result.items()}))


if __name__ == "__main__":
    main()
