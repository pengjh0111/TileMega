"""Upstream DNN export and forward-plan tools."""


def load_program(path):
    # HF's ModelOutput pytree types must be registered before deserialization;
    # importing their original definitions preserves BERT's output structure.
    import transformers.modeling_outputs  # noqa: F401
    import torch
    return torch.export.load(path)
