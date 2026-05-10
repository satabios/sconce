"""
sconce/transformer_pruner.py
Backward-compatibility shim.

All transformer pruning code now lives in sconce/pruner.py inside
:class:`TransformerPruner`.  This module re-exports the public API so
that existing imports continue to work:

    from sconce.transformer_pruner import TransformerPruner
    from sconce.transformer_pruner import find_transformer_layers
    from sconce.transformer_pruner import transformer_sensitivity_scan
    from sconce.transformer_pruner import transformer_structured_prune
    from sconce.transformer_pruner import TransformerLayerSpec, AttentionSpec, FFNSpec
"""
from .pruner import (  # noqa: F401
    TransformerPruner,
    find_transformer_layers,
    transformer_sensitivity_scan,
    transformer_structured_prune,
    TransformerLayerSpec,
    AttentionSpec,
    FFNSpec,
)
