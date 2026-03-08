"""Shared fixtures and helpers for ViT pruning tests."""

import pytest
import torch

from sconce.pruner import prune


def param_count(model):
    """Return total number of parameters in a model."""
    return sum(p.numel() for p in model.parameters())


def make_small_hf_vit(num_classes=None):
    """Create a small HuggingFace ViT for fast testing.

    Args:
        num_classes: If provided, returns ViTForImageClassification with this
            many output classes.  Otherwise returns a plain ViTModel.
    """
    from transformers import ViTConfig, ViTModel, ViTForImageClassification

    config = ViTConfig(
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
        intermediate_size=1536,
        image_size=224,
        patch_size=16,
    )

    if num_classes is not None:
        config.num_labels = num_classes
        return ViTForImageClassification(config)

    return ViTModel(config)


@pytest.fixture
def pruner():
    return prune()


@pytest.fixture
def example_inputs():
    return torch.randn(1, 3, 224, 224)
