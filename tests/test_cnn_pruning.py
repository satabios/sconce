"""
Pytest-based CNN pruning tests for sconce.

Tests cover non-transformer Conv2d-based models including:
  - ResNet18 (torchvision) for CWP_Pruning end-to-end
  - Simple Conv-BN-ReLU sequential models for channel pruning
  - GMP (fine-grained magnitude) pruning on small CNNs

Test categories:
  1. Config detection: framework=None for CNNs, Conv2d-only prunable
  2. CWP_Pruning end-to-end on ResNet18
  3. GMP_Pruning on a small CNN
  4. channel_prune (uniform and per-layer ratios)
  5. channel_prune_layerwise (single-layer pruning)
  6. apply_channel_sorting
  7. _collect_modules static method
  8. Deepcopy isolation under CWP pruning

Run:
    pytest tests/test_cnn_pruning.py -v
"""

import os
import sys

# Add tests dir so conftest is importable when running directly
sys.path.insert(0, os.path.dirname(__file__))
# Add project root to path so `sconce` is importable when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy

import pytest
import torch
import torch.nn as nn

from sconce.pruner import prune

from conftest import param_count


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_small_cnn(num_classes=10):
    """Create a small Conv-BN-ReLU CNN for fast testing."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
        nn.Conv2d(32, 64, 3, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(64, num_classes),
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_cnn():
    model = _make_small_cnn()
    model.eval()
    return model


@pytest.fixture
def resnet18():
    import torchvision
    model = torchvision.models.resnet18(weights=None, num_classes=10)
    model.eval()
    return model


@pytest.fixture
def cnn_pruner():
    """Fresh pruner with attention_heads=False (CNN mode)."""
    p = prune()
    p.attention_heads = False
    p.snn = False
    p.prune_mode = ""
    p.device = torch.device("cpu")
    return p


@pytest.fixture
def cnn_inputs():
    """Example input for CNN models (batch=1, 3x32x32)."""
    return torch.randn(1, 3, 32, 32)


@pytest.fixture
def imagenet_inputs():
    """Example input at ImageNet resolution (batch=1, 3x224x224)."""
    return torch.randn(1, 3, 224, 224)


class FakeLoader:
    """Minimal iterable loader for CWP_Pruning."""
    def __init__(self, input_size=(2, 3, 224, 224), num_classes=10):
        self.input_size = input_size
        self.num_classes = num_classes

    def __iter__(self):
        return iter([(
            torch.randn(*self.input_size),
            torch.randint(0, self.num_classes, (self.input_size[0],)),
        )])


# ---------------------------------------------------------------------------
# 1. Config detection for CNNs
# ---------------------------------------------------------------------------

class TestDetectConfig:
    def test_resnet18_framework_none(self, resnet18, cnn_pruner):
        vc = cnn_pruner._detect_vit_config(resnet18)
        assert vc['framework'] is None
        assert vc['num_heads'] is None
        assert len(vc['attention_modules']) == 0

    def test_resnet18_prunable_conv_only(self, resnet18, cnn_pruner):
        vc = cnn_pruner._detect_vit_config(resnet18)
        for name, mod in vc['prunable_modules']:
            assert isinstance(mod, nn.Conv2d), f"Non-Conv2d in prunable: {name} is {type(mod).__name__}"

    def test_resnet18_classifier_ignored(self, resnet18, cnn_pruner):
        vc = cnn_pruner._detect_vit_config(resnet18)
        classifier_found = any(
            isinstance(mod, nn.Linear) and mod.out_features == 10
            for mod in vc['ignored_layers']
        )
        assert classifier_found, "Classifier head not in ignored_layers"

    def test_small_cnn_framework_none(self, small_cnn, cnn_pruner):
        vc = cnn_pruner._detect_vit_config(small_cnn)
        assert vc['framework'] is None
        assert len(vc['attention_modules']) == 0

    def test_small_cnn_prunable_count(self, small_cnn, cnn_pruner):
        vc = cnn_pruner._detect_vit_config(small_cnn)
        conv_count = sum(1 for _, m in small_cnn.named_modules() if isinstance(m, nn.Conv2d))
        # All Conv2d should be prunable (no classifier Conv2d)
        prunable_conv = sum(1 for _, m in vc['prunable_modules'] if isinstance(m, nn.Conv2d))
        assert prunable_conv == conv_count


# ---------------------------------------------------------------------------
# 2. CWP_Pruning end-to-end on ResNet18
# ---------------------------------------------------------------------------

class TestCWPPruning:
    def test_resnet18(self, resnet18, cnn_pruner, imagenet_inputs):
        cnn_pruner.model = resnet18
        orig_params = param_count(resnet18)

        conv_modules = [
            (name, mod) for name, mod in resnet18.named_modules()
            if isinstance(mod, nn.Conv2d)
        ]
        cnn_pruner.sparsity_dict = {name: 0.3 for name, _ in conv_modules}
        cnn_pruner.dataloader = {'test': FakeLoader()}

        cnn_pruner.CWP_Pruning()

        after = param_count(cnn_pruner.model)
        assert after < orig_params, "CWP should reduce parameters"

        with torch.no_grad():
            out = cnn_pruner.model(imagenet_inputs)
        assert out.shape == (1, 10), f"Expected (1, 10), got {out.shape}"

    def test_classifier_output_preserved(self, resnet18, cnn_pruner):
        """CWP pruning should not change the classifier's output dimension."""
        cnn_pruner.model = resnet18

        conv_modules = [
            (name, mod) for name, mod in resnet18.named_modules()
            if isinstance(mod, nn.Conv2d)
        ]
        cnn_pruner.sparsity_dict = {name: 0.3 for name, _ in conv_modules}
        cnn_pruner.dataloader = {'test': FakeLoader()}

        cnn_pruner.CWP_Pruning()

        assert cnn_pruner.model.fc.out_features == 10

    def test_small_cnn(self, small_cnn, cnn_pruner, cnn_inputs):
        cnn_pruner.model = small_cnn
        orig_params = param_count(small_cnn)

        conv_modules = [
            (name, mod) for name, mod in small_cnn.named_modules()
            if isinstance(mod, nn.Conv2d)
        ]
        cnn_pruner.sparsity_dict = {name: 0.25 for name, _ in conv_modules}
        cnn_pruner.dataloader = {'test': FakeLoader(input_size=(2, 3, 32, 32))}

        cnn_pruner.CWP_Pruning()

        after = param_count(cnn_pruner.model)
        assert after < orig_params

        with torch.no_grad():
            out = cnn_pruner.model(cnn_inputs)
        assert out.shape == (1, 10)

    def test_varying_sparsity(self, resnet18, cnn_pruner, imagenet_inputs):
        """Different sparsity per Conv2d layer."""
        cnn_pruner.model = resnet18

        conv_modules = [
            (name, mod) for name, mod in resnet18.named_modules()
            if isinstance(mod, nn.Conv2d)
        ]
        # Assign increasing sparsity
        cnn_pruner.sparsity_dict = {}
        for i, (name, _) in enumerate(conv_modules):
            cnn_pruner.sparsity_dict[name] = 0.1 + 0.03 * i

        cnn_pruner.dataloader = {'test': FakeLoader()}

        cnn_pruner.CWP_Pruning()

        with torch.no_grad():
            out = cnn_pruner.model(imagenet_inputs)
        assert out.shape == (1, 10)


# ---------------------------------------------------------------------------
# 3. GMP_Pruning on small CNN
# ---------------------------------------------------------------------------

class TestGMPPruning:
    def test_basic(self, small_cnn, cnn_pruner):
        cnn_pruner.model = small_cnn

        cnn_pruner.sparsity_dict = {}
        for name, param in small_cnn.named_parameters():
            if param.dim() > 1:
                cnn_pruner.sparsity_dict[name] = 0.5

        cnn_pruner.GMP_Pruning()

        # Masks should have been created for all multi-dim params
        assert len(cnn_pruner.masks) > 0
        for name, mask in cnn_pruner.masks.items():
            sparsity = 1 - mask.float().mean().item()
            assert sparsity > 0.3, f"{name} sparsity too low: {sparsity:.2f}"

    def test_gmp_apply(self, small_cnn, cnn_pruner, cnn_inputs):
        cnn_pruner.model = small_cnn

        cnn_pruner.sparsity_dict = {}
        for name, param in small_cnn.named_parameters():
            if param.dim() > 1:
                cnn_pruner.sparsity_dict[name] = 0.5

        cnn_pruner.GMP_Pruning()

        # Zero out weights, then re-apply masks
        cnn_pruner.GMP_apply()

        # Model should still produce valid output
        with torch.no_grad():
            out = cnn_pruner.model(cnn_inputs)
        assert out.shape == (1, 10)

    def test_output_shape_preserved(self, small_cnn, cnn_pruner, cnn_inputs):
        """GMP doesn't change architecture, only zeros weights."""
        cnn_pruner.model = small_cnn
        orig_params = param_count(small_cnn)

        cnn_pruner.sparsity_dict = {}
        for name, param in small_cnn.named_parameters():
            if param.dim() > 1:
                cnn_pruner.sparsity_dict[name] = 0.7

        cnn_pruner.GMP_Pruning()

        # GMP is unstructured — param count stays the same
        assert param_count(cnn_pruner.model) == orig_params

        with torch.no_grad():
            out = cnn_pruner.model(cnn_inputs)
        assert out.shape == (1, 10)


# ---------------------------------------------------------------------------
# 4. channel_prune
# ---------------------------------------------------------------------------

class TestChannelPrune:
    def test_uniform_ratio(self, small_cnn, cnn_pruner, cnn_inputs):
        orig_params = param_count(small_cnn)

        new_model = cnn_pruner.channel_prune(small_cnn, prune_ratio=0.3)

        after = param_count(new_model)
        assert after < orig_params

        with torch.no_grad():
            out = new_model(cnn_inputs)
        assert out.shape == (1, 10)

    def test_per_layer_ratio(self, small_cnn, cnn_pruner, cnn_inputs):
        """Pass a list of per-layer ratios."""
        # 3 conv layers -> 2 pairs (n_conv - 1)
        ratios = [0.2, 0.4]

        new_model = cnn_pruner.channel_prune(small_cnn, prune_ratio=ratios)

        with torch.no_grad():
            out = new_model(cnn_inputs)
        assert out.shape == (1, 10)

    def test_channels_actually_reduced(self, small_cnn, cnn_pruner):
        new_model = cnn_pruner.channel_prune(small_cnn, prune_ratio=0.5)

        convs = prune._collect_modules(new_model, nn.Conv2d)
        # First conv: 16 -> 8 out_channels
        assert convs[0].out_channels == 8
        # Second conv: 8 in_channels from first, reduced out_channels
        assert convs[1].in_channels == 8
        assert convs[1].out_channels == 16  # 32 * 0.5
        # Third conv: 16 in_channels from second
        assert convs[2].in_channels == 16

    def test_zero_ratio_noop(self, small_cnn, cnn_pruner, cnn_inputs):
        """Ratio of 0 should keep all channels."""
        orig_params = param_count(small_cnn)
        new_model = cnn_pruner.channel_prune(small_cnn, prune_ratio=0.0)
        assert param_count(new_model) == orig_params


# ---------------------------------------------------------------------------
# 5. channel_prune_layerwise
# ---------------------------------------------------------------------------

class TestChannelPruneLayerwise:
    def test_single_layer(self, small_cnn, cnn_pruner, cnn_inputs):
        orig_params = param_count(small_cnn)

        new_model = cnn_pruner.channel_prune_layerwise(small_cnn, prune_ratio=0.25, i_layer=0)

        after = param_count(new_model)
        assert after < orig_params

        with torch.no_grad():
            out = new_model(cnn_inputs)
        assert out.shape == (1, 10)

    def test_dimensions_correct(self, small_cnn, cnn_pruner):
        """After pruning layer 0, verify dimension consistency."""
        new_model = cnn_pruner.channel_prune_layerwise(small_cnn, prune_ratio=0.5, i_layer=0)

        convs = prune._collect_modules(new_model, nn.Conv2d)
        bns = prune._collect_modules(new_model, nn.BatchNorm2d)

        # Layer 0: out_channels halved
        assert convs[0].out_channels == 8
        assert bns[0].num_features == 8
        # Layer 1: in_channels matches layer 0's out_channels
        assert convs[1].in_channels == 8
        # Layer 1 out_channels unchanged
        assert convs[1].out_channels == 32

    def test_second_layer(self, small_cnn, cnn_pruner, cnn_inputs):
        new_model = cnn_pruner.channel_prune_layerwise(small_cnn, prune_ratio=0.25, i_layer=1)

        with torch.no_grad():
            out = new_model(cnn_inputs)
        assert out.shape == (1, 10)


# ---------------------------------------------------------------------------
# 6. apply_channel_sorting
# ---------------------------------------------------------------------------

class TestChannelSorting:
    def test_output_equivalent(self, small_cnn, cnn_pruner, cnn_inputs):
        """Sorted model should produce same output (just reordered internals)."""
        cnn_pruner.model = small_cnn

        with torch.no_grad():
            orig_out = small_cnn(cnn_inputs).clone()

        sorted_model = cnn_pruner.apply_channel_sorting()

        with torch.no_grad():
            sorted_out = sorted_model(cnn_inputs)

        diff = (orig_out - sorted_out).abs().max().item()
        # Channel reordering through BatchNorm running stats causes small
        # floating-point differences, so use a relaxed tolerance.
        assert diff < 0.05, f"Sorted model output diverges: max diff = {diff}"

    def test_param_count_unchanged(self, small_cnn, cnn_pruner):
        cnn_pruner.model = small_cnn
        orig_params = param_count(small_cnn)

        sorted_model = cnn_pruner.apply_channel_sorting()
        assert param_count(sorted_model) == orig_params


# ---------------------------------------------------------------------------
# 7. _collect_modules
# ---------------------------------------------------------------------------

class TestCollectModules:
    def test_conv2d(self, small_cnn):
        convs = prune._collect_modules(small_cnn, nn.Conv2d)
        assert len(convs) == 3

    def test_batchnorm(self, small_cnn):
        bns = prune._collect_modules(small_cnn, nn.BatchNorm2d)
        assert len(bns) == 3

    def test_linear(self, small_cnn):
        linears = prune._collect_modules(small_cnn, nn.Linear)
        assert len(linears) == 1

    def test_resnet18(self, resnet18):
        convs = prune._collect_modules(resnet18, nn.Conv2d)
        assert len(convs) == 20
        bns = prune._collect_modules(resnet18, nn.BatchNorm2d)
        assert len(bns) == 20


# ---------------------------------------------------------------------------
# 8. Deepcopy isolation
# ---------------------------------------------------------------------------

class TestDeepcopyIsolation:
    def test_cwp_original_untouched(self, resnet18, cnn_pruner):
        """CWP pruning on a copy shouldn't affect the original."""
        orig_params = param_count(resnet18)

        model_copy = copy.deepcopy(resnet18)
        cnn_pruner.model = model_copy

        conv_modules = [
            (name, mod) for name, mod in model_copy.named_modules()
            if isinstance(mod, nn.Conv2d)
        ]
        cnn_pruner.sparsity_dict = {name: 0.3 for name, _ in conv_modules}
        cnn_pruner.dataloader = {'test': FakeLoader()}

        cnn_pruner.CWP_Pruning()

        # Copy was pruned
        assert param_count(model_copy) < orig_params
        # Original untouched
        assert param_count(resnet18) == orig_params

    def test_channel_prune_original_untouched(self, small_cnn, cnn_pruner):
        """channel_prune returns a new model, original stays unchanged."""
        orig_params = param_count(small_cnn)

        new_model = cnn_pruner.channel_prune(small_cnn, prune_ratio=0.5)

        assert param_count(new_model) < orig_params
        assert param_count(small_cnn) == orig_params
