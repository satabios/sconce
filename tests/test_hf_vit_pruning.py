"""
Pytest-based HuggingFace ViT pruning tests for sconce.

Tests cover HuggingFace Vision Transformer variants including:
  - ViTModel / ViTForImageClassification
  - DeiTModel (Data-efficient Image Transformer)
  - BeitModel (Bidirectional Encoder from Image Transformers)

Test categories:
  1. Config detection across variants
  2. Separate Q/K/V head importance computation
  3. DG-based head pruning standalone
  4. MLP-only MetaPruner pruning
  5. Two-path combined (MLP + heads) pruning
  6. CWP_Pruning end-to-end simulation
  7. Embed_dim preservation after pruning
  8. Per-layer head pruning with varying sparsities
  9. Edge cases: single head prune, max sparsity clamp
  10. Deepcopy isolation

Run:
    pytest tests/test_hf_vit_pruning.py -v
"""

import os
import sys

# Add tests dir so conftest is importable when running directly
sys.path.insert(0, os.path.dirname(__file__))
# Add project root to path so `sconce` is importable when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_pruning as tp

from sconce.pruner import prune

from conftest import param_count, make_small_hf_vit

try:
    from transformers import (
        ViTModel, ViTConfig, ViTForImageClassification,
        DeiTModel, DeiTConfig,
        BeitModel, BeitConfig,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

pytestmark = pytest.mark.skipif(not HF_AVAILABLE, reason="transformers not installed")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_small_deit():
    """Create a small DeiT for fast testing."""
    config = DeiTConfig(
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
        intermediate_size=1536,
        image_size=224,
        patch_size=16,
    )
    return DeiTModel(config)


def _make_small_beit():
    """Create a small BEiT for fast testing."""
    config = BeitConfig(
        hidden_size=384,
        num_hidden_layers=4,
        num_attention_heads=6,
        intermediate_size=1536,
        image_size=224,
        patch_size=16,
    )
    return BeitModel(config)


@pytest.fixture
def hf_vit():
    model = make_small_hf_vit()
    model.eval()
    return model


@pytest.fixture
def hf_vit_classifier():
    model = make_small_hf_vit(num_classes=10)
    model.eval()
    return model


@pytest.fixture
def hf_deit():
    model = _make_small_deit()
    model.eval()
    return model


@pytest.fixture
def hf_beit():
    model = _make_small_beit()
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. Config detection across HF variants
# ---------------------------------------------------------------------------

class TestDetectConfig:
    def test_vit_model(self, hf_vit, pruner):
        vc = pruner._detect_vit_config(hf_vit)

        assert vc['framework'] == 'huggingface'

        # prunable_modules: only intermediate.dense (MLP fc1)
        assert len(vc['prunable_modules']) == 4
        for name, mod in vc['prunable_modules']:
            assert 'intermediate.dense' in name
            assert isinstance(mod, nn.Linear)
            assert mod.in_features == 384
            assert mod.out_features == 1536

        # attention_modules: 4 ViTSelfAttention modules
        assert len(vc['attention_modules']) == 4
        for name, mod in vc['attention_modules']:
            assert hasattr(mod, 'query')
            assert hasattr(mod, 'key')
            assert hasattr(mod, 'value')
            assert mod.num_attention_heads == 6

        # num_heads: maps query, key, value for each layer = 4*3 = 12
        assert vc['num_heads'] is not None
        assert len(vc['num_heads']) == 12

        # unwrapped_parameters
        assert vc['unwrapped_parameters'] is not None

        # Q/K/V/output.dense excluded from prunable
        prunable_names = {n for n, _ in vc['prunable_modules']}
        for n, _ in hf_vit.named_modules():
            if n.endswith('.query') or n.endswith('.key') or n.endswith('.value'):
                assert n not in prunable_names
            if n.endswith('.output.dense'):
                assert n not in prunable_names

    def test_vit_for_classification(self, hf_vit_classifier, pruner):
        vc = pruner._detect_vit_config(hf_vit_classifier)

        assert vc['framework'] == 'huggingface'

        # Classifier in ignored_layers
        assert len(vc['ignored_layers']) >= 1
        classifier_found = any(
            isinstance(mod, nn.Linear) and mod.out_features == 10
            for mod in vc['ignored_layers']
        )
        assert classifier_found

        # Classifier NOT in prunable_modules
        for name, mod in vc['prunable_modules']:
            assert mod.out_features != 10

        assert len(vc['prunable_modules']) == 4
        assert len(vc['attention_modules']) == 4

    def test_deit_model(self, hf_deit, pruner):
        """DeiT shares HF ViT architecture — should detect as huggingface."""
        vc = pruner._detect_vit_config(hf_deit)

        assert vc['framework'] == 'huggingface'
        assert len(vc['prunable_modules']) == 4
        assert len(vc['attention_modules']) == 4

        for name, mod in vc['prunable_modules']:
            assert 'intermediate.dense' in name
            assert mod.in_features == 384
            assert mod.out_features == 1536

        for name, mod in vc['attention_modules']:
            assert hasattr(mod, 'query')
            assert mod.num_attention_heads == 6

        assert vc['num_heads'] is not None
        assert len(vc['num_heads']) == 12

    def test_beit_model(self, hf_beit, pruner):
        """BEiT shares HF ViT architecture — should detect as huggingface."""
        vc = pruner._detect_vit_config(hf_beit)

        assert vc['framework'] == 'huggingface'
        assert len(vc['prunable_modules']) == 4
        assert len(vc['attention_modules']) == 4

        for name, mod in vc['prunable_modules']:
            assert 'intermediate.dense' in name
            assert mod.in_features == 384
            assert mod.out_features == 1536

        for name, mod in vc['attention_modules']:
            assert hasattr(mod, 'query')
            assert mod.num_attention_heads == 6

        assert vc['num_heads'] is not None
        assert len(vc['num_heads']) == 12


# ---------------------------------------------------------------------------
# 2. Head importance — separate Q/K/V
# ---------------------------------------------------------------------------

class TestHeadImportance:
    def test_separate_qkv(self, pruner):
        num_heads, head_dim, embed_dim = 6, 64, 384
        q_w = torch.randn(num_heads * head_dim, embed_dim)
        k_w = torch.randn(num_heads * head_dim, embed_dim)
        v_w = torch.randn(num_heads * head_dim, embed_dim)

        importance = pruner.get_attention_head_importance([q_w, k_w, v_w], num_heads)
        assert importance.shape == (num_heads,)
        assert torch.all(importance >= 0)

    def test_zeroed_head_detected(self, pruner):
        """Zeroing a head's weights in Q/K/V makes it least important."""
        num_heads, head_dim, embed_dim = 6, 64, 384
        q_w = torch.randn(num_heads * head_dim, embed_dim)
        k_w = torch.randn(num_heads * head_dim, embed_dim)
        v_w = torch.randn(num_heads * head_dim, embed_dim)

        target_head = 2
        for w in [q_w, k_w, v_w]:
            w[target_head * head_dim:(target_head + 1) * head_dim, :] = 0

        importance = pruner.get_attention_head_importance([q_w, k_w, v_w], num_heads)
        assert torch.argmin(importance).item() == target_head


# ---------------------------------------------------------------------------
# 3. DG head pruning standalone
# ---------------------------------------------------------------------------

class TestDGHeadPruning:
    def test_vit(self, hf_vit, pruner, example_inputs):
        orig_params = param_count(hf_vit)
        vc = pruner._detect_vit_config(hf_vit)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                hf_vit, example_inputs, attn_mod, aname,
                sparsity=2/6, framework='huggingface', vit_config=vc,
            )

        after = param_count(hf_vit)
        assert after < orig_params

        with torch.no_grad():
            out = hf_vit(example_inputs)
        assert out.last_hidden_state.shape[0] == 1
        assert out.last_hidden_state.shape[2] == 384

        for _, mod in hf_vit.named_modules():
            if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
                assert mod.num_attention_heads == 4
                assert mod.query.out_features == 4 * 64
                assert mod.key.out_features == 4 * 64
                assert mod.value.out_features == 4 * 64

    def test_deit(self, hf_deit, pruner, example_inputs):
        orig_params = param_count(hf_deit)
        vc = pruner._detect_vit_config(hf_deit)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                hf_deit, example_inputs, attn_mod, aname,
                sparsity=2/6, framework='huggingface', vit_config=vc,
            )

        after = param_count(hf_deit)
        assert after < orig_params

        with torch.no_grad():
            out = hf_deit(example_inputs)
        assert out.last_hidden_state.shape[2] == 384

        for _, mod in hf_deit.named_modules():
            if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
                assert mod.num_attention_heads == 4
                break

    def test_beit(self, hf_beit, pruner, example_inputs):
        orig_params = param_count(hf_beit)
        vc = pruner._detect_vit_config(hf_beit)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                hf_beit, example_inputs, attn_mod, aname,
                sparsity=2/6, framework='huggingface', vit_config=vc,
            )

        after = param_count(hf_beit)
        assert after < orig_params

        with torch.no_grad():
            out = hf_beit(example_inputs)
        assert out.last_hidden_state.shape[2] == 384

        for _, mod in hf_beit.named_modules():
            if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
                assert mod.num_attention_heads == 4
                break


# ---------------------------------------------------------------------------
# 4. MLP-only MetaPruner
# ---------------------------------------------------------------------------

class TestMLPOnlyPruning:
    def test_vit(self, hf_vit, pruner, example_inputs):
        vc = pruner._detect_vit_config(hf_vit)
        orig_params = param_count(hf_vit)

        ratios = np.linspace(0.1, 0.4, len(vc['prunable_modules']))
        mlp_ratio_dict = {m: float(r) for (_, m), r in zip(vc['prunable_modules'], ratios)}

        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vc))
        mp = tp.pruner.MetaPruner(hf_vit, example_inputs, **pruner_kwargs)
        mp.step()

        after = param_count(hf_vit)
        assert after < orig_params

        # embed_dim preserved
        for _, mod in hf_vit.named_modules():
            if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
                assert mod.query.in_features == 384
        for name, mod in hf_vit.named_modules():
            if name.endswith('.output.dense') and isinstance(mod, nn.Linear):
                assert mod.out_features == 384

        with torch.no_grad():
            out = hf_vit(example_inputs)
        assert out.last_hidden_state.shape[2] == 384

        # Intermediate sizes actually reduced
        for name, mod in hf_vit.named_modules():
            if 'intermediate.dense' in name and isinstance(mod, nn.Linear):
                assert mod.out_features < 1536

    def test_deit(self, hf_deit, pruner, example_inputs):
        vc = pruner._detect_vit_config(hf_deit)
        orig_params = param_count(hf_deit)

        mlp_ratio_dict = {m: 0.3 for _, m in vc['prunable_modules']}
        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vc))
        mp = tp.pruner.MetaPruner(hf_deit, example_inputs, **pruner_kwargs)
        mp.step()

        after = param_count(hf_deit)
        assert after < orig_params

        with torch.no_grad():
            out = hf_deit(example_inputs)
        assert out.last_hidden_state.shape[2] == 384

    def test_beit(self, hf_beit, pruner, example_inputs):
        vc = pruner._detect_vit_config(hf_beit)
        orig_params = param_count(hf_beit)

        mlp_ratio_dict = {m: 0.3 for _, m in vc['prunable_modules']}
        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vc))
        mp = tp.pruner.MetaPruner(hf_beit, example_inputs, **pruner_kwargs)
        mp.step()

        after = param_count(hf_beit)
        assert after < orig_params

        with torch.no_grad():
            out = hf_beit(example_inputs)
        assert out.last_hidden_state.shape[2] == 384


# ---------------------------------------------------------------------------
# 5. Two-path combined (MLP + heads)
# ---------------------------------------------------------------------------

class TestTwoPathCombined:
    def _run_two_path(self, model, pruner, example_inputs, embed_dim=384):
        """Shared two-path pruning logic for any HF model."""
        vit_config = pruner._detect_vit_config(model)
        orig_params = param_count(model)

        # Step 1: MLP pruning
        mlp_ratio_dict = {m: 0.3 for _, m in vit_config['prunable_modules']}
        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vit_config))
        mp = tp.pruner.MetaPruner(model, example_inputs, **pruner_kwargs)
        mp.step()

        after_mlp = param_count(model)
        assert after_mlp < orig_params

        # Step 2: Head pruning via DG
        current_vc = pruner._detect_vit_config(model)
        for aname, attn_mod in current_vc['attention_modules']:
            pruner._prune_attention_heads(
                model, example_inputs, attn_mod, aname,
                0.25, 'huggingface', current_vc,
            )

        after = param_count(model)
        assert after < after_mlp

        # Heads reduced
        for _, mod in model.named_modules():
            if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
                assert mod.num_attention_heads < 6
                break

        with torch.no_grad():
            out = model(example_inputs)
        assert out.last_hidden_state.shape[0] == 1
        assert out.last_hidden_state.shape[2] == embed_dim

    def test_vit(self, hf_vit, pruner, example_inputs):
        self._run_two_path(hf_vit, pruner, example_inputs)

    def test_deit(self, hf_deit, pruner, example_inputs):
        self._run_two_path(hf_deit, pruner, example_inputs)

    def test_beit(self, hf_beit, pruner, example_inputs):
        self._run_two_path(hf_beit, pruner, example_inputs)


# ---------------------------------------------------------------------------
# 6. CWP_Pruning end-to-end
# ---------------------------------------------------------------------------

class TestCWPPruning:
    def test_vit_for_classification(self, hf_vit_classifier, pruner, example_inputs):
        vc = pruner._detect_vit_config(hf_vit_classifier)
        orig_params = param_count(hf_vit_classifier)

        pruner.sparsity_dict = {}
        for name, _ in vc['prunable_modules']:
            pruner.sparsity_dict[name] = 0.3
        for name, _ in vc['attention_modules']:
            pruner.sparsity_dict[name] = 0.25

        class FakeLoader:
            def __iter__(self):
                return iter([(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))])

        pruner.model = hf_vit_classifier
        pruner.dataloader = {'test': FakeLoader()}
        pruner.CWP_Pruning()

        after = param_count(pruner.model)
        assert after < orig_params

        with torch.no_grad():
            out = pruner.model(example_inputs)
        assert out.logits.shape == torch.Size([1, 10])

        # Verify structure
        for _, mod in pruner.model.named_modules():
            if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
                assert mod.num_attention_heads < 6
                assert mod.query.out_features == mod.key.out_features
                assert mod.query.out_features == mod.value.out_features
                break


# ---------------------------------------------------------------------------
# 7. Per-layer varying head sparsity
# ---------------------------------------------------------------------------

class TestPerLayerSparsity:
    def test_varying_sparsity(self, hf_vit, pruner, example_inputs):
        """Prune different numbers of heads per attention layer."""
        vc = pruner._detect_vit_config(hf_vit)
        orig_params = param_count(hf_vit)

        # layer 0 -> 1/6, layer 1 -> 2/6, layer 2 -> 3/6, layer 3 -> 1/6
        sparsities = [1/6, 2/6, 3/6, 1/6]
        expected_remaining = [5, 4, 3, 5]

        for (aname, attn_mod), sp in zip(vc['attention_modules'], sparsities):
            pruner._prune_attention_heads(
                hf_vit, example_inputs, attn_mod, aname,
                sp, 'huggingface', vc,
            )

        after = param_count(hf_vit)
        assert after < orig_params

        attn_modules = [
            (name, mod) for name, mod in hf_vit.named_modules()
            if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads')
        ]

        for i, ((name, mod), expected) in enumerate(zip(attn_modules, expected_remaining)):
            assert mod.num_attention_heads == expected, \
                f"Layer {i} ({name}): expected {expected} heads, got {mod.num_attention_heads}"

        with torch.no_grad():
            out = hf_vit(example_inputs)
        assert out.last_hidden_state.shape[2] == 384


# ---------------------------------------------------------------------------
# 8. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_single_head_prune(self, hf_vit, pruner, example_inputs):
        """Prune exactly 1 head from each attention layer."""
        vc = pruner._detect_vit_config(hf_vit)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                hf_vit, example_inputs, attn_mod, aname,
                sparsity=0.1, framework='huggingface', vit_config=vc,
            )

        for _, mod in hf_vit.named_modules():
            if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
                assert mod.num_attention_heads == 5
                assert mod.query.out_features == 5 * 64
                break

        with torch.no_grad():
            out = hf_vit(example_inputs)
        assert out.last_hidden_state.shape[2] == 384

    def test_max_sparsity_clamp(self, hf_vit, pruner, example_inputs):
        """Sparsity of 0.99 should not remove ALL heads — at least 1 must remain."""
        vc = pruner._detect_vit_config(hf_vit)

        # Only prune first attention layer
        aname, attn_mod = vc['attention_modules'][0]
        pruner._prune_attention_heads(
            hf_vit, example_inputs, attn_mod, aname,
            sparsity=0.99, framework='huggingface', vit_config=vc,
        )

        for _, mod in hf_vit.named_modules():
            if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
                assert mod.num_attention_heads >= 1
                break

        with torch.no_grad():
            out = hf_vit(example_inputs)
        assert out.last_hidden_state.shape[2] == 384


# ---------------------------------------------------------------------------
# 9. Embed dim preservation (thorough)
# ---------------------------------------------------------------------------

class TestEmbedDimPreservation:
    def test_after_head_pruning(self, hf_vit, pruner, example_inputs):
        """After head pruning, embed_dim is preserved throughout the model."""
        vc = pruner._detect_vit_config(hf_vit)
        embed_dim = 384

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                hf_vit, example_inputs, attn_mod, aname,
                sparsity=0.5, framework='huggingface', vit_config=vc,
            )

        # position_embeddings dim
        for name, mod in hf_vit.named_modules():
            if hasattr(mod, 'position_embeddings'):
                assert mod.position_embeddings.shape[-1] == embed_dim

        # cls_token dim
        for name, mod in hf_vit.named_modules():
            if hasattr(mod, 'cls_token'):
                assert mod.cls_token.shape[-1] == embed_dim

        # output.dense layers output embed_dim
        for name, mod in hf_vit.named_modules():
            if name.endswith('.output.dense') and isinstance(mod, nn.Linear):
                assert mod.out_features == embed_dim

        # intermediate.dense layers input = embed_dim
        for name, mod in hf_vit.named_modules():
            if 'intermediate.dense' in name and isinstance(mod, nn.Linear):
                assert mod.in_features == embed_dim

        with torch.no_grad():
            out = hf_vit(example_inputs)
        assert out.last_hidden_state.shape[2] == embed_dim


# ---------------------------------------------------------------------------
# 10. Deepcopy isolation
# ---------------------------------------------------------------------------

class TestDeepcopyIsolation:
    def test_original_untouched(self, hf_vit, pruner, example_inputs):
        """Pruning a deepcopy doesn't affect the original model."""
        orig_params = param_count(hf_vit)
        orig_head_count = None
        for _, mod in hf_vit.named_modules():
            if hasattr(mod, 'num_attention_heads'):
                orig_head_count = mod.num_attention_heads
                break

        model_copy = copy.deepcopy(hf_vit)
        vc = pruner._detect_vit_config(model_copy)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                model_copy, example_inputs, attn_mod, aname,
                sparsity=0.5, framework='huggingface', vit_config=vc,
            )

        # Copy was pruned
        assert param_count(model_copy) < orig_params

        # Original is untouched
        assert param_count(hf_vit) == orig_params
        for _, mod in hf_vit.named_modules():
            if hasattr(mod, 'num_attention_heads'):
                assert mod.num_attention_heads == orig_head_count
                break
