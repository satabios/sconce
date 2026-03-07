"""
Pytest-based smoke tests for ViT structural pruning support in sconce.

Tests cover three ViT frameworks (timm, torchvision, HuggingFace) across:
  1. _detect_vit_config: correct framework detection, module classification
  2. get_attention_head_importance: fused QKV and separate Q/K/V paths
  3. Two-path pruning: MLP via MetaPruner + heads via DG
  4. CWP_Pruning end-to-end simulation

Run:
    pytest tests/test_vit_pruning.py -v
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

from conftest import param_count


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def timm_vit():
    import timm
    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
    model.eval()
    return model


@pytest.fixture
def torchvision_vit():
    import torchvision
    model = torchvision.models.vit_b_16(weights=None)
    model.eval()
    return model


@pytest.fixture
def hf_vit():
    from transformers import ViTModel, ViTConfig
    config = ViTConfig(
        hidden_size=384, num_hidden_layers=4,
        num_attention_heads=6, intermediate_size=1536,
    )
    model = ViTModel(config)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# 1. _detect_vit_config tests
# ---------------------------------------------------------------------------

class TestDetectConfig:
    def test_timm(self, timm_vit, pruner):
        vc = pruner._detect_vit_config(timm_vit)

        assert vc['framework'] == 'timm'

        # prunable_modules should only contain fc1 (MLP intermediate) layers
        for name, mod in vc['prunable_modules']:
            assert '.fc1' in name, f"Unexpected prunable module: {name}"
            assert isinstance(mod, nn.Linear)
            assert mod.out_features > mod.in_features

        # attention_modules should contain Attention parents (with .qkv, .proj)
        for name, mod in vc['attention_modules']:
            assert hasattr(mod, 'qkv')
            assert hasattr(mod, 'num_heads')

        # num_heads maps qkv Linear -> int
        assert vc['num_heads'] is not None
        for key, val in vc['num_heads'].items():
            assert isinstance(key, nn.Linear)
            assert isinstance(val, int) and val > 0

        # unwrapped_parameters: (Parameter, int) tuples
        assert vc['unwrapped_parameters'] is not None
        for param, dim in vc['unwrapped_parameters']:
            assert isinstance(param, nn.Parameter)
            assert isinstance(dim, int)

        # ignored_layers should have the classifier head
        assert len(vc['ignored_layers']) >= 1

    def test_torchvision(self, torchvision_vit, pruner):
        vc = pruner._detect_vit_config(torchvision_vit)

        assert vc['framework'] == 'torchvision'

        # prunable_modules: only MLP intermediate (mlp.0)
        for name, mod in vc['prunable_modules']:
            assert isinstance(mod, nn.Linear)
            assert mod.out_features > mod.in_features

        # No Conv2d (patch embed), no MHA, no out_proj in prunable
        prunable_names = [n for n, _ in vc['prunable_modules']]
        for name in prunable_names:
            assert 'conv_proj' not in name
            assert 'out_proj' not in name

        # attention_modules: only ONE representative nn.MultiheadAttention
        assert len(vc['attention_modules']) == 1
        for name, mod in vc['attention_modules']:
            assert isinstance(mod, nn.MultiheadAttention)

        assert vc['num_heads'] is not None
        # num_heads should map ALL MHAs (not just the representative one)
        assert len(vc['num_heads']) == 12
        assert vc['unwrapped_parameters'] is not None

    def test_huggingface(self, hf_vit, pruner):
        vc = pruner._detect_vit_config(hf_vit)

        assert vc['framework'] == 'huggingface'

        # prunable_modules: intermediate.dense only (MLP fc1)
        for name, mod in vc['prunable_modules']:
            assert 'intermediate.dense' in name
            assert mod.out_features > mod.in_features

        # Should NOT include output.dense, query, key, value
        prunable_names = [n for n, _ in vc['prunable_modules']]
        for name in prunable_names:
            assert '.output.dense' not in name
            assert not name.endswith('.query')
            assert not name.endswith('.key')
            assert not name.endswith('.value')

        # attention_modules: ViTSelfAttention with .query, .key, .value
        for name, mod in vc['attention_modules']:
            assert hasattr(mod, 'query')
            assert hasattr(mod, 'num_attention_heads')

        assert vc['num_heads'] is not None
        assert vc['unwrapped_parameters'] is not None

    def test_non_vit(self, pruner):
        """_detect_vit_config on a plain CNN should return framework=None."""
        import torchvision
        model = torchvision.models.resnet18(weights=None)
        vc = pruner._detect_vit_config(model)

        assert vc['framework'] is None
        assert vc['num_heads'] is None
        assert len(vc['attention_modules']) == 0
        has_conv = any(isinstance(m, nn.Conv2d) for _, m in vc['prunable_modules'])
        assert has_conv


# ---------------------------------------------------------------------------
# 2. get_attention_head_importance tests
# ---------------------------------------------------------------------------

class TestHeadImportance:
    def test_fused_qkv(self, pruner):
        num_heads, head_dim, embed_dim = 12, 64, 768
        w = torch.randn(3 * num_heads * head_dim, embed_dim)

        importance = pruner.get_attention_head_importance(w, num_heads)
        assert importance.shape == (num_heads,)
        assert torch.all(importance >= 0)

    def test_separate_qkv(self, pruner):
        num_heads, head_dim, embed_dim = 6, 64, 384
        q_w = torch.randn(num_heads * head_dim, embed_dim)
        k_w = torch.randn(num_heads * head_dim, embed_dim)
        v_w = torch.randn(num_heads * head_dim, embed_dim)

        importance = pruner.get_attention_head_importance([q_w, k_w, v_w], num_heads)
        assert importance.shape == (num_heads,)
        assert torch.all(importance >= 0)


# ---------------------------------------------------------------------------
# 3. Standalone pruning paths
# ---------------------------------------------------------------------------

class TestStandalonePruning:
    def test_mlp_only_timm(self, timm_vit, pruner, example_inputs):
        vc = pruner._detect_vit_config(timm_vit)
        orig_params = param_count(timm_vit)

        ratios = np.linspace(0.1, 0.5, len(vc['prunable_modules']))
        mlp_ratio_dict = {m: float(r) for (_, m), r in zip(vc['prunable_modules'], ratios)}

        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vc))
        mp = tp.pruner.MetaPruner(timm_vit, example_inputs, **pruner_kwargs)
        mp.step()

        after = param_count(timm_vit)
        assert after < orig_params

        # embed_dim preserved
        for _, mod in timm_vit.named_modules():
            if hasattr(mod, 'qkv'):
                assert mod.qkv.in_features == 384
            if hasattr(mod, 'proj') and hasattr(mod, 'qkv'):
                assert mod.proj.out_features == 384

        out = timm_vit(example_inputs)
        assert out.shape == torch.Size([1, 10])

    def test_dg_head_pruning_timm(self, timm_vit, pruner, example_inputs):
        orig_params = param_count(timm_vit)

        pruner._patch_timm_attention(timm_vit)
        vc = pruner._detect_vit_config(timm_vit)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                timm_vit, example_inputs, attn_mod, aname,
                sparsity=2/6, framework='timm', vit_config=vc,
            )

        after = param_count(timm_vit)
        assert after < orig_params

        with torch.no_grad():
            out = timm_vit(example_inputs)
        assert out.shape == torch.Size([1, 10])

        for _, mod in timm_vit.named_modules():
            if hasattr(mod, 'qkv') and hasattr(mod, 'num_heads'):
                assert mod.num_heads == 4
                break

    def test_dg_head_pruning_torchvision(self, torchvision_vit, pruner, example_inputs):
        orig_params = param_count(torchvision_vit)
        vc = pruner._detect_vit_config(torchvision_vit)

        assert len(vc['attention_modules']) == 1

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                torchvision_vit, example_inputs, attn_mod, aname,
                sparsity=0.25, framework='torchvision', vit_config=vc,
            )

        after = param_count(torchvision_vit)
        assert after < orig_params

        with torch.no_grad():
            out = torchvision_vit(example_inputs)
        assert out.shape == torch.Size([1, 1000])

        for _, mod in torchvision_vit.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                assert mod.num_heads == 9
                break


# ---------------------------------------------------------------------------
# 4. Two-path combined (MLP + heads)
# ---------------------------------------------------------------------------

class TestTwoPathCombined:
    def test_timm(self, timm_vit, pruner, example_inputs):
        vit_config = pruner._detect_vit_config(timm_vit)
        orig_params = param_count(timm_vit)

        # Step 1: MLP pruning
        mlp_ratio_dict = {m: 0.3 for _, m in vit_config['prunable_modules']}
        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vit_config))
        mp = tp.pruner.MetaPruner(timm_vit, example_inputs, **pruner_kwargs)
        mp.step()

        # embed_dim unchanged
        for _, mod in timm_vit.named_modules():
            if hasattr(mod, 'qkv'):
                assert mod.qkv.in_features == 384
                break

        # Step 2: Head pruning via DG
        pruner._patch_timm_attention(timm_vit)
        current_vc = pruner._detect_vit_config(timm_vit)
        for aname, attn_mod in current_vc['attention_modules']:
            pruner._prune_attention_heads(
                timm_vit, example_inputs, attn_mod, aname,
                0.25, 'timm', current_vc,
            )

        after = param_count(timm_vit)
        assert after < orig_params

        with torch.no_grad():
            out = timm_vit(example_inputs)
        assert out.shape == torch.Size([1, 10])

        for _, mod in timm_vit.named_modules():
            if hasattr(mod, 'qkv') and hasattr(mod, 'num_heads'):
                assert mod.num_heads < 6
                break

    def test_torchvision(self, torchvision_vit, pruner, example_inputs):
        vit_config = pruner._detect_vit_config(torchvision_vit)
        orig_params = param_count(torchvision_vit)

        # Step 1: MLP pruning
        mlp_ratio_dict = {m: 0.3 for _, m in vit_config['prunable_modules']}
        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vit_config))
        mp = tp.pruner.MetaPruner(torchvision_vit, example_inputs, **pruner_kwargs)
        mp.step()

        assert torchvision_vit.conv_proj.weight.shape[0] == 768

        # Step 2: Head pruning via DG
        current_vc = pruner._detect_vit_config(torchvision_vit)
        assert len(current_vc['attention_modules']) == 1
        for aname, attn_mod in current_vc['attention_modules']:
            pruner._prune_attention_heads(
                torchvision_vit, example_inputs, attn_mod, aname,
                0.25, 'torchvision', current_vc,
            )

        after = param_count(torchvision_vit)
        assert after < orig_params

        for _, mod in torchvision_vit.named_modules():
            if isinstance(mod, nn.MultiheadAttention):
                assert mod.num_heads == 9
                break

        with torch.no_grad():
            out = torchvision_vit(example_inputs)
        assert out.shape == torch.Size([1, 1000])

    def test_huggingface(self, hf_vit, pruner, example_inputs):
        vit_config = pruner._detect_vit_config(hf_vit)
        orig_params = param_count(hf_vit)

        # Step 1: MLP pruning
        mlp_ratio_dict = {m: 0.3 for _, m in vit_config['prunable_modules']}
        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vit_config))
        mp = tp.pruner.MetaPruner(hf_vit, example_inputs, **pruner_kwargs)
        mp.step()

        after_mlp = param_count(hf_vit)
        assert after_mlp < orig_params

        # Step 2: Head pruning via DG
        current_vc = pruner._detect_vit_config(hf_vit)
        for aname, attn_mod in current_vc['attention_modules']:
            pruner._prune_attention_heads(
                hf_vit, example_inputs, attn_mod, aname,
                0.25, 'huggingface', current_vc,
            )

        after = param_count(hf_vit)
        assert after < after_mlp

        with torch.no_grad():
            out = hf_vit(example_inputs)
        assert out.last_hidden_state.shape[0] == 1


# ---------------------------------------------------------------------------
# 5. CWP_Pruning end-to-end
# ---------------------------------------------------------------------------

class TestCWPPruning:
    def test_timm(self, timm_vit, pruner, example_inputs):
        vc = pruner._detect_vit_config(timm_vit)
        orig_params = param_count(timm_vit)

        pruner.sparsity_dict = {}
        for name, _ in vc['prunable_modules']:
            pruner.sparsity_dict[name] = 0.3
        for name, _ in vc['attention_modules']:
            pruner.sparsity_dict[name] = 0.25

        class FakeLoader:
            def __iter__(self):
                return iter([(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))])

        pruner.model = timm_vit
        pruner.dataloader = {'test': FakeLoader()}
        pruner.CWP_Pruning()

        after = param_count(pruner.model)
        assert after < orig_params

        with torch.no_grad():
            out = pruner.model(example_inputs)
        assert out.shape == torch.Size([1, 10])


# ---------------------------------------------------------------------------
# 6. Patched timm forward equivalence
# ---------------------------------------------------------------------------

class TestPatchedForward:
    def test_timm_equivalence(self, timm_vit, pruner, example_inputs):
        """Patched forward produces same output as original on unpruned model."""
        with torch.no_grad():
            original_out = timm_vit(example_inputs).clone()

        pruner._patch_timm_attention(timm_vit)

        with torch.no_grad():
            patched_out = timm_vit(example_inputs)

        diff = (original_out - patched_out).abs().max().item()
        assert diff < 1e-5, f"Patched forward diverges: max diff = {diff}"
