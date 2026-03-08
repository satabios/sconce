"""
Pytest-based InternViT pruning tests for sconce.

Tests cover OpenGVLab InternViT variants (trust_remote_code) using small
locally-configured models.  Both V1 and V2.5 share the same InternAttention
architecture with fused QKV and are detected as 'timm'-like.

Test categories:
  1. Config detection — detected as timm framework
  2. Fused QKV head importance computation
  3. DG-based head pruning standalone
  4. MLP-only MetaPruner pruning
  5. Two-path combined (MLP + heads) pruning
  6. CWP_Pruning end-to-end simulation
  7. Embed_dim preservation after pruning
  8. Deepcopy isolation

Run:
    pytest tests/test_internvit_pruning.py -v
"""

import os
import sys

# Add tests dir so conftest is importable when running directly
sys.path.insert(0, os.path.dirname(__file__))
# Add project root to path so `sconce` is importable when running directly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import copy
import types

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch_pruning as tp

from sconce.pruner import prune

from conftest import param_count

try:
    from transformers import AutoConfig, AutoModel
    _test_config = AutoConfig.from_pretrained(
        'OpenGVLab/InternViT-300M-448px', trust_remote_code=True,
    )
    INTERNVIT_AVAILABLE = True
except Exception:
    INTERNVIT_AVAILABLE = False

try:
    _test_config_v25 = AutoConfig.from_pretrained(
        'OpenGVLab/InternViT-300M-448px-V2_5', trust_remote_code=True,
    )
    INTERNVIT_V25_AVAILABLE = True
except Exception:
    INTERNVIT_V25_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not INTERNVIT_AVAILABLE, reason="InternViT not available",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_small_internvit():
    """Create a small InternViT for fast testing."""
    config = AutoConfig.from_pretrained(
        'OpenGVLab/InternViT-300M-448px', trust_remote_code=True,
    )
    config.hidden_size = 384
    config.num_hidden_layers = 4
    config.num_attention_heads = 6
    config.intermediate_size = 1536
    config.image_size = 224
    config.patch_size = 16
    config.use_flash_attn = False
    config.drop_path_rate = 0.0
    model = AutoModel.from_config(config, trust_remote_code=True)
    # InternViT defaults to bfloat16; cast to float32 for stable testing
    return model.float()


def make_small_internvit_v25():
    """Create a small InternViT V2.5 for fast testing."""
    config = AutoConfig.from_pretrained(
        'OpenGVLab/InternViT-300M-448px-V2_5', trust_remote_code=True,
    )
    config.hidden_size = 384
    config.num_hidden_layers = 4
    config.num_attention_heads = 6
    config.intermediate_size = 1536
    config.image_size = 224
    config.patch_size = 16
    config.use_flash_attn = False
    config.drop_path_rate = 0.0
    model = AutoModel.from_config(config, trust_remote_code=True)
    return model.float()


def _patched_internvit_attn_forward(self, hidden_states):
    """Patched forward for InternAttention compatible with pruned head counts.

    InternViT's original ``_naive_attn`` uses ``C // self.num_heads`` to
    compute head_dim from the *input* dimension.  After head pruning
    num_heads changes while head_dim must stay constant, so this version
    uses ``self.head_dim`` directly and ``-1`` for the final reshape.
    """
    B, N, C = hidden_states.shape
    qkv = self.qkv(hidden_states).reshape(
        B, N, 3, self.num_heads, self.head_dim,
    ).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)

    if self.qk_normalization:
        B_, H_, N_, D_ = q.shape
        q = self.q_norm(
            q.transpose(1, 2).flatten(-2, -1)
        ).view(B_, N_, H_, D_).transpose(1, 2)
        k = self.k_norm(
            k.transpose(1, 2).flatten(-2, -1)
        ).view(B_, N_, H_, D_).transpose(1, 2)

    attn = (q * self.scale) @ k.transpose(-2, -1)
    attn = attn.softmax(dim=-1)
    attn = self.attn_drop(attn)

    x = (attn @ v).transpose(1, 2).reshape(B, N, -1)
    x = self.proj(x)
    x = self.proj_drop(x)
    return x


def patch_internvit_attention(model):
    """Patch all InternAttention modules for pruning compatibility."""
    for _, module in model.named_modules():
        if hasattr(module, 'qkv') and hasattr(module, 'proj') and hasattr(module, 'num_heads'):
            module.forward = types.MethodType(_patched_internvit_attn_forward, module)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def internvit():
    model = make_small_internvit()
    model.eval()
    return model


@pytest.fixture
def internvit_v25():
    model = make_small_internvit_v25()
    model.eval()
    return model


skip_v25 = pytest.mark.skipif(
    not INTERNVIT_V25_AVAILABLE, reason="InternViT V2.5 not available",
)


# ---------------------------------------------------------------------------
# 1. Config detection
# ---------------------------------------------------------------------------

class TestDetectConfig:
    def test_internvit(self, internvit, pruner):
        vc = pruner._detect_vit_config(internvit)

        # InternViT detected as timm-like (qkv, proj, num_heads)
        assert vc['framework'] == 'timm'

        # 4 layers -> 4 prunable fc1 modules
        assert len(vc['prunable_modules']) == 4
        for name, mod in vc['prunable_modules']:
            assert '.fc1' in name
            assert isinstance(mod, nn.Linear)
            assert mod.in_features == 384
            assert mod.out_features == 1536

        # 4 attention modules with fused QKV
        assert len(vc['attention_modules']) == 4
        for name, mod in vc['attention_modules']:
            assert hasattr(mod, 'qkv')
            assert hasattr(mod, 'proj')
            assert mod.num_heads == 6

        # num_heads maps qkv Linear -> int for each layer
        assert vc['num_heads'] is not None
        assert len(vc['num_heads']) == 4
        for key, val in vc['num_heads'].items():
            assert isinstance(key, nn.Linear)
            assert val == 6

        # unwrapped_parameters: position_embedding
        assert vc['unwrapped_parameters'] is not None
        assert len(vc['unwrapped_parameters']) >= 1

        # QKV, proj, fc2 excluded from prunable
        prunable_names = {n for n, _ in vc['prunable_modules']}
        for n, _ in internvit.named_modules():
            if n.endswith('.qkv') or n.endswith('.proj') or n.endswith('.fc2'):
                assert n not in prunable_names

    @skip_v25
    def test_internvit_v25(self, internvit_v25, pruner):
        vc = pruner._detect_vit_config(internvit_v25)

        assert vc['framework'] == 'timm'
        assert len(vc['prunable_modules']) == 4
        assert len(vc['attention_modules']) == 4

        for name, mod in vc['prunable_modules']:
            assert '.fc1' in name
            assert mod.in_features == 384
            assert mod.out_features == 1536

        for name, mod in vc['attention_modules']:
            assert hasattr(mod, 'qkv')
            assert mod.num_heads == 6

        assert vc['num_heads'] is not None
        assert len(vc['num_heads']) == 4


# ---------------------------------------------------------------------------
# 2. Head importance — fused QKV
# ---------------------------------------------------------------------------

class TestHeadImportance:
    def test_fused_qkv(self, pruner):
        num_heads, head_dim, embed_dim = 6, 64, 384
        w = torch.randn(3 * num_heads * head_dim, embed_dim)

        importance = pruner.get_attention_head_importance(w, num_heads)
        assert importance.shape == (num_heads,)
        assert torch.all(importance >= 0)

    def test_zeroed_head_detected(self, pruner):
        """Zeroing a head's weights in the fused QKV makes it least important."""
        num_heads, head_dim, embed_dim = 6, 64, 384
        w = torch.randn(3 * num_heads * head_dim, embed_dim)

        target_head = 2
        for qkv_offset in range(3):
            start = qkv_offset * num_heads * head_dim + target_head * head_dim
            w[start:start + head_dim, :] = 0

        importance = pruner.get_attention_head_importance(w, num_heads)
        assert torch.argmin(importance).item() == target_head


# ---------------------------------------------------------------------------
# 3. DG head pruning standalone
# ---------------------------------------------------------------------------

class TestDGHeadPruning:
    def test_internvit(self, internvit, pruner, example_inputs):
        orig_params = param_count(internvit)
        patch_internvit_attention(internvit)
        vc = pruner._detect_vit_config(internvit)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                internvit, example_inputs, attn_mod, aname,
                sparsity=2/6, framework='timm', vit_config=vc,
            )

        after = param_count(internvit)
        assert after < orig_params

        with torch.no_grad():
            out = internvit(example_inputs)
        assert out.last_hidden_state.shape[0] == 1
        assert out.last_hidden_state.shape[2] == 384

        for _, mod in internvit.named_modules():
            if hasattr(mod, 'qkv') and hasattr(mod, 'num_heads'):
                assert mod.num_heads == 4
                assert mod.qkv.out_features == 3 * 4 * 64
                break

    @skip_v25
    def test_internvit_v25(self, internvit_v25, pruner, example_inputs):
        orig_params = param_count(internvit_v25)
        patch_internvit_attention(internvit_v25)
        vc = pruner._detect_vit_config(internvit_v25)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                internvit_v25, example_inputs, attn_mod, aname,
                sparsity=2/6, framework='timm', vit_config=vc,
            )

        after = param_count(internvit_v25)
        assert after < orig_params

        with torch.no_grad():
            out = internvit_v25(example_inputs)
        assert out.last_hidden_state.shape[0] == 1
        assert out.last_hidden_state.shape[2] == 384

        for _, mod in internvit_v25.named_modules():
            if hasattr(mod, 'qkv') and hasattr(mod, 'num_heads'):
                assert mod.num_heads == 4
                break


# ---------------------------------------------------------------------------
# 4. MLP-only MetaPruner
# ---------------------------------------------------------------------------

class TestMLPOnlyPruning:
    def test_internvit(self, internvit, pruner, example_inputs):
        vc = pruner._detect_vit_config(internvit)
        orig_params = param_count(internvit)

        ratios = np.linspace(0.1, 0.4, len(vc['prunable_modules']))
        mlp_ratio_dict = {m: float(r) for (_, m), r in zip(vc['prunable_modules'], ratios)}

        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vc))
        mp = tp.pruner.MetaPruner(internvit, example_inputs, **pruner_kwargs)
        mp.step()

        after = param_count(internvit)
        assert after < orig_params

        # embed_dim preserved
        for _, mod in internvit.named_modules():
            if hasattr(mod, 'qkv'):
                assert mod.qkv.in_features == 384
            if hasattr(mod, 'proj') and hasattr(mod, 'qkv'):
                assert mod.proj.out_features == 384

        with torch.no_grad():
            out = internvit(example_inputs)
        assert out.last_hidden_state.shape[2] == 384

        # Intermediate sizes actually reduced
        for name, mod in internvit.named_modules():
            if name.endswith('.fc1') and isinstance(mod, nn.Linear):
                assert mod.out_features < 1536

    @skip_v25
    def test_internvit_v25(self, internvit_v25, pruner, example_inputs):
        vc = pruner._detect_vit_config(internvit_v25)
        orig_params = param_count(internvit_v25)

        mlp_ratio_dict = {m: 0.3 for _, m in vc['prunable_modules']}
        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vc))
        mp = tp.pruner.MetaPruner(internvit_v25, example_inputs, **pruner_kwargs)
        mp.step()

        after = param_count(internvit_v25)
        assert after < orig_params

        with torch.no_grad():
            out = internvit_v25(example_inputs)
        assert out.last_hidden_state.shape[2] == 384


# ---------------------------------------------------------------------------
# 5. Two-path combined (MLP + heads)
# ---------------------------------------------------------------------------

class TestTwoPathCombined:
    def test_internvit(self, internvit, pruner, example_inputs):
        vit_config = pruner._detect_vit_config(internvit)
        orig_params = param_count(internvit)

        # Step 1: MLP pruning
        mlp_ratio_dict = {m: 0.3 for _, m in vit_config['prunable_modules']}
        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vit_config))
        mp = tp.pruner.MetaPruner(internvit, example_inputs, **pruner_kwargs)
        mp.step()

        after_mlp = param_count(internvit)
        assert after_mlp < orig_params

        # Step 2: Head pruning via DG
        patch_internvit_attention(internvit)
        current_vc = pruner._detect_vit_config(internvit)
        for aname, attn_mod in current_vc['attention_modules']:
            pruner._prune_attention_heads(
                internvit, example_inputs, attn_mod, aname,
                0.25, 'timm', current_vc,
            )

        after = param_count(internvit)
        assert after < after_mlp

        # Heads reduced
        for _, mod in internvit.named_modules():
            if hasattr(mod, 'qkv') and hasattr(mod, 'num_heads'):
                assert mod.num_heads < 6
                break

        with torch.no_grad():
            out = internvit(example_inputs)
        assert out.last_hidden_state.shape[0] == 1
        assert out.last_hidden_state.shape[2] == 384

    @skip_v25
    def test_internvit_v25(self, internvit_v25, pruner, example_inputs):
        vit_config = pruner._detect_vit_config(internvit_v25)
        orig_params = param_count(internvit_v25)

        # Step 1: MLP pruning
        mlp_ratio_dict = {m: 0.3 for _, m in vit_config['prunable_modules']}
        pruner_kwargs = {
            'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            'pruning_ratio': 0,
            'pruning_ratio_dict': mlp_ratio_dict,
        }
        pruner_kwargs.update(pruner._build_metapruner_kwargs(vit_config))
        mp = tp.pruner.MetaPruner(internvit_v25, example_inputs, **pruner_kwargs)
        mp.step()

        after_mlp = param_count(internvit_v25)
        assert after_mlp < orig_params

        # Step 2: Head pruning via DG
        patch_internvit_attention(internvit_v25)
        current_vc = pruner._detect_vit_config(internvit_v25)
        for aname, attn_mod in current_vc['attention_modules']:
            pruner._prune_attention_heads(
                internvit_v25, example_inputs, attn_mod, aname,
                0.25, 'timm', current_vc,
            )

        after = param_count(internvit_v25)
        assert after < after_mlp

        with torch.no_grad():
            out = internvit_v25(example_inputs)
        assert out.last_hidden_state.shape[2] == 384


# ---------------------------------------------------------------------------
# 6. CWP_Pruning end-to-end
# ---------------------------------------------------------------------------

class TestCWPPruning:
    def test_internvit(self, internvit, pruner, example_inputs):
        vc = pruner._detect_vit_config(internvit)
        orig_params = param_count(internvit)

        pruner.sparsity_dict = {}
        for name, _ in vc['prunable_modules']:
            pruner.sparsity_dict[name] = 0.3
        for name, _ in vc['attention_modules']:
            pruner.sparsity_dict[name] = 0.25

        class FakeLoader:
            def __iter__(self):
                return iter([(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))])

        # Override _patch_timm_attention so CWP uses the InternViT-compatible
        # patched forward instead of the generic timm one.
        pruner._patch_timm_attention = lambda model: patch_internvit_attention(model)

        pruner.model = internvit
        pruner.dataloader = {'test': FakeLoader()}
        pruner.attention_heads = True
        pruner.CWP_Pruning()

        after = param_count(pruner.model)
        assert after < orig_params

        with torch.no_grad():
            out = pruner.model(example_inputs)
        assert out.last_hidden_state.shape[0] == 1
        assert out.last_hidden_state.shape[2] == 384

        # Verify structure
        for _, mod in pruner.model.named_modules():
            if hasattr(mod, 'qkv') and hasattr(mod, 'num_heads'):
                assert mod.num_heads < 6
                assert mod.proj.out_features == 384
                break

    @skip_v25
    def test_internvit_v25(self, internvit_v25, pruner, example_inputs):
        vc = pruner._detect_vit_config(internvit_v25)
        orig_params = param_count(internvit_v25)

        pruner.sparsity_dict = {}
        for name, _ in vc['prunable_modules']:
            pruner.sparsity_dict[name] = 0.3
        for name, _ in vc['attention_modules']:
            pruner.sparsity_dict[name] = 0.25

        class FakeLoader:
            def __iter__(self):
                return iter([(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))])

        pruner._patch_timm_attention = lambda model: patch_internvit_attention(model)

        pruner.model = internvit_v25
        pruner.dataloader = {'test': FakeLoader()}
        pruner.attention_heads = True
        pruner.CWP_Pruning()

        after = param_count(pruner.model)
        assert after < orig_params

        with torch.no_grad():
            out = pruner.model(example_inputs)
        assert out.last_hidden_state.shape[2] == 384


# ---------------------------------------------------------------------------
# 7. Embed dim preservation
# ---------------------------------------------------------------------------

class TestEmbedDimPreservation:
    def test_after_head_pruning(self, internvit, pruner, example_inputs):
        """After head pruning, embed_dim is preserved throughout the model."""
        patch_internvit_attention(internvit)
        vc = pruner._detect_vit_config(internvit)
        embed_dim = 384

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                internvit, example_inputs, attn_mod, aname,
                sparsity=0.5, framework='timm', vit_config=vc,
            )

        # position_embedding dim
        for name, param in internvit.named_parameters():
            if 'position_embedding' in name:
                assert param.shape[-1] == embed_dim

        # Attention proj output = embed_dim
        for name, mod in internvit.named_modules():
            if name.endswith('.attn.proj') and isinstance(mod, nn.Linear):
                assert mod.out_features == embed_dim

        # fc1 input = embed_dim
        for name, mod in internvit.named_modules():
            if name.endswith('.fc1') and isinstance(mod, nn.Linear):
                assert mod.in_features == embed_dim

        # fc2 output = embed_dim
        for name, mod in internvit.named_modules():
            if name.endswith('.fc2') and isinstance(mod, nn.Linear):
                assert mod.out_features == embed_dim

        with torch.no_grad():
            out = internvit(example_inputs)
        assert out.last_hidden_state.shape[2] == embed_dim

    @skip_v25
    def test_after_head_pruning_v25(self, internvit_v25, pruner, example_inputs):
        """After head pruning, embed_dim is preserved (V2.5)."""
        patch_internvit_attention(internvit_v25)
        vc = pruner._detect_vit_config(internvit_v25)
        embed_dim = 384

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                internvit_v25, example_inputs, attn_mod, aname,
                sparsity=0.5, framework='timm', vit_config=vc,
            )

        for name, mod in internvit_v25.named_modules():
            if name.endswith('.attn.proj') and isinstance(mod, nn.Linear):
                assert mod.out_features == embed_dim
        for name, mod in internvit_v25.named_modules():
            if name.endswith('.fc1') and isinstance(mod, nn.Linear):
                assert mod.in_features == embed_dim

        with torch.no_grad():
            out = internvit_v25(example_inputs)
        assert out.last_hidden_state.shape[2] == embed_dim


# ---------------------------------------------------------------------------
# 8. Deepcopy isolation
# ---------------------------------------------------------------------------

class TestDeepcopyIsolation:
    def test_original_untouched(self, internvit, pruner, example_inputs):
        """Pruning a deepcopy doesn't affect the original model."""
        orig_params = param_count(internvit)
        orig_head_count = None
        for _, mod in internvit.named_modules():
            if hasattr(mod, 'num_heads') and hasattr(mod, 'qkv'):
                orig_head_count = mod.num_heads
                break

        model_copy = copy.deepcopy(internvit)
        patch_internvit_attention(model_copy)
        vc = pruner._detect_vit_config(model_copy)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                model_copy, example_inputs, attn_mod, aname,
                sparsity=0.5, framework='timm', vit_config=vc,
            )

        # Copy was pruned
        assert param_count(model_copy) < orig_params

        # Original is untouched
        assert param_count(internvit) == orig_params
        for _, mod in internvit.named_modules():
            if hasattr(mod, 'num_heads') and hasattr(mod, 'qkv'):
                assert mod.num_heads == orig_head_count
                break

    @skip_v25
    def test_original_untouched_v25(self, internvit_v25, pruner, example_inputs):
        """Pruning a deepcopy doesn't affect the original (V2.5)."""
        orig_params = param_count(internvit_v25)
        orig_head_count = None
        for _, mod in internvit_v25.named_modules():
            if hasattr(mod, 'num_heads') and hasattr(mod, 'qkv'):
                orig_head_count = mod.num_heads
                break

        model_copy = copy.deepcopy(internvit_v25)
        patch_internvit_attention(model_copy)
        vc = pruner._detect_vit_config(model_copy)

        for aname, attn_mod in vc['attention_modules']:
            pruner._prune_attention_heads(
                model_copy, example_inputs, attn_mod, aname,
                sparsity=0.5, framework='timm', vit_config=vc,
            )

        assert param_count(model_copy) < orig_params

        assert param_count(internvit_v25) == orig_params
        for _, mod in internvit_v25.named_modules():
            if hasattr(mod, 'num_heads') and hasattr(mod, 'qkv'):
                assert mod.num_heads == orig_head_count
                break
