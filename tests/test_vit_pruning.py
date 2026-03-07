"""
Smoke tests for ViT structural pruning support in sconce.

Tests cover three ViT frameworks (timm, torchvision, HuggingFace) across:
  1. _detect_vit_config: correct framework detection, module classification
  2. get_attention_head_importance: fused QKV and separate Q/K/V paths
  3. Two-path pruning: MLP via MetaPruner + heads via DG
  4. CWP_Pruning end-to-end simulation

Run:
    python tests/test_vit_pruning.py
"""

import os
import sys

# Add project root to path so `sconce` is importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import traceback
import torch
import torch.nn as nn
import numpy as np
import copy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

PASS = 0
FAIL = 0
SKIP = 0


def run_test(name, fn):
    global PASS, FAIL, SKIP
    print(f"\n{'='*70}")
    print(f"  {name}")
    print(f"{'='*70}")
    try:
        result = fn()
        if result == "SKIP":
            SKIP += 1
            print(f"  >> SKIPPED")
        else:
            PASS += 1
            print(f"  >> PASSED")
    except Exception as e:
        FAIL += 1
        print(f"  >> FAILED: {e}")
        traceback.print_exc()


def assert_eq(a, b, msg=""):
    assert a == b, f"Expected {b}, got {a}. {msg}"


def assert_true(cond, msg=""):
    assert cond, msg


def forward_check(model, example_inputs, label=""):
    """Run a forward pass and return output shape. Raises on failure."""
    with torch.no_grad():
        out = model(example_inputs)
    print(f"  Forward {label}: output shape = {out.shape}")
    return out.shape


def param_count(model):
    return sum(p.numel() for p in model.parameters())


# ---------------------------------------------------------------------------
# 1. _detect_vit_config tests
# ---------------------------------------------------------------------------

def test_detect_config_timm():
    import timm
    from sconce.pruner import prune

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
    p = prune()
    p.attention_heads = True
    vc = p._detect_vit_config(model)

    assert_eq(vc['framework'], 'timm')

    # prunable_modules should only contain fc1 (MLP intermediate) layers
    for name, mod in vc['prunable_modules']:
        assert_true('.fc1' in name, f"Unexpected prunable module: {name}")
        assert_true(isinstance(mod, nn.Linear))
        assert_true(mod.out_features > mod.in_features,
                    f"{name}: fc1 should expand ({mod.in_features}->{mod.out_features})")
    print(f"  Prunable modules: {len(vc['prunable_modules'])} (all fc1)")

    # attention_modules should contain Attention parents (with .qkv, .proj)
    for name, mod in vc['attention_modules']:
        assert_true(hasattr(mod, 'qkv'), f"{name} missing .qkv")
        assert_true(hasattr(mod, 'num_heads'), f"{name} missing .num_heads")
    print(f"  Attention modules: {len(vc['attention_modules'])}")

    # num_heads maps qkv Linear -> int
    assert_true(vc['num_heads'] is not None)
    for key, val in vc['num_heads'].items():
        assert_true(isinstance(key, nn.Linear))
        assert_true(isinstance(val, int) and val > 0)

    # unwrapped_parameters: (Parameter, int) tuples
    assert_true(vc['unwrapped_parameters'] is not None)
    for param, dim in vc['unwrapped_parameters']:
        assert_true(isinstance(param, nn.Parameter))
        assert_true(isinstance(dim, int))
    print(f"  Unwrapped params: {len(vc['unwrapped_parameters'])}")

    # ignored_layers should have the classifier head
    assert_true(len(vc['ignored_layers']) >= 1, "Should ignore classifier head")
    print(f"  Ignored layers: {[type(m).__name__ for m in vc['ignored_layers']]}")


def test_detect_config_torchvision():
    import torchvision
    from sconce.pruner import prune

    model = torchvision.models.vit_b_16(weights=None)
    p = prune()
    p.attention_heads = True
    vc = p._detect_vit_config(model)

    assert_eq(vc['framework'], 'torchvision')

    # prunable_modules: only MLP intermediate (mlp.0)
    for name, mod in vc['prunable_modules']:
        assert_true(isinstance(mod, nn.Linear))
        assert_true(mod.out_features > mod.in_features,
                    f"{name}: should expand ({mod.in_features}->{mod.out_features})")
    print(f"  Prunable modules: {len(vc['prunable_modules'])}")

    # No Conv2d (patch embed), no MHA, no out_proj, no mlp.3 in prunable
    prunable_names = [n for n, _ in vc['prunable_modules']]
    for name in prunable_names:
        assert_true('conv_proj' not in name, f"conv_proj should not be prunable: {name}")
        assert_true('out_proj' not in name, f"out_proj should not be prunable: {name}")
    print(f"  No conv_proj/out_proj in prunable: OK")

    # attention_modules: only ONE representative nn.MultiheadAttention
    # (torchvision DG cascades embed_dim globally, so only one is needed)
    assert_eq(len(vc['attention_modules']), 1,
              "torchvision should have exactly 1 representative attention module")
    for name, mod in vc['attention_modules']:
        assert_true(isinstance(mod, nn.MultiheadAttention))
    print(f"  Attention modules: {len(vc['attention_modules'])} (1 representative, DG cascades to all)")

    assert_true(vc['num_heads'] is not None)
    # num_heads should map ALL MHAs (not just the representative one)
    assert_eq(len(vc['num_heads']), 12,
              "num_heads should have all 12 MHAs for MetaPruner compat")
    assert_true(vc['unwrapped_parameters'] is not None)
    print(f"  num_heads entries: {len(vc['num_heads'])} (all MHAs)")
    print(f"  Unwrapped params: {len(vc['unwrapped_parameters'])}")


def test_detect_config_huggingface():
    try:
        from transformers import ViTModel, ViTConfig
    except ImportError:
        print("  transformers not installed")
        return "SKIP"

    from sconce.pruner import prune

    config = ViTConfig(
        hidden_size=384, num_hidden_layers=4,
        num_attention_heads=6, intermediate_size=1536,
    )
    model = ViTModel(config)
    p = prune()
    p.attention_heads = True
    vc = p._detect_vit_config(model)

    assert_eq(vc['framework'], 'huggingface')

    # prunable_modules: intermediate.dense only (MLP fc1)
    for name, mod in vc['prunable_modules']:
        assert_true('intermediate.dense' in name,
                    f"Unexpected prunable: {name}")
        assert_true(mod.out_features > mod.in_features)
    print(f"  Prunable modules: {len(vc['prunable_modules'])}")

    # Should NOT include output.dense, query, key, value
    prunable_names = [n for n, _ in vc['prunable_modules']]
    for name in prunable_names:
        assert_true('.output.dense' not in name, f"output.dense should not be prunable: {name}")
        assert_true(not name.endswith('.query'), f"query should not be prunable: {name}")
        assert_true(not name.endswith('.key'), f"key should not be prunable: {name}")
        assert_true(not name.endswith('.value'), f"value should not be prunable: {name}")
    print(f"  Excluded Q/K/V/output.dense: OK")

    # attention_modules: ViTSelfAttention with .query, .key, .value
    for name, mod in vc['attention_modules']:
        assert_true(hasattr(mod, 'query'))
        assert_true(hasattr(mod, 'num_attention_heads'))
    print(f"  Attention modules: {len(vc['attention_modules'])}")

    assert_true(vc['num_heads'] is not None)
    assert_true(vc['unwrapped_parameters'] is not None)


# ---------------------------------------------------------------------------
# 2. get_attention_head_importance tests
# ---------------------------------------------------------------------------

def test_head_importance_fused_qkv():
    from sconce.pruner import prune
    p = prune()

    num_heads = 12
    head_dim = 64
    embed_dim = 768
    # Fused QKV: (3*num_heads*head_dim, embed_dim)
    w = torch.randn(3 * num_heads * head_dim, embed_dim)

    importance = p.get_attention_head_importance(w, num_heads)
    assert_eq(importance.shape, (num_heads,), "Shape should be (num_heads,)")
    assert_true(torch.all(importance >= 0), "Importance should be non-negative")
    print(f"  Fused QKV importance: shape={importance.shape}, "
          f"min={importance.min():.3f}, max={importance.max():.3f}")


def test_head_importance_separate_qkv():
    from sconce.pruner import prune
    p = prune()

    num_heads = 6
    head_dim = 64
    embed_dim = 384
    q_w = torch.randn(num_heads * head_dim, embed_dim)
    k_w = torch.randn(num_heads * head_dim, embed_dim)
    v_w = torch.randn(num_heads * head_dim, embed_dim)

    importance = p.get_attention_head_importance([q_w, k_w, v_w], num_heads)
    assert_eq(importance.shape, (num_heads,))
    assert_true(torch.all(importance >= 0))
    print(f"  Separate Q/K/V importance: shape={importance.shape}, "
          f"min={importance.min():.3f}, max={importance.max():.3f}")


# ---------------------------------------------------------------------------
# 3. Two-path pruning tests (MLP MetaPruner + DG head pruning)
# ---------------------------------------------------------------------------

def test_two_path_timm():
    import timm
    import torch_pruning as tp
    from sconce.pruner import prune

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
    model.eval()
    p = prune()
    p.attention_heads = True

    vit_config = p._detect_vit_config(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)
    print(f"  Original params: {orig_params:,}")

    forward_check(model, example_inputs, "before pruning")

    # Step 1: MLP pruning via MetaPruner
    mlp_ratio_dict = {m: 0.3 for _, m in vit_config['prunable_modules']}
    pruner_kwargs = {
        'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
        'pruning_ratio': 0,
        'pruning_ratio_dict': mlp_ratio_dict,
    }
    pruner_kwargs.update(p._build_metapruner_kwargs(vit_config))
    pruner = tp.pruner.MetaPruner(model, example_inputs, **pruner_kwargs)
    pruner.step()

    after_mlp = param_count(model)
    print(f"  After MLP pruning: {after_mlp:,} ({1 - after_mlp/orig_params:.1%} reduction)")
    forward_check(model, example_inputs, "after MLP pruning")

    # Verify conv_proj / embed_dim unchanged
    for name, mod in model.named_modules():
        if hasattr(mod, 'qkv'):
            assert_eq(mod.qkv.in_features, 384,
                      f"embed_dim input to qkv should be unchanged")
            break

    # Step 2: Head pruning via DG
    p._patch_timm_attention(model)
    current_vc = p._detect_vit_config(model)
    for aname, attn_mod in current_vc['attention_modules']:
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            0.25, 'timm', current_vc,
        )

    after_heads = param_count(model)
    print(f"  After head pruning: {after_heads:,} ({1 - after_heads/orig_params:.1%} total reduction)")

    out_shape = forward_check(model, example_inputs, "after head pruning")
    assert_eq(out_shape, torch.Size([1, 10]), "Output shape should be [1, 10]")

    # Check head count was updated
    for _, mod in model.named_modules():
        if hasattr(mod, 'qkv') and hasattr(mod, 'num_heads'):
            assert_true(mod.num_heads < 6, f"Heads should be reduced from 6, got {mod.num_heads}")
            break


def test_two_path_torchvision():
    import torchvision
    import torch_pruning as tp
    from sconce.pruner import prune

    model = torchvision.models.vit_b_16(weights=None)
    model.eval()
    p = prune()
    p.attention_heads = True

    vit_config = p._detect_vit_config(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)
    print(f"  Original params: {orig_params:,}")

    forward_check(model, example_inputs, "before pruning")

    # Step 1: MLP pruning
    mlp_ratio_dict = {m: 0.3 for _, m in vit_config['prunable_modules']}
    pruner_kwargs = {
        'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
        'pruning_ratio': 0,
        'pruning_ratio_dict': mlp_ratio_dict,
    }
    pruner_kwargs.update(p._build_metapruner_kwargs(vit_config))
    pruner = tp.pruner.MetaPruner(model, example_inputs, **pruner_kwargs)
    pruner.step()

    after_mlp = param_count(model)
    print(f"  After MLP pruning: {after_mlp:,} ({1 - after_mlp/orig_params:.1%} reduction)")
    forward_check(model, example_inputs, "after MLP pruning")

    # Verify embed_dim unchanged
    assert_eq(model.conv_proj.weight.shape[0], 768, "conv_proj output channels should be unchanged")

    # Step 2: Head pruning via DG (only 1 representative MHA — DG cascades globally)
    current_vc = p._detect_vit_config(model)
    assert_eq(len(current_vc['attention_modules']), 1,
              "torchvision should have 1 representative attention module")
    for aname, attn_mod in current_vc['attention_modules']:
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            0.25, 'torchvision', current_vc,
        )

    after_heads = param_count(model)
    print(f"  After head pruning: {after_heads:,} ({1 - after_heads/orig_params:.1%} total reduction)")

    # Verify all MHAs have reduced heads
    for _, mod in model.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            assert_eq(mod.num_heads, 9, f"Should have 9 heads after pruning 3/12")
            break

    out_shape = forward_check(model, example_inputs, "after head pruning")
    assert_eq(out_shape, torch.Size([1, 1000]), "Output shape should be [1, 1000]")


def test_two_path_huggingface():
    try:
        from transformers import ViTModel, ViTConfig
    except ImportError:
        print("  transformers not installed")
        return "SKIP"

    import torch_pruning as tp
    from sconce.pruner import prune

    config = ViTConfig(
        hidden_size=384, num_hidden_layers=4,
        num_attention_heads=6, intermediate_size=1536,
    )
    model = ViTModel(config)
    model.eval()
    p = prune()
    p.attention_heads = True

    vit_config = p._detect_vit_config(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)
    print(f"  Original params: {orig_params:,}")

    with torch.no_grad():
        out = model(example_inputs)
    print(f"  Forward before pruning: last_hidden_state={out.last_hidden_state.shape}")

    # Step 1: MLP pruning
    mlp_ratio_dict = {m: 0.3 for _, m in vit_config['prunable_modules']}
    pruner_kwargs = {
        'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
        'pruning_ratio': 0,
        'pruning_ratio_dict': mlp_ratio_dict,
    }
    pruner_kwargs.update(p._build_metapruner_kwargs(vit_config))
    pruner = tp.pruner.MetaPruner(model, example_inputs, **pruner_kwargs)
    pruner.step()

    after_mlp = param_count(model)
    print(f"  After MLP pruning: {after_mlp:,} ({1 - after_mlp/orig_params:.1%} reduction)")
    with torch.no_grad():
        out = model(example_inputs)
    print(f"  Forward after MLP: last_hidden_state={out.last_hidden_state.shape}")

    # Step 2: Head pruning via DG
    current_vc = p._detect_vit_config(model)
    for aname, attn_mod in current_vc['attention_modules']:
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            0.25, 'huggingface', current_vc,
        )

    after_heads = param_count(model)
    print(f"  After head pruning: {after_heads:,} ({1 - after_heads/orig_params:.1%} total reduction)")

    with torch.no_grad():
        out = model(example_inputs)
    print(f"  Forward after heads: last_hidden_state={out.last_hidden_state.shape}")
    assert_eq(out.last_hidden_state.shape[0], 1)


# ---------------------------------------------------------------------------
# 4. DG-only head pruning (no MLP step first)
# ---------------------------------------------------------------------------

def test_dg_head_pruning_timm_standalone():
    """Head pruning only (no MLP step) on timm - simplest case."""
    import timm
    from sconce.pruner import prune

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
    model.eval()
    p = prune()
    p.attention_heads = True

    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)

    p._patch_timm_attention(model)
    vc = p._detect_vit_config(model)

    # Prune 2 heads (out of 6) from every attention block
    for aname, attn_mod in vc['attention_modules']:
        orig_heads = attn_mod.num_heads
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            sparsity=2/6, framework='timm', vit_config=vc,
        )

    after = param_count(model)
    print(f"  Params: {orig_params:,} -> {after:,} ({1 - after/orig_params:.1%} reduction)")

    out = forward_check(model, example_inputs, "after head-only pruning")
    assert_eq(out, torch.Size([1, 10]))

    # Verify heads reduced
    for _, mod in model.named_modules():
        if hasattr(mod, 'qkv') and hasattr(mod, 'num_heads'):
            assert_eq(mod.num_heads, 4, f"Should have 4 heads after pruning 2/6")
            break


def test_dg_head_pruning_torchvision_standalone():
    """Head pruning only on torchvision ViT.

    torchvision DG cascades embed_dim globally, so _detect_vit_config
    returns only ONE representative MHA. Pruning it once affects all layers.
    """
    import torchvision
    from sconce.pruner import prune

    model = torchvision.models.vit_b_16(weights=None)
    model.eval()
    p = prune()
    p.attention_heads = True

    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)

    vc = p._detect_vit_config(model)

    # Should have exactly 1 representative attention module
    assert_eq(len(vc['attention_modules']), 1,
              "torchvision should have 1 representative attention module")

    # Prune 3 heads (out of 12) — DG cascades to ALL MHAs
    for aname, attn_mod in vc['attention_modules']:
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            sparsity=0.25, framework='torchvision', vit_config=vc,
        )

    after = param_count(model)
    print(f"  Params: {orig_params:,} -> {after:,} ({1 - after/orig_params:.1%} reduction)")

    out = forward_check(model, example_inputs, "after head-only pruning")
    assert_eq(out, torch.Size([1, 1000]))

    # Verify num_heads reduced on all MHAs
    for _, mod in model.named_modules():
        if isinstance(mod, nn.MultiheadAttention):
            assert_eq(mod.num_heads, 9, f"Should have 9 heads after pruning 3/12")
            break


# ---------------------------------------------------------------------------
# 5. MLP-only MetaPruner (no head pruning)
# ---------------------------------------------------------------------------

def test_mlp_only_timm():
    """MLP-only pruning via MetaPruner on timm ViT."""
    import timm
    import torch_pruning as tp
    from sconce.pruner import prune

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
    model.eval()
    p = prune()
    p.attention_heads = True

    vc = p._detect_vit_config(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)

    # Varying ratios per block
    ratios = np.linspace(0.1, 0.5, len(vc['prunable_modules']))
    mlp_ratio_dict = {
        m: float(r) for (_, m), r in zip(vc['prunable_modules'], ratios)
    }

    pruner_kwargs = {
        'importance': tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
        'pruning_ratio': 0,
        'pruning_ratio_dict': mlp_ratio_dict,
    }
    pruner_kwargs.update(p._build_metapruner_kwargs(vc))
    pruner = tp.pruner.MetaPruner(model, example_inputs, **pruner_kwargs)
    pruner.step()

    after = param_count(model)
    print(f"  Params: {orig_params:,} -> {after:,} ({1 - after/orig_params:.1%} reduction)")

    # Verify embed_dim unchanged (qkv input and proj output should be 384)
    for _, mod in model.named_modules():
        if hasattr(mod, 'qkv'):
            assert_eq(mod.qkv.in_features, 384, "qkv input (embed_dim) should be preserved")
        if hasattr(mod, 'proj') and hasattr(mod, 'qkv'):
            assert_eq(mod.proj.out_features, 384, "proj output (embed_dim) should be preserved")

    out = forward_check(model, example_inputs, "after MLP-only pruning")
    assert_eq(out, torch.Size([1, 10]))
    print(f"  embed_dim preserved: OK")


# ---------------------------------------------------------------------------
# 6. CWP_Pruning simulation (mimics the full compress() flow)
# ---------------------------------------------------------------------------

def test_cwp_pruning_timm():
    """Simulate CWP_Pruning with a pre-set sparsity_dict on timm."""
    import timm
    from sconce.pruner import prune

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
    model.eval()
    p = prune()
    p.attention_heads = True

    vc = p._detect_vit_config(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)

    # Build a fake sparsity_dict: 0.3 for each MLP fc1, 0.25 for each attention
    p.sparsity_dict = {}
    for name, _ in vc['prunable_modules']:
        p.sparsity_dict[name] = 0.3
    for name, _ in vc['attention_modules']:
        p.sparsity_dict[name] = 0.25

    # Mock dataloader for CWP_Pruning
    class FakeLoader:
        def __iter__(self):
            return iter([(torch.randn(2, 3, 224, 224), torch.tensor([0, 1]))])
    p.model = model
    p.dataloader = {'test': FakeLoader()}

    p.CWP_Pruning()

    after = param_count(p.model)
    print(f"  Params: {orig_params:,} -> {after:,} ({1 - after/orig_params:.1%} reduction)")

    out = forward_check(p.model, example_inputs, "after CWP_Pruning")
    assert_eq(out, torch.Size([1, 10]))


# ---------------------------------------------------------------------------
# 7. Patched timm forward correctness
# ---------------------------------------------------------------------------

def test_patched_timm_forward_equivalence():
    """Verify patched forward produces same output as original on unpruned model."""
    import timm
    from sconce.pruner import prune

    model = timm.create_model('vit_small_patch16_224', pretrained=False, num_classes=10)
    model.eval()

    example_inputs = torch.randn(1, 3, 224, 224)

    with torch.no_grad():
        original_out = model(example_inputs).clone()

    p = prune()
    p._patch_timm_attention(model)

    with torch.no_grad():
        patched_out = model(example_inputs)

    diff = (original_out - patched_out).abs().max().item()
    print(f"  Max absolute difference: {diff:.2e}")
    assert_true(diff < 1e-5, f"Patched forward diverges: max diff = {diff}")


# ---------------------------------------------------------------------------
# 8. Non-ViT model backward compat (Conv2d-only path should still work)
# ---------------------------------------------------------------------------

def test_detect_config_non_vit():
    """_detect_vit_config on a plain CNN should return framework=None."""
    from sconce.pruner import prune
    import torchvision

    model = torchvision.models.resnet18(weights=None)
    p = prune()
    p.attention_heads = True
    vc = p._detect_vit_config(model)

    assert_true(vc['framework'] is None, f"Expected None, got {vc['framework']}")
    assert_true(vc['num_heads'] is None)
    assert_true(len(vc['attention_modules']) == 0)
    # prunable_modules should have Conv2d and Linear layers
    has_conv = any(isinstance(m, nn.Conv2d) for _, m in vc['prunable_modules'])
    assert_true(has_conv, "Should find Conv2d in ResNet")
    print(f"  framework=None, {len(vc['prunable_modules'])} prunable modules: OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("  ViT Pruning Smoke Tests")
    print("=" * 70)

    # Config detection
    run_test("1a. _detect_vit_config — timm", test_detect_config_timm)
    run_test("1b. _detect_vit_config — torchvision", test_detect_config_torchvision)
    run_test("1c. _detect_vit_config — HuggingFace", test_detect_config_huggingface)
    run_test("1d. _detect_vit_config — non-ViT (ResNet)", test_detect_config_non_vit)

    # Head importance
    run_test("2a. Head importance — fused QKV", test_head_importance_fused_qkv)
    run_test("2b. Head importance — separate Q/K/V", test_head_importance_separate_qkv)

    # Standalone pruning paths
    run_test("3a. MLP-only MetaPruner — timm", test_mlp_only_timm)
    run_test("3b. DG head pruning standalone — timm", test_dg_head_pruning_timm_standalone)
    run_test("3c. DG head pruning standalone — torchvision", test_dg_head_pruning_torchvision_standalone)

    # Two-path combined
    run_test("4a. Two-path (MLP + heads) — timm", test_two_path_timm)
    run_test("4b. Two-path (MLP + heads) — torchvision", test_two_path_torchvision)
    run_test("4c. Two-path (MLP + heads) — HuggingFace", test_two_path_huggingface)

    # CWP_Pruning simulation
    run_test("5.  CWP_Pruning end-to-end — timm", test_cwp_pruning_timm)

    # Auxiliary
    run_test("6.  Patched timm forward equivalence", test_patched_timm_forward_equivalence)

    # Summary
    total = PASS + FAIL + SKIP
    print(f"\n{'='*70}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {SKIP} skipped  (total: {total})")
    print(f"{'='*70}")

    sys.exit(1 if FAIL > 0 else 0)
