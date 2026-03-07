"""
HuggingFace ViT pruning tests for sconce.

Focused tests for HuggingFace Vision Transformer models including:
  1. Config detection across ViT variants (ViTModel, ViTForImageClassification)
  2. Separate Q/K/V head importance computation
  3. DG-based head pruning standalone
  4. MLP-only MetaPruner pruning
  5. Two-path combined (MLP + heads) pruning
  6. CWP_Pruning end-to-end simulation
  7. Embed_dim preservation after pruning
  8. Per-layer head pruning with varying sparsities
  9. Edge cases: single head prune, max sparsity clamp

Run:
    python tests/test_hf_vit_pruning.py
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
# Check HuggingFace availability
# ---------------------------------------------------------------------------

try:
    from transformers import ViTModel, ViTConfig, ViTForImageClassification
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

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
    if not HF_AVAILABLE:
        SKIP += 1
        print(f"  >> SKIPPED (transformers not installed)")
        return
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


def param_count(model):
    return sum(p.numel() for p in model.parameters())


def forward_check(model, example_inputs, label=""):
    """Run a forward pass and return output. Raises on failure."""
    with torch.no_grad():
        out = model(example_inputs)
    if hasattr(out, 'last_hidden_state'):
        print(f"  Forward {label}: last_hidden_state={out.last_hidden_state.shape}")
        return out.last_hidden_state.shape
    elif hasattr(out, 'logits'):
        print(f"  Forward {label}: logits={out.logits.shape}")
        return out.logits.shape
    else:
        print(f"  Forward {label}: output shape = {out.shape}")
        return out.shape


def make_small_hf_vit(num_classes=None):
    """Create a small HF ViT for fast testing."""
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


# ---------------------------------------------------------------------------
# 1. Config detection — ViTModel (no classifier head)
# ---------------------------------------------------------------------------

def test_detect_config_vit_model():
    from sconce.pruner import prune

    model = make_small_hf_vit()
    p = prune()
    p.attention_heads = True
    vc = p._detect_vit_config(model)

    assert_eq(vc['framework'], 'huggingface')

    # prunable_modules: only intermediate.dense (MLP fc1)
    assert_eq(len(vc['prunable_modules']), 4, "Should have 4 prunable (one per layer)")
    for name, mod in vc['prunable_modules']:
        assert_true('intermediate.dense' in name, f"Unexpected prunable: {name}")
        assert_true(isinstance(mod, nn.Linear))
        assert_eq(mod.in_features, 384, "fc1 input should be hidden_size=384")
        assert_eq(mod.out_features, 1536, "fc1 output should be intermediate_size=1536")
    print(f"  Prunable modules: {len(vc['prunable_modules'])} (all intermediate.dense)")

    # attention_modules: 4 ViTSelfAttention modules
    assert_eq(len(vc['attention_modules']), 4)
    for name, mod in vc['attention_modules']:
        assert_true(hasattr(mod, 'query'))
        assert_true(hasattr(mod, 'key'))
        assert_true(hasattr(mod, 'value'))
        assert_eq(mod.num_attention_heads, 6)
    print(f"  Attention modules: {len(vc['attention_modules'])}")

    # num_heads: maps query, key, value for each layer = 4*3 = 12
    assert_true(vc['num_heads'] is not None)
    assert_eq(len(vc['num_heads']), 12, "Should have 12 num_heads entries (3 per layer)")
    print(f"  num_heads entries: {len(vc['num_heads'])}")

    # unwrapped_parameters: position_embeddings (no cls_token param directly in HF ViT)
    assert_true(vc['unwrapped_parameters'] is not None)
    print(f"  Unwrapped params: {len(vc['unwrapped_parameters'])}")

    # Excluded modules: query, key, value, output.dense should NOT be prunable
    prunable_names = set(n for n, _ in vc['prunable_modules'])
    all_names = set(n for n, _ in model.named_modules())
    for n in all_names:
        if n.endswith('.query') or n.endswith('.key') or n.endswith('.value'):
            assert_true(n not in prunable_names, f"{n} should not be prunable")
        if n.endswith('.output.dense'):
            assert_true(n not in prunable_names, f"{n} should not be prunable")
    print(f"  Q/K/V/output.dense excluded from prunable: OK")


# ---------------------------------------------------------------------------
# 2. Config detection — ViTForImageClassification (has classifier head)
# ---------------------------------------------------------------------------

def test_detect_config_vit_for_classification():
    from sconce.pruner import prune

    model = make_small_hf_vit(num_classes=10)
    p = prune()
    p.attention_heads = True
    vc = p._detect_vit_config(model)

    assert_eq(vc['framework'], 'huggingface')

    # Should have classifier in ignored_layers
    assert_true(len(vc['ignored_layers']) >= 1, "Should ignore classifier head")
    classifier_found = False
    for mod in vc['ignored_layers']:
        if isinstance(mod, nn.Linear) and mod.out_features == 10:
            classifier_found = True
    assert_true(classifier_found, "Classifier (out_features=10) should be in ignored_layers")
    print(f"  Ignored layers: {len(vc['ignored_layers'])} (classifier found)")

    # Classifier should NOT be in prunable_modules
    for name, mod in vc['prunable_modules']:
        assert_true(mod.out_features != 10,
                    f"Classifier should not be in prunable_modules: {name}")
    print(f"  Classifier excluded from prunable: OK")

    # Everything else same as ViTModel
    assert_eq(len(vc['prunable_modules']), 4)
    assert_eq(len(vc['attention_modules']), 4)
    print(f"  Prunable: {len(vc['prunable_modules'])}, Attention: {len(vc['attention_modules'])}")


# ---------------------------------------------------------------------------
# 3. Head importance — separate Q/K/V
# ---------------------------------------------------------------------------

def test_head_importance_separate_qkv():
    from sconce.pruner import prune
    p = prune()

    num_heads = 6
    head_dim = 64
    embed_dim = 384

    # Create Q, K, V weight tensors with known structure
    q_w = torch.randn(num_heads * head_dim, embed_dim)
    k_w = torch.randn(num_heads * head_dim, embed_dim)
    v_w = torch.randn(num_heads * head_dim, embed_dim)

    importance = p.get_attention_head_importance([q_w, k_w, v_w], num_heads)
    assert_eq(importance.shape, (num_heads,))
    assert_true(torch.all(importance >= 0), "Importance should be non-negative")
    print(f"  Shape: {importance.shape}, min={importance.min():.3f}, max={importance.max():.3f}")

    # Test that zeroing one head's weights in all Q/K/V makes it least important
    q_w_zeroed = q_w.clone()
    k_w_zeroed = k_w.clone()
    v_w_zeroed = v_w.clone()
    target_head = 2
    q_w_zeroed[target_head * head_dim:(target_head + 1) * head_dim, :] = 0
    k_w_zeroed[target_head * head_dim:(target_head + 1) * head_dim, :] = 0
    v_w_zeroed[target_head * head_dim:(target_head + 1) * head_dim, :] = 0

    importance_zeroed = p.get_attention_head_importance(
        [q_w_zeroed, k_w_zeroed, v_w_zeroed], num_heads
    )
    least_important = torch.argmin(importance_zeroed).item()
    assert_eq(least_important, target_head,
              f"Head {target_head} was zeroed but head {least_important} is least important")
    print(f"  Zeroed head {target_head} correctly identified as least important: OK")


# ---------------------------------------------------------------------------
# 4. DG head pruning standalone — HuggingFace
# ---------------------------------------------------------------------------

def test_dg_head_pruning_standalone():
    from sconce.pruner import prune

    model = make_small_hf_vit()
    model.eval()
    p = prune()
    p.attention_heads = True

    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)

    vc = p._detect_vit_config(model)

    # Prune 2 heads (out of 6) from every attention block
    for aname, attn_mod in vc['attention_modules']:
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            sparsity=2/6, framework='huggingface', vit_config=vc,
        )

    after = param_count(model)
    print(f"  Params: {orig_params:,} -> {after:,} ({1 - after/orig_params:.1%} reduction)")

    out_shape = forward_check(model, example_inputs, "after head-only pruning")
    assert_eq(out_shape[0], 1, "Batch size should be 1")
    assert_eq(out_shape[2], 384, "Embed dim should be preserved (384)")

    # Verify head count reduced on all attention modules
    for _, mod in model.named_modules():
        if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
            assert_eq(mod.num_attention_heads, 4,
                      f"Should have 4 heads after pruning 2/6, got {mod.num_attention_heads}")
            # Verify Q/K/V dimensions consistent
            expected_out = 4 * 64  # 4 heads * 64 head_dim = 256
            assert_eq(mod.query.out_features, expected_out,
                      f"query.out_features should be {expected_out}")
            assert_eq(mod.key.out_features, expected_out)
            assert_eq(mod.value.out_features, expected_out)
    print(f"  All attention modules: 4 heads, Q/K/V dims consistent: OK")


# ---------------------------------------------------------------------------
# 5. MLP-only MetaPruner — HuggingFace
# ---------------------------------------------------------------------------

def test_mlp_only_pruning():
    import torch_pruning as tp
    from sconce.pruner import prune

    model = make_small_hf_vit()
    model.eval()
    p = prune()
    p.attention_heads = True

    vc = p._detect_vit_config(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)

    # Varying ratios per block
    ratios = np.linspace(0.1, 0.4, len(vc['prunable_modules']))
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

    # Verify embed_dim preserved: query input and output.dense output should still be 384
    for _, mod in model.named_modules():
        if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
            assert_eq(mod.query.in_features, 384,
                      "query input (embed_dim) should be preserved")
    for name, mod in model.named_modules():
        if name.endswith('.output.dense') and isinstance(mod, nn.Linear):
            assert_eq(mod.out_features, 384,
                      f"{name} output (embed_dim) should be preserved")
    print(f"  embed_dim preserved at 384: OK")

    out_shape = forward_check(model, example_inputs, "after MLP-only pruning")
    assert_eq(out_shape[2], 384, "Hidden dim should remain 384")

    # Verify intermediate sizes were actually reduced
    for name, mod in model.named_modules():
        if 'intermediate.dense' in name and isinstance(mod, nn.Linear):
            assert_true(mod.out_features < 1536,
                        f"{name}: intermediate should be reduced from 1536, got {mod.out_features}")
    print(f"  Intermediate sizes reduced: OK")


# ---------------------------------------------------------------------------
# 6. Two-path combined (MLP + heads) — HuggingFace
# ---------------------------------------------------------------------------

def test_two_path_combined():
    import torch_pruning as tp
    from sconce.pruner import prune

    model = make_small_hf_vit()
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

    # Step 2: Head pruning via DG
    current_vc = p._detect_vit_config(model)
    for aname, attn_mod in current_vc['attention_modules']:
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            0.25, 'huggingface', current_vc,
        )

    after_heads = param_count(model)
    print(f"  After head pruning: {after_heads:,} ({1 - after_heads/orig_params:.1%} total reduction)")

    # Verify head count reduced
    for _, mod in model.named_modules():
        if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
            # 6 heads * 0.25 = 1.5, rounded to 2 pruned -> 4 remaining
            assert_true(mod.num_attention_heads < 6,
                        f"Heads should be reduced, got {mod.num_attention_heads}")
            break

    out_shape = forward_check(model, example_inputs, "after two-path pruning")
    assert_eq(out_shape[0], 1)
    assert_eq(out_shape[2], 384, "embed_dim should be preserved")


# ---------------------------------------------------------------------------
# 7. CWP_Pruning end-to-end — HuggingFace ViTForImageClassification
# ---------------------------------------------------------------------------

def test_cwp_pruning_end_to_end():
    from sconce.pruner import prune

    model = make_small_hf_vit(num_classes=10)
    model.eval()
    p = prune()
    p.attention_heads = True

    vc = p._detect_vit_config(model)
    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)

    # Build a sparsity_dict: 0.3 for MLP, 0.25 for attention
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

    out_shape = forward_check(p.model, example_inputs, "after CWP_Pruning")
    assert_eq(out_shape, torch.Size([1, 10]), "Should output [1, num_classes]")

    # Verify model structure is valid
    for _, mod in p.model.named_modules():
        if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
            assert_true(mod.num_attention_heads < 6,
                        f"Heads should be reduced from 6, got {mod.num_attention_heads}")
            # Q/K/V output dims should be consistent
            assert_eq(mod.query.out_features, mod.key.out_features)
            assert_eq(mod.query.out_features, mod.value.out_features)
            break
    print(f"  Model structure valid after CWP_Pruning: OK")


# ---------------------------------------------------------------------------
# 8. Per-layer varying head sparsity
# ---------------------------------------------------------------------------

def test_varying_head_sparsity_per_layer():
    """Prune different numbers of heads per attention layer."""
    from sconce.pruner import prune

    model = make_small_hf_vit()
    model.eval()
    p = prune()
    p.attention_heads = True

    example_inputs = torch.randn(1, 3, 224, 224)
    orig_params = param_count(model)

    vc = p._detect_vit_config(model)

    # Varying sparsities: layer 0 -> 1/6, layer 1 -> 2/6, layer 2 -> 3/6, layer 3 -> 1/6
    sparsities = [1/6, 2/6, 3/6, 1/6]
    expected_remaining = [5, 4, 3, 5]

    for (aname, attn_mod), sp in zip(vc['attention_modules'], sparsities):
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            sp, 'huggingface', vc,
        )

    after = param_count(model)
    print(f"  Params: {orig_params:,} -> {after:,} ({1 - after/orig_params:.1%} reduction)")

    # Verify each layer has the expected head count
    attn_modules = []
    for name, mod in model.named_modules():
        if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
            attn_modules.append((name, mod))

    for i, ((name, mod), expected) in enumerate(zip(attn_modules, expected_remaining)):
        assert_eq(mod.num_attention_heads, expected,
                  f"Layer {i} ({name}): expected {expected} heads, got {mod.num_attention_heads}")
        print(f"  Layer {i}: {mod.num_attention_heads} heads (expected {expected})")

    out_shape = forward_check(model, example_inputs, "after per-layer head pruning")
    assert_eq(out_shape[2], 384, "embed_dim should be preserved")


# ---------------------------------------------------------------------------
# 9. Edge case: prune exactly 1 head
# ---------------------------------------------------------------------------

def test_single_head_prune():
    """Prune exactly 1 head from each attention layer."""
    from sconce.pruner import prune

    model = make_small_hf_vit()
    model.eval()
    p = prune()
    p.attention_heads = True

    example_inputs = torch.randn(1, 3, 224, 224)
    vc = p._detect_vit_config(model)

    # Very small sparsity — should remove exactly 1 head
    # max(1, round(6 * 0.1)) = max(1, 1) = 1
    for aname, attn_mod in vc['attention_modules']:
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            sparsity=0.1, framework='huggingface', vit_config=vc,
        )

    for _, mod in model.named_modules():
        if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
            assert_eq(mod.num_attention_heads, 5,
                      f"Should have 5 heads after pruning 1, got {mod.num_attention_heads}")
            assert_eq(mod.query.out_features, 5 * 64,
                      f"query.out_features should be 320")
            break

    out_shape = forward_check(model, example_inputs, "after single-head pruning")
    assert_eq(out_shape[2], 384, "embed_dim should be preserved")
    print(f"  Single head pruned correctly: 6 -> 5 heads")


# ---------------------------------------------------------------------------
# 10. Edge case: high sparsity clamp (don't remove all heads)
# ---------------------------------------------------------------------------

def test_max_sparsity_clamp():
    """Sparsity of 1.0 should not remove ALL heads — at least 1 must remain."""
    from sconce.pruner import prune

    model = make_small_hf_vit()
    model.eval()
    p = prune()
    p.attention_heads = True

    example_inputs = torch.randn(1, 3, 224, 224)
    vc = p._detect_vit_config(model)

    # Only prune first attention layer with extreme sparsity
    aname, attn_mod = vc['attention_modules'][0]
    p._prune_attention_heads(
        model, example_inputs, attn_mod, aname,
        sparsity=0.99, framework='huggingface', vit_config=vc,
    )

    # Should have at least 1 head remaining
    for _, mod in model.named_modules():
        if hasattr(mod, 'query') and hasattr(mod, 'num_attention_heads'):
            assert_true(mod.num_attention_heads >= 1,
                        f"Should have at least 1 head, got {mod.num_attention_heads}")
            print(f"  Heads remaining after 0.99 sparsity: {mod.num_attention_heads}")
            break

    out_shape = forward_check(model, example_inputs, "after max-sparsity pruning")
    assert_eq(out_shape[2], 384, "embed_dim should be preserved")


# ---------------------------------------------------------------------------
# 11. Embed dim preservation check — deeper inspection
# ---------------------------------------------------------------------------

def test_embed_dim_preserved_thoroughly():
    """After head pruning, verify embed_dim is preserved throughout the model."""
    from sconce.pruner import prune

    model = make_small_hf_vit()
    model.eval()
    p = prune()
    p.attention_heads = True

    example_inputs = torch.randn(1, 3, 224, 224)
    vc = p._detect_vit_config(model)

    # Prune half the heads
    for aname, attn_mod in vc['attention_modules']:
        p._prune_attention_heads(
            model, example_inputs, attn_mod, aname,
            sparsity=0.5, framework='huggingface', vit_config=vc,
        )

    embed_dim = 384

    # Check position_embeddings dimension (HF uses raw nn.Parameter, not nn.Embedding)
    for name, mod in model.named_modules():
        if hasattr(mod, 'position_embeddings'):
            pe = mod.position_embeddings
            # nn.Parameter with shape (1, num_patches+1, hidden_size)
            assert_eq(pe.shape[-1], embed_dim,
                      f"position_embeddings last dim should be {embed_dim}")
            print(f"  position_embeddings shape: {pe.shape}")

    # Check cls_token dimension
    for name, mod in model.named_modules():
        if hasattr(mod, 'cls_token'):
            ct = mod.cls_token
            assert_eq(ct.shape[-1], embed_dim,
                      f"cls_token last dim should be {embed_dim}")
            print(f"  cls_token shape: {ct.shape}")

    # Check output.dense layers (MLP fc2 equivalent) output embed_dim
    for name, mod in model.named_modules():
        if name.endswith('.output.dense') and isinstance(mod, nn.Linear):
            assert_eq(mod.out_features, embed_dim,
                      f"{name} should output embed_dim={embed_dim}, got {mod.out_features}")

    # Check intermediate.dense layers input = embed_dim
    for name, mod in model.named_modules():
        if 'intermediate.dense' in name and isinstance(mod, nn.Linear):
            assert_eq(mod.in_features, embed_dim,
                      f"{name} input should be embed_dim={embed_dim}, got {mod.in_features}")

    out_shape = forward_check(model, example_inputs, "after 50% head pruning")
    assert_eq(out_shape[2], embed_dim, f"Output hidden dim should be {embed_dim}")
    print(f"  All embed_dim checks passed at {embed_dim}")


# ---------------------------------------------------------------------------
# 12. Deepcopy + prune — verify original is untouched
# ---------------------------------------------------------------------------

def test_deepcopy_isolation():
    """Verify pruning a deepcopy doesn't affect the original model."""
    from sconce.pruner import prune

    model = make_small_hf_vit()
    model.eval()
    orig_params = param_count(model)
    orig_head_count = None
    for _, mod in model.named_modules():
        if hasattr(mod, 'num_attention_heads'):
            orig_head_count = mod.num_attention_heads
            break

    # Deepcopy and prune the copy
    model_copy = copy.deepcopy(model)
    p = prune()
    p.attention_heads = True
    example_inputs = torch.randn(1, 3, 224, 224)
    vc = p._detect_vit_config(model_copy)

    for aname, attn_mod in vc['attention_modules']:
        p._prune_attention_heads(
            model_copy, example_inputs, attn_mod, aname,
            sparsity=0.5, framework='huggingface', vit_config=vc,
        )

    # Verify copy was pruned
    copy_params = param_count(model_copy)
    assert_true(copy_params < orig_params, "Copy should have fewer params after pruning")
    print(f"  Copy params: {orig_params:,} -> {copy_params:,}")

    # Verify original is untouched
    assert_eq(param_count(model), orig_params, "Original param count should be unchanged")
    for _, mod in model.named_modules():
        if hasattr(mod, 'num_attention_heads'):
            assert_eq(mod.num_attention_heads, orig_head_count,
                      "Original head count should be unchanged")
            break
    print(f"  Original model unchanged: OK")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import warnings
    warnings.filterwarnings("ignore")

    print("=" * 70)
    print("  HuggingFace ViT Pruning Tests")
    print("=" * 70)

    # Config detection
    run_test("1.  Config detection — ViTModel", test_detect_config_vit_model)
    run_test("2.  Config detection — ViTForImageClassification", test_detect_config_vit_for_classification)

    # Head importance
    run_test("3.  Head importance — separate Q/K/V with zeroed head", test_head_importance_separate_qkv)

    # Standalone pruning paths
    run_test("4.  DG head pruning standalone", test_dg_head_pruning_standalone)
    run_test("5.  MLP-only MetaPruner", test_mlp_only_pruning)

    # Combined
    run_test("6.  Two-path combined (MLP + heads)", test_two_path_combined)
    run_test("7.  CWP_Pruning end-to-end — ViTForImageClassification", test_cwp_pruning_end_to_end)

    # Per-layer and edge cases
    run_test("8.  Per-layer varying head sparsity", test_varying_head_sparsity_per_layer)
    run_test("9.  Edge case — single head prune", test_single_head_prune)
    run_test("10. Edge case — max sparsity clamp", test_max_sparsity_clamp)

    # Structural integrity
    run_test("11. Embed dim preservation (thorough)", test_embed_dim_preserved_thoroughly)
    run_test("12. Deepcopy isolation", test_deepcopy_isolation)

    # Summary
    total = PASS + FAIL + SKIP
    print(f"\n{'='*70}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {SKIP} skipped  (total: {total})")
    print(f"{'='*70}")

    sys.exit(1 if FAIL > 0 else 0)
