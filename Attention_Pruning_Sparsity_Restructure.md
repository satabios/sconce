# Attention Pruning & Sparsity Restructure Plan

**Goal:** enable robust, direct activation-based structural pruning for ViT- and LLM-class
HuggingFace models (per the "Direct Activation-Based Pruning for Sconce (No NAS)" RFC),
by unifying the divergent branch implementations and filling the verified gaps.

**Date:** 2026-07-27 · **Base branch:** `experiment/apr14` · **Env verified against:**
torch 2.5.1, transformers **5.2.0**, torch_pruning installed.

---

## 1. Current State of the Repo (branch survey)

| Branch | What it holds | Status |
|---|---|---|
| `experiment/apr14` (current) | `sconce/pruner.py` with **`TransformerPruner`** — name-registry-based structural pruning (fused/separate QKV, GQA, SwiGLU, sensitivity scan); `scripts/run_llm_experiment.py` with standalone Qwen2 depth/FFN/head pruning + int8/fp16/nf4 quant + finetune; `scripts/run_sensitivity_scan.py`; `scripts/run_experiment.py` (CNN) | Most advanced LLM path; ViT support broken (see gaps) |
| `ViT` | Completely different `sconce/pruner.py` built on **torch_pruning DependencyGraph**: `_detect_vit_config` (timm / HF ViT / torchvision), HF nested-attention head pruning, **timm forward patch** (`-1` reshape), MetaPruner MLP path, plus ~2,200 lines of tests (`test_hf_vit_pruning.py`, `test_internvit_pruning.py`, `test_vit_pruning.py`, `test_cnn_pruning.py`, `conftest.py`) | Best ViT coverage; no LLM/GQA/SwiGLU support; drops `TransformerPruner` entirely |
| `origin/attention` | Older iteration of the DG/unwrapped-parameter detection work | Superseded by `ViT` |
| `main` / `paper` | Legacy sconce (CNN GMP/CWP + venum) | Baseline only |

**The two lines never merged.** `experiment/apr14` is LLM-first (weight surgery via name
registries); `ViT` is vision-first (graph tracing via torch_pruning). The restructure
must combine them under one API.

---

## 2. Verified Gaps (empirically confirmed with a smoke test, transformers 5.2.0)

### G1 — HF ViT/BERT-family detection fails completely ❌
`TransformerPruner._find_transformer_layers(ViTModel(...))` → **0 blocks**
("could not parse attention projections"). Cause: HF vision/BERT models nest projections
two levels deep — `block.attention.attention.query` and `block.attention.output.dense` —
but `_build_attn_spec` only does flat `getattr(attn_module, name)` lookups. Same failure
for DeiT, BEiT, BERT, RoBERTa. The `["encoder","layer"]` entry in `BLOCK_PATH_SEQUENCES`
is dead code today.

### G2 — Pruned GQA LLM cannot be reloaded (ConfigUpdater incomplete) ❌
Qwen2 tiny GQA model: detection, head/FFN prune, and forward all **work** on
transformers 5.x (`num_key_value_groups` module attr is updated correctly). But
`save_pretrained` → `from_pretrained` **fails with size mismatches on every attention
tensor**. `_update_model_config` writes `num_attention_heads` but not `head_dim`, so
reload recomputes `head_dim = hidden_size // new_num_heads` (fractional/wrong).
Also: heterogeneous per-layer head counts are collapsed with `min(...)` (lossy),
`num_hidden_layers` is never touched (no depth pruning in the library), and
`intermediate_size` gets the same `min()` treatment.

### G3 — timm ViT forward crashes after head pruning ❌
`TransformerPruner` prunes timm `vit_tiny` fine, but forward raises
`shape '[1,197,192]' invalid for input of size 25216`: timm's attention reshapes the
output with literal `C = embed_dim`, which no longer equals `num_heads*head_dim`.
The fix (`_patched_timm_attn_forward` with `reshape(B, N, -1)`) already exists **on the
ViT branch only** and must be ported.

### G4 — GPT-2 / fused-Conv1D models: 0 blocks detected ❌
`c_attn` is `transformers.Conv1D`, not `nn.Linear`; `_find_linear_by_names` skips it.
Fused-QKV handling also assumes `out_features // 3` — wrong for GQA-fused models
(Phi-3 `qkv_proj` packs `q_dim + 2*kv_dim`).

### G5 — No activation-based importance anywhere (core of the RFC) ❌
All scoring is **weight L2 magnitude** (`out_proj` columns for heads, `down_proj`
rows/cols for FFN). There is no calibration forward pass, no hooks, no
`SensitivityHookManager`, no `ActivationPruner`. The existing
`sensitivity_scan` is eval-accuracy-driven (n_blocks × n_sparsities × 2 full
evaluations — exactly the NAS-ish loop the RFC removes) and its plan-builder semantics
assume "accuracy %" (sign conventions break for perplexity).

### G6 — Depth pruning exists only as a script hack ❌
`prune_depth()` lives in `scripts/run_llm_experiment.py`, hardcoded to
`model.model.layers`, selects layers **uniformly spaced** (not by importance), and does
not renumber `layer_idx` on surviving decoder layers (KV-cache indexing gets gaps).
Nothing in the `sconce` package does depth pruning.

### G7 — No unstructured→structured re-packing (`SparsityPacker`) ❌
GMP produces zero-masks (`self.masks`) that are never physically sliced. No dead-neuron
discovery, no coupled gate/up slicing, no config sync. RFC Phase 3 is entirely missing.

### G8 — No knowledge distillation module ❌
`finetune()` in the LLM script is plain causal-LM cross-entropy (commit `feb07ae`'s
"knowledge distillation finetune" message notwithstanding). No `sconce.distillation.Distiller`,
no teacher-student KL loss.

### G9 — No calibration-data helper ❌
RFC API references `sconce.data.get_calibration_dataset(...)`; nothing exists. The LLM
script has WikiText-2 loaders that should be generalized.

### Smaller, concrete defects found while reading
- `pruner.py:1077-1084` (CNN `sensitivity_scan`): `accuracy` list is misaligned with
  `sparsities` (`np.argmax(accuracy)` indexes the wrong sparsity), and the first
  failing accuracy is never appended.
- `GMP_Pruning` (`pruner.py:1295`): `KeyError` if `sparsity_dict` lacks any dim>1 param.
- `OUT_PROJ_NAMES` contains `"dense"` and `FFN_UP_NAMES` contains `"dense"` — ambiguous
  for BERT-family once nesting is fixed; ordering must be per-architecture.
- `_build_sparsity_plan` hardcodes `-0.60` accuracy-point fallback; not metric-aware.
- KV-head pruning is unsupported (only Q heads are ever cut) — fine as a default
  (`lock_head_dim=True` semantics) but should be an explicit, documented constraint.
- `queue` imported but unused in `pruner.py`.

---

## 3. Target Architecture

Single package layout (all under `sconce/`), mapping RFC modules to code:

```
sconce/
  pruning/
    __init__.py            # ActivationPruner, SparsityPacker, TransformerPruner re-exports
    activation_pruner.py   # NEW — RFC orchestrator (calibration → score → slice)
    hooks.py               # NEW — SensitivityHookManager (L1/L2 activation norms)
    slicer.py              # MOVED/EXTENDED — GQA-aware head/FFN/depth weight surgery
    packer.py              # NEW — SparsityPacker (unstructured → structured)
    registry.py            # name registries + per-architecture adapters (nested paths)
    config_sync.py         # NEW — ConfigUpdater (head_dim, num_hidden_layers, ...)
  distillation/
    __init__.py            # Distiller (logit KD + optional hidden-state MSE)
  data/
    calibration.py         # get_calibration_dataset("wikitext2" | "c4" | callable)
  pruner.py                # kept as a compat shim (CNN mixin + re-exports)
```

### `ActivationPruner` (the RFC front door)

```python
pruner = sconce.pruning.ActivationPruner(model, calibration_data)
pruner.compute_importance_scores(num_batches=…)   # one calibration pass, hooks on:
                                                  #   • block outputs   → layer_importance (depth)
                                                  #   • attn out_proj in → head_importance (width)
                                                  #   • FFN activation  → neuron_importance (width)
pruned = pruner.prune(depth_prune_ratio=0.2, width_prune_ratio=0.15, lock_head_dim=True)
pruned.save_pretrained(...)                       # must round-trip (G2 fix)
```

Design decisions:
1. **One calibration pass, all hooks at once** (block, head, neuron). Cheap; avoids
   re-running for depth vs width.
2. **Depth importance:** cosine-similarity between block input and output
   (redundancy = high similarity ⇒ low importance), the standard for layer dropping —
   plus the RFC's plain L2-of-output as an option. Falls back to weight heuristics when
   no calibration data is given.
3. **Head importance:** L2 norm of the per-head slice of the input to `out_proj`
   (activation-weighted), aggregated over calibration batches. GQA: heads ranked within
   each KV group; equal removal per group (reuse `_select_heads_gqa`).
4. **Neuron importance:** mean L2 of the activation entering `down_proj`
   (captures `silu(gate)·up` jointly for SwiGLU).
5. `lock_head_dim=True`: only whole Q-heads are cut, KV heads intact (current behavior,
   made explicit). `False` reserved for future KV-head cutting.
6. Depth prune runs **before** width prune (widths scored on the surviving stack).

---

## 4. Implementation Phases

### Phase 0 — Merge the two worlds & fix breakage (prereq, ~1 PR)
1. **Port from `ViT` branch into `experiment/apr14`:**
   - `_patched_timm_attn_forward` + `_patch_timm_attention` (fixes **G3**; apply
     automatically after any timm head prune).
   - Nested HF attention adapter: teach `_build_attn_spec` to descend
     `attention.attention.{query,key,value}` / `attention.output.dense` and
     `intermediate.dense` + `output.dense` for the FFN (fixes **G1** for
     ViT/DeiT/BEiT/BERT/RoBERTa). Implement as a small per-architecture adapter table in
     `registry.py` rather than more flat-name guessing.
   - The 4 test files + `conftest.py` (they are the only real test suite in the repo);
     adapt their imports to the unified API.
2. **Conv1D support:** wrap `transformers.Conv1D` (weight is transposed vs `nn.Linear`)
   in `_find_linear_by_names` / `_rebuild_linear` (fixes **G4** for GPT-2).
   GQA-fused `qkv_proj` (Phi-3): parse split sizes from config
   (`num_attention_heads`, `num_key_value_heads`, `head_dim`) instead of `//3`.
3. **Fix the small defects list** (accuracy/sparsity misalignment, `GMP_Pruning`
   KeyError, unused import).
4. Keep `TransformerPruner` API intact; everything lands as internal fixes.

**Acceptance:** smoke test passes for HF ViT (blocks found, prune, forward), timm ViT
(forward after prune), GPT-2 (blocks found), Qwen2 (unchanged). ViT-branch pytest suite
green against the unified pruner.

### Phase 1 — ConfigUpdater + depth pruning in the library
1. `config_sync.py` (**fixes G2**):
   - Always write `config.head_dim` explicitly when heads change (transformers ≥4.44
     honors it for Llama/Qwen-family; for configs that reject it, emit a
     hard error instead of a silent-corrupt save).
   - Write `num_hidden_layers` after depth prune; `intermediate_size` after FFN prune.
   - **Uniform-plan enforcement:** for `save_pretrained` round-trips, offer
     `uniform=True` (same ratio every layer — HF-config representable). For
     heterogeneous plans keep the model in-memory/state-dict workflows and warn once,
     precisely (replaces the current `min()` corruption).
   - Acceptance test: prune → save → `from_pretrained` → identical logits (fp32 tolerance).
2. **Depth pruning moves into the library** (`slicer.py`, fixes **G6**):
   - `prune_depth(model, ratio | keep_idx)` using `_find_block_container` (works for
     `model.layers`, `transformer.h`, `blocks`, `encoder.layer`).
   - Renumber `layer_idx` on surviving decoder layers (`self_attn.layer_idx`, and the
     layer module's own `layer_idx`/`attention_type` bookkeeping in transformers 5.x).
   - Importance-driven selection (from Phase 2 scores) with the uniform-spacing
     heuristic as the no-data fallback; never drop first/last block by default.
3. Migrate `scripts/run_llm_experiment.py` to call the library versions (its
   `prune_ffn_width`/`prune_depth`/`prune_attention_heads` become thin wrappers, then
   get deleted).

### Phase 2 — Activation-based scoring (`ActivationPruner`, the RFC core)
1. `hooks.py` — `SensitivityHookManager` (fixes **G5**):
   - Registers forward hooks on: each block (input+output), each `out_proj`
     (pre-forward, captures per-head input), each `down_proj` (pre-forward, captures
     per-neuron activation).
   - Accumulates running mean L2 (and optional L1) in fp32 on CPU; handles tuple
     outputs, `past_key_values`, attention masks; `num_batches` cap; always removes
     hooks in `finally`.
2. `activation_pruner.py` — `ActivationPruner` orchestrator per §3, delegating physical
   surgery to the existing (Phase-0-fixed) slicing ops.
3. `data/calibration.py` (fixes **G9**): `get_calibration_dataset(name, tokenizer,
   samples, seq_len)` for `wikitext2` (generalize the loaders already in
   `run_llm_experiment.py`) and `c4`; accepts any iterable of dict-batches; for ViTs,
   accepts an image dataloader directly.
4. Metric-aware plan semantics: sensitivity/threshold logic parameterized by
   `higher_is_better` so perplexity works (fixes the sign bug in `_build_sparsity_plan`).

**Acceptance:** on Qwen2-0.5B — `compute_importance_scores` (512 WikiText-2 samples),
`prune(depth=0.2, width=0.15)`, ppl measured before/after, save/reload round-trip.
On `google/vit-base-patch16-224` — same flow with an image dataloader, top-1 sanity eval.

### Phase 3 — `SparsityPacker` (unstructured → structured)
1. `packer.py` (fixes **G7**):
   - `discover_unstructured_sparsity(tolerance=1e-8)` → per-layer dead-index map:
     - standard FFN: rows of `fc1` all-zero **and** matching cols of `fc2` (or either,
       policy flag);
     - SwiGLU: indices where `gate_proj.weight[i,:]==0` **AND** `up_proj.weight[i,:]==0`
       (RFC rule), sliced identically across gate/up/down;
     - attention: only whole-dead heads (all `head_dim` rows zero across Q and the
       `out_proj` cols) are packable — partial-head sparsity is reported but not packed.
   - `pack_to_structured()` → reuses `_rebuild_linear` slicing; calls ConfigUpdater;
     returns the physically smaller model.
   - Integration: after `GMP_Pruning`, `packer.pack_to_structured()` converts sconce's
     own masks into real speedups.
2. Guard rails: refuse to pack coupled layers with mismatched dead sets unless
   `intersect=True` (default) — the RFC's "safety check".

**Acceptance:** unit test — mask N random SwiGLU neurons to zero → pack → shapes shrink
by exactly N, logits identical to the masked model (atol 0), config
`intermediate_size` updated, save/reload OK.

### Phase 4 — Distillation (`sconce.distillation.Distiller`)
1. Fixes **G8**: teacher/student wrapper with
   `loss = α·CE(student, labels) + (1-α)·T²·KL(student/T ‖ teacher/T)`; optional
   hidden-state MSE on matched layers (post-depth-prune mapping comes from the kept
   layer indices). Teacher runs `no_grad`, optionally on a second device / fp16.
2. Replace `finetune()` in the LLM runner with `Distiller.train(...)`; keep plain-CE
   finetune as `distill=False`.

**Acceptance:** Qwen2-0.5B, 20% depth + 15% width: KD recovers strictly more ppl than
plain finetune at equal token budget (recorded in `results.tsv`).

### Phase 5 — Consolidation & CI
- `sconce/__init__.py` exports: `ActivationPruner`, `SparsityPacker`, `Distiller`,
  `TransformerPruner` (compat), CNN `prune` mixin untouched.
- Delete the ViT branch's duplicate `pruner.py` after its tests pass against the
  unified module; merge `ViT` → `experiment/apr14` → PR to `main`.
- CI matrix (CPU-only, tiny configs): HF ViT, DeiT, BEiT, timm ViT, torchvision ViT,
  GPT-2, Qwen2-GQA-tiny, Llama-tiny, BERT — each: detect → prune(width) →
  prune(depth) → forward → save/reload.

---

## 5. Test Plan Summary

| Test | Covers |
|---|---|
| `test_detection.py`: every arch in the CI matrix yields expected block count, heads, kv_heads, ffn_type | G1, G4 |
| `test_roundtrip.py`: prune → save_pretrained → from_pretrained → logits match | G2 |
| `test_timm_forward.py`: timm/EVA02 forward after head prune | G3 |
| `test_activation_scores.py`: hooks produce finite, batch-count-invariant scores; a layer fed zeros scores ~0 | G5 |
| `test_depth.py`: layer_idx renumbering, KV-cache generation works after drop | G6 |
| `test_packer.py`: masked ≡ packed logits, exact shape reduction | G7 |
| `test_distiller.py`: KD loss decreases; teacher unchanged | G8 |
| Ported ViT-branch suites (`test_hf_vit_pruning.py` etc.) | regression |

---

## 6. Risks & Notes

- **transformers 5.x churn:** attention modules are config-driven (verified: no
  `num_heads` attr on `Qwen2Attention`; `num_key_value_groups` still present). Keep all
  metadata write-backs attr-existence-guarded (current approach is right) and pin CI to
  both a 4.5x LTS and 5.x.
- **Heterogeneous per-layer pruning vs HF config** is fundamentally lossy; the plan's
  answer is uniform-mode for exportability + explicit error otherwise. Do not ship the
  `min()` behavior.
- **torch_pruning dependency:** keep DG-based pruning as the ViT fallback path
  (it handles residual/LayerNorm coupling for exotic vision models); the name-registry
  slicer stays primary for LLMs where DG tracing of KV caches is unreliable.
- **Depth pruning + DynamicCache:** transformers 5.x caches are indexed by layer_idx —
  renumbering (Phase 1.2) is mandatory, verified by a `generate()` test.
- KV-head pruning (`lock_head_dim=False`) is explicitly out of scope until Phase 2 is
  stable.
