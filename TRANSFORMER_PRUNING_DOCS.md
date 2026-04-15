# Transformer Pruning Functions Documentation

## File Location
`/local/mnt/workspace/users/sathya/projects/sconce/scripts/run_experiment.py`

---

## Architecture Detection

### `_is_eva(model)` — Lines 528-534
**Signature:**
```python
def _is_eva(model) -> bool:
```

**Purpose:** Detect whether a model uses EVA-style attention (separate q/k/v projections) vs standard ViT (fused qkv).

**Algorithm:**
1. Check if model has `blocks` attribute and is non-empty
2. Get first block's attention module
3. Test conditions:
   - Has `q_proj` attribute that is `nn.Linear`
   - **AND** does NOT have a real fused `qkv` Linear layer (i.e., `qkv` is not an `nn.Linear`)
4. Return True if EVA-style detected, False otherwise

**Return:** Boolean

**Usage in code:** Lines 805, 818, 827, 844 — Routes to correct pruning function based on architecture.

---

## Attention Head Pruning (Non-Structural)

### `prune_attention_heads(model, dataloader, device, head_sparsity=0.25)` — Lines 325-366
**Signature:**
```python
def prune_attention_heads(model, dataloader, device, head_sparsity=0.25) -> model:
```

**Type:** Non-structural (zeros out weights)

**Architecture Compatibility:** timm ViT only (requires `model.blocks[i].attn` with fused `qkv`)

**Algorithm:**
1. **Early exit:** If model lacks `blocks`, return unchanged
2. **Detect head properties:**
   - `num_heads = model.blocks[0].attn.num_heads`
   - `head_dim = model.blocks[0].attn.head_dim`
3. **Score each head (per-layer):**
   - Iterate each block
   - For each head h:
     - Extract columns `[h*head_dim : (h+1)*head_dim]` from `block.attn.proj.weight` (shape: `[d, d]`)
     - Compute L1 norm (mean absolute value) of these columns
     - Store in `head_importance[layer_idx, head_idx]`
4. **Select pruning threshold (global):**
   - Flatten all head scores
   - Compute k-th smallest value where k = `int(total_heads * head_sparsity)`
   - Use this as threshold
5. **Zero out columns:**
   - For each head scoring ≤ threshold:
     - Set `block.attn.proj.weight.data[:, h*head_dim:(h+1)*head_dim] = 0.0`
6. **Preserve architecture:** No parameter resizing, only zeroing

**Layer Attributes Accessed:**
- `block.attn.proj.weight` (reads and zeros columns)
- `block.attn.num_heads` (reads)
- `block.attn.head_dim` (reads)

**What It Modifies:**
- Zeros entire head columns in `proj.weight` only
- Does NOT modify `qkv.weight` or `qkv.bias`
- Parameter count unchanged

**Coupling:** None (self-contained)

---

## Attention Head Structural Pruning (timm ViT)

### `prune_attention_structural(model, head_sparsity=0.33)` — Lines 373-437
**Signature:**
```python
def prune_attention_structural(model, head_sparsity=0.33) -> model:
```

**Type:** Structural (physically removes heads, resizes layers)

**Architecture:** timm ViT with **fused qkv** (`block.attn.qkv` is single `nn.Linear`)

**Algorithm:**
1. **Early exit:** If no `blocks`, return unchanged
2. **Per-block processing:**
   - Get `num_heads`, `head_dim`, compute `inner_dim = num_heads * head_dim`
   - Compute `n_prune = int(num_heads * head_sparsity)`, `n_keep = max(1, num_heads - n_prune)`
   - Skip block if `n_keep == num_heads`
3. **Score heads by importance:**
   - For each head h, compute L1 norm of columns `[h*head_dim : (h+1)*head_dim]` in `proj.weight`
   - Select top `n_keep` heads by score, sort indices
4. **Build row selection for qkv:**
   - qkv outputs 3 blocks: Q, K, V (each size `inner_dim`)
   - For each block (offset 0, inner_dim, 2*inner_dim):
     - For each kept head h: add row range `[offset + h*head_dim : offset + (h+1)*head_dim]`
   - Result: `keep_rows` tensor of selected row indices
5. **Build column selection for proj:**
   - For each kept head h: add column range `[h*head_dim : (h+1)*head_dim]`
   - Result: `keep_cols` tensor
6. **Create new layers:**
   - `new_qkv = nn.Linear(embed_dim, 3*n_keep*head_dim, bias=has_bias)`
   - `new_proj = nn.Linear(n_keep*head_dim, embed_dim, bias=has_bias)`
7. **Copy selected weights:**
   - `new_qkv.weight = qkv.weight[keep_rows, :]` (select rows)
   - `new_qkv.bias = qkv.bias[keep_rows]` (if exists)
   - `new_proj.weight = proj.weight[:, keep_cols]` (select columns)
   - `new_proj.bias = proj.bias` (unchanged)
8. **Update block attributes:**
   - `attn.qkv = new_qkv`
   - `attn.proj = new_proj`
   - `attn.num_heads = n_keep`
   - `attn.attn_dim = n_keep * head_dim` (new fused dimension)

**Layer Attributes Accessed:**
- `block.attn.qkv` (in_features=embed_dim, out_features=3*inner_dim)
- `block.attn.proj` (in_features=inner_dim, out_features=embed_dim)
- `block.attn.num_heads` (read & updated)
- `block.attn.head_dim` (read)
- `block.attn.attn_dim` (updated)

**What It Modifies:**
- Physically resizes `qkv` and `proj` layers
- Changes parameter count
- Preserves q/k/v output ordering (contiguous blocks)

**Coupling Handled:**
- When output of qkv changes (3*old_inner_dim → 3*new_inner_dim), proj input also changes
- Carefully selects rows from qkv and columns from proj to maintain Q-K-V block structure

---

## Attention Head Structural Pruning (EVA02)

### `prune_attention_structural_eva(model, head_sparsity=0.33)` — Lines 537-595
**Signature:**
```python
def prune_attention_structural_eva(model, head_sparsity=0.33) -> model:
```

**Type:** Structural (physically removes heads, resizes layers)

**Architecture:** EVA02 with **separate q_proj, k_proj, v_proj** (no fused qkv)

**Algorithm:**
1. **Early exit:** If no `blocks`, return unchanged
2. **Per-block processing:**
   - Get `num_heads`, `head_dim`, `inner_dim = num_heads * head_dim`
   - Compute `n_prune`, `n_keep`, skip if unchanged
3. **Score heads:**
   - From `attn.proj.weight` (shape: [embed_dim, inner_dim])
   - For each head h: L1 norm of columns `[h*head_dim : (h+1)*head_dim]`
   - Select top `n_keep` heads, sort
4. **Build row selection:**
   - For each kept head h: add row range `[h*head_dim : (h+1)*head_dim]`
   - Result: `keep_rows` tensor
5. **Resize q_proj and v_proj (with bias):**
   - For each in ("q_proj", "v_proj"):
     - Get source layer
     - Create `new_l = nn.Linear(embed_dim, n_keep*head_dim, bias=has_bias)`
     - Copy: `new_l.weight = src.weight[keep_rows, :]`
     - Copy: `new_l.bias = src.bias[keep_rows]` (if exists)
     - Update: `setattr(attn, name, new_l)`
6. **Resize k_proj (typically no bias in EVA02):**
   - Create `new_k = nn.Linear(embed_dim, n_keep*head_dim, bias=has_bias)`
   - Copy weights and bias (if exists)
   - Update: `attn.k_proj = new_k`
7. **Resize proj (output projection):**
   - Create `new_proj = nn.Linear(n_keep*head_dim, embed_dim, bias=has_bias)`
   - Copy: `new_proj.weight = proj.weight[:, keep_rows]`
   - Copy: `new_proj.bias = proj.bias` (if exists)
   - Update: `attn.proj = new_proj`
8. **Update metadata:**
   - `attn.num_heads = n_keep`
   - Note: EVA02 handles `head_dim` separately (not updated)

**Layer Attributes Accessed:**
- `block.attn.q_proj` (in=embed_dim, out=inner_dim)
- `block.attn.k_proj` (in=embed_dim, out=inner_dim)
- `block.attn.v_proj` (in=embed_dim, out=inner_dim)
- `block.attn.proj` (in=inner_dim, out=embed_dim)
- `block.attn.num_heads` (updated)
- `block.attn.head_dim` (read-only)

**What It Modifies:**
- Physically resizes all four linear layers
- Changes parameter count

**Coupling Handled:**
- q/k/v outputs change from inner_dim → new_inner_dim
- proj input changes to match
- All three separate projections share the same output dimension (by definition)

**Key Differences from timm:**
- Separate q/k/v instead of fused qkv
- Can handle k_proj with/without bias (typical: no bias)
- Row selection applied to q/k/v outputs (not qkv as single layer)
- Column selection applied to proj input

---

## FFN Neuron Pruning (Non-Structural)

### `prune_ffn_neurons(model, ffn_sparsity=0.25)` — Lines 444-471
**Signature:**
```python
def prune_ffn_neurons(model, ffn_sparsity=0.25) -> model:
```

**Type:** Non-structural (zeros out weights)

**Architecture:** timm ViT with MLP (`block.mlp.fc1`, `block.mlp.fc2`)

**Algorithm:**
1. **Early exit:** If no `blocks`, return unchanged
2. **Per-block processing:**
   - `fc1` (dense layer up: hidden_dim → intermediate_dim)
   - `fc2` (dense layer down: intermediate_dim → hidden_dim)
3. **Score neurons by importance:**
   - Compute L1 norm (mean absolute value) per neuron:
     - `neuron_importance = fc2.weight.abs().mean(dim=0)` (shape: [intermediate_dim])
     - This averages fc2 input weights for each intermediate neuron
4. **Select neurons to prune:**
   - k = `int(intermediate_dim * ffn_sparsity)` (number to prune)
   - Find indices of k smallest-scoring neurons
5. **Zero out selected neurons:**
   - `fc1.weight[prune_idx, :] = 0.0` (zero rows in fc1)
   - `fc1.bias[prune_idx] = 0.0` (if bias exists)
   - `fc2.weight[:, prune_idx] = 0.0` (zero columns in fc2)
6. **No resizing:** Parameters remain allocated but inactive

**Layer Attributes Accessed:**
- `block.mlp.fc1` (reads weight, writes)
- `block.mlp.fc2` (reads weight)
- `block.mlp.fc1.bias` (if exists)

**What It Modifies:**
- Zeros rows in `fc1.weight` and `fc1.bias`
- Zeros columns in `fc2.weight`
- Parameter count unchanged (weights allocated but sparse)

**Coupling:** Self-contained (only affects fc1/fc2 of same block)

---

## FFN Structural Pruning (timm ViT)

### `prune_ffn_structural(model, ffn_sparsity=0.5)` — Lines 478-521
**Signature:**
```python
def prune_ffn_structural(model, ffn_sparsity=0.5) -> model:
```

**Type:** Structural (physically resizes layers)

**Architecture:** timm ViT with standard MLP

**Algorithm:**
1. **Early exit:** If no `blocks`, return unchanged
2. **Per-block processing:**
   - Get intermediate dimension: `intermediate_dim = fc1.out_features`
   - Compute `n_keep = int(intermediate_dim * (1.0 - ffn_sparsity))`
   - Ensure `n_keep ≥ 1`
3. **Score neurons:**
   - `neuron_importance = fc2.weight.abs().mean(dim=0)` (shape: [intermediate_dim])
   - Get indices of top `n_keep` neurons
   - Sort indices to maintain order: `keep_idx = topk(...).indices.sort().values`
4. **Create new layers:**
   - `new_fc1 = nn.Linear(in_features, n_keep, bias=has_bias)` 
     - in_features = `fc1.in_features` (original hidden dim)
     - out_features = `n_keep` (pruned)
   - `new_fc2 = nn.Linear(n_keep, out_features, bias=has_bias)`
     - in_features = `n_keep`
     - out_features = `fc2.out_features` (original hidden dim)
5. **Copy selected weights:**
   - `new_fc1.weight = fc1.weight[keep_idx, :]` (select rows — intermediate neurons)
   - `new_fc1.bias = fc1.bias[keep_idx]` (if exists)
   - `new_fc2.weight = fc2.weight[:, keep_idx]` (select columns — connections to kept neurons)
   - `new_fc2.bias = fc2.bias` (unchanged, same output size)
6. **Replace layers:**
   - `mlp.fc1 = new_fc1`
   - `mlp.fc2 = new_fc2`

**Layer Attributes Accessed:**
- `block.mlp.fc1` (in=hidden_dim, out=intermediate_dim)
- `block.mlp.fc2` (in=intermediate_dim, out=hidden_dim)
- Both weights and biases (if exist)

**What It Modifies:**
- Physically shrinks fc1 output dimension: intermediate_dim → n_keep
- Physically shrinks fc2 input dimension: intermediate_dim → n_keep
- Reduces parameter count

**Coupling Handled:**
- When fc1 output size changes, fc2 input size must match
- Both changes made atomically per block

---

## FFN Structural Pruning (EVA02 SwiGLU)

### `prune_ffn_structural_eva(model, ffn_sparsity=0.5)` — Lines 598-647
**Signature:**
```python
def prune_ffn_structural_eva(model, ffn_sparsity=0.5) -> model:
```

**Type:** Structural (physically resizes layers)

**Architecture:** EVA02 with **SwiGLU** FFN (separate fc1_g, fc1_x, fc2, optional norm)

**Algorithm:**
1. **Early exit:** If no `blocks`, return unchanged
2. **Per-block processing:**
   - `fc1_g` (gate branch: hidden_dim → intermediate_dim)
   - `fc1_x` (value/up branch: hidden_dim → intermediate_dim)
   - `fc2` (output: intermediate_dim → hidden_dim)
   - Optional: `mlp.norm` (LayerNorm if exists)
3. **Get dimensions:**
   - `intermediate_dim = fc1_g.out_features`
   - `n_keep = int(intermediate_dim * (1.0 - ffn_sparsity))`
4. **Score neurons (SwiGLU-aware):**
   - Combine importance from gate, value, and output:
   ```
   importance = (fc1_g.weight.abs().mean(dim=1) *
                 fc1_x.weight.abs().mean(dim=1) *
                 fc2.weight.abs().mean(dim=0))
   ```
   - Geometric mean of three components
   - Select top `n_keep` neurons
   - Sort: `keep_idx = topk(...).indices.sort().values`
5. **Resize fc1_g and fc1_x (gate and value branches):**
   - For each in ("fc1_g", "fc1_x"):
     - Create `new_l = nn.Linear(in_f, n_keep, bias=has_bias)`
       - in_f = `fc1_g.in_features`
     - Copy: `new_l.weight = src.weight[keep_idx, :]`
     - Copy: `new_l.bias = src.bias[keep_idx]` (if exists)
     - Update: `setattr(mlp, name, new_l)`
6. **Resize fc2:**
   - Create `new_fc2 = nn.Linear(n_keep, out_f, bias=has_bias)`
     - out_f = `fc2.out_features`
   - Copy: `new_fc2.weight = fc2.weight[:, keep_idx]`
   - Copy: `new_fc2.bias = fc2.bias` (if exists)
   - Update: `mlp.fc2 = new_fc2`
7. **Resize LayerNorm (if present):**
   - Get old norm: `mlp.norm`
   - Create `new_norm = nn.LayerNorm(n_keep, eps=old_norm.eps)`
   - Copy: `new_norm.weight = old_norm.weight[keep_idx]`
   - Copy: `new_norm.bias = old_norm.bias[keep_idx]`
   - Update: `mlp.norm = new_norm`

**Layer Attributes Accessed:**
- `block.mlp.fc1_g` (in=hidden_dim, out=intermediate_dim)
- `block.mlp.fc1_x` (in=hidden_dim, out=intermediate_dim)
- `block.mlp.fc2` (in=intermediate_dim, out=hidden_dim)
- `block.mlp.norm` (optional, size=intermediate_dim)
- All weights and biases

**What It Modifies:**
- Physically shrinks both fc1_g and fc1_x output: intermediate_dim → n_keep
- Physically shrinks fc2 input: intermediate_dim → n_keep
- Resizes LayerNorm if present
- Reduces parameter count

**Coupling Handled:**
- Gate and value branches must have same output size (inherent to SwiGLU)
- Both must match fc2 input size
- LayerNorm (if used) operates on intermediate dimension

**Key Differences from timm:**
- Two separate fc1 branches (gate & value) instead of single fc1
- Scoring combines three weight matrices (geometric mean)
- Optional LayerNorm resizing

---

## Depth Pruning

### `prune_depth(model, depth_sparsity=0.25)` — Lines 650-685
**Signature:**
```python
def prune_depth(model, depth_sparsity=0.25) -> model:
```

**Type:** Structural (removes entire blocks)

**Architecture:** Any ViT with `model.blocks` as `nn.Sequential` or `nn.ModuleList`

**Algorithm:**
1. **Early exit:** If no `blocks`, return unchanged
2. **Get dimensions:**
   - `n_blocks = len(blocks)`
   - `n_remove = int(n_blocks * depth_sparsity)`
   - `n_keep = n_blocks - n_remove`
3. **Early exit if nothing to remove:**
   - If `n_remove == 0`, return unchanged
4. **Score all blocks:**
   - For each block, compute sum of L1-norms of all parameters:
     ```
     score = sum(p.data.abs().mean().item() for p in block.parameters())
     ```
   - Store as (score, index) tuples
5. **Select blocks to remove:**
   - Sort by score ascending (least important first)
   - Take first `n_remove` blocks
   - Create set of indices to remove
6. **Build new block list:**
   - Filter out removed blocks: `kept_blocks = [b for i, b in enumerate(blocks) if i not in remove_set]`
7. **Replace model.blocks:**
   - `model.blocks = nn.ModuleList(kept_blocks)`

**Layer Attributes Accessed:**
- `model.blocks` (reads structure, writes new ModuleList)

**What It Modifies:**
- Removes entire transformer blocks
- Changes model depth (number of layers)
- Significantly reduces parameter count

**No Coupling:** Each block is independent

---

## Global Magnitude Pruning (GMP)

### `gmp_prune_transformer(model, sparsity)` — Lines 300-306
**Signature:**
```python
def gmp_prune_transformer(model, sparsity) -> model:
```

**Type:** Non-structured (unstructured magnitude pruning on all Linear layers)

**Algorithm:**
1. **Iterate all modules:**
   - For each `nn.Linear` module in model
2. **Apply L1 pruning:**
   - `torch.nn.utils.prune.l1_unstructured(module, name="weight", amount=sparsity)`
   - This zeros the smallest `sparsity` fraction of weight elements globally
3. **Result:** Sparse weight matrices (zeros scattered throughout)

**Layer Attributes Accessed:**
- All `nn.Linear.weight` parameters (reads and applies pruning masks)

**What It Modifies:**
- Adds pruning masks to Linear layers via `module.weight_orig` and `module.weight_mask`
- Parameters not actually resized (masked, not removed)

**Remove Masks:**
```python
def gmp_remove_masks(model):
    # Makes pruning permanent by removing reparameterization
    for module in model.modules():
        if isinstance(module, nn.Linear):
            torch.nn.utils.prune.remove(module, "weight")
```
- Replaces `module.weight` with masked result
- Converts sparse masks to actual zeros in weights

---

## Summary Table

| Function | Type | Arch | What Changes | Coupling |
|----------|------|------|--------------|----------|
| `prune_attention_heads` | Non-struct | timm ViT | Zeros proj columns | Self-contained |
| `prune_attention_structural` | Struct | timm ViT (fused qkv) | Resizes qkv, proj | q/k/v output ↔ proj input |
| `prune_attention_structural_eva` | Struct | EVA02 (sep q/k/v) | Resizes all 4 layers | Outputs ↔ proj input |
| `prune_ffn_neurons` | Non-struct | timm ViT | Zeros fc1 rows, fc2 cols | Self-contained |
| `prune_ffn_structural` | Struct | timm ViT | Resizes fc1, fc2 | fc1 output ↔ fc2 input |
| `prune_ffn_structural_eva` | Struct | EVA02 (SwiGLU) | Resizes 2×fc1, fc2, norm | All coupled |
| `prune_depth` | Struct | Any ViT | Removes blocks entirely | None |
| `gmp_prune_transformer` | Non-struct | Any | Masks Linear weights | None |
| `_is_eva` | Detection | Any | Returns bool | N/A |

---

## Key Architectural Insights

### Head Detection Pattern
- **timm ViT:** `attn.num_heads` and `attn.head_dim` exist
- **EVA02:** Same attributes, but different layer structure

### Dimension Tracking
- **Q-K-V fused (timm):** Single `qkv` layer outputs `3*inner_dim`, split by code
- **Q-K-V separate (EVA02):** Three separate projections, each outputs `inner_dim`

### Parameter Coupling
1. **Attention head pruning:**
   - qkv output dim ↔ proj input dim must match
   - When heads removed: all 3 blocks (Q, K, V) shrink together
   
2. **FFN pruning:**
   - fc1 output dim ↔ fc2 input dim must match
   - Non-structural doesn't change this (just zeros)
   - Structural must update both atomically

3. **EVA SwiGLU:**
   - fc1_g output ↔ fc1_x output (must be equal)
   - Both ↔ fc2 input (must match)
   - Optional norm operates on this dimension

### No GQA Handling
- Code assumes standard multi-head attention (num_heads >= 1 per dimension)
- No support for Grouped Query Attention (GQA where num_kv_heads < num_q_heads)

### No HuggingFace Model Handling (in pruning)
- Model loading supports HF via `transformers.AutoModelForImageClassification` (line 50)
- But pruning functions only work on `model.blocks` structure
- Would fail on HF models with different block layouts (e.g., `model.encoder.layer`)

---

## Integration Points (Main Script)

Lines 805-847: Architecture detection and routing
- Checks `_is_eva()` to route structural pruning
- Supports combined pruning modes (e.g., depth+ffn)
- After pruning: optional fine-tuning with KD support

