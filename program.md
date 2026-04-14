# sconce — Generic Transformer Compression Framework

## Main Goal

Make sconce a **model-agnostic** compression toolkit that works out-of-the-box on any transformer from HuggingFace (`transformers`) or `timm` — ViTs, LLMs, VLM encoders, encoder-decoders, etc.

```python
from sconce import sconce

s = sconce()
s.model = model                    # any HuggingFace / timm transformer
s.dataloader = dataloader
s.compress()                       # → structured-pruned + quantized + benchmarked
```

The framework should handle architecture detection, pruning-group discovery, and importance scoring generically — **not hard-coded per model family**. A new model should work by just passing it in; model-specific quirks (GQA grouping, tied embeddings, etc.) are handled by detection logic, not user flags.

## Pruning Scope

**Structured pruning is the primary target.** The goal is physically smaller, faster models — not sparse-weight matrices that still need the same compute.

### Core Methods

- **CWP (Channel-Wise Pruning)**: remove entire filters / neurons / attention heads. The resulting model has genuinely smaller weight tensors — no sparse-kernel support required for speedup.
- **Wanda-style importance scoring**: use `|W| * ||X||` (weight magnitude scaled by input activation norm) to rank structures for removal. This is calibration-data-aware, one-shot (no iterative retraining), and works across Linear, Conv, and attention projections. It generalises magnitude pruning by incorporating activation statistics from a small calibration set.

### Structured Targets (what gets removed)

These apply generically across any transformer:
- **Attention heads**: remove full heads (structured). For GQA/MQA models, respect key-value group boundaries.
- **FFN neurons**: remove entire intermediate neurons in MLP/FFN blocks (shrink hidden dim).
- **Layer removal (depth pruning)**: drop full transformer blocks when sensitivity allows.
- **Embedding channels**: reduce hidden dimension across the model (width pruning).

### Role of GMP

GMP (Gradual Magnitude Pruning) is used as an **exploratory / comparison baseline** — it measures how much accuracy a given sparsity level costs per layer. GMP results inform structured-pruning decisions (e.g. layers that tolerate high unstructured sparsity → candidates for structured head/neuron removal or full-layer drop) but are **not end-deliverables**. The shipped artifact is always a structurally smaller model.

### Why CWP + Wanda

| Property | CWP | Wanda | CWP + Wanda |
|----------|-----|-------|-------------|
| Structured (real speedup) | Yes | Can be adapted | Yes |
| Calibration-aware | No (magnitude only) | Yes (W * X) | Yes |
| One-shot (no retraining to prune) | Yes | Yes | Yes |
| Works on Linear + Conv | Yes | Yes | Yes |
| Architecture-agnostic | Yes | Yes | Yes |

Pure CWP uses weight magnitude alone. Adding Wanda-style scoring (`|W_ij| * ||X_j||_2`) gives better importance estimates at near-zero extra cost (one forward pass over calibration data). This combination gives us structured pruning with data-aware importance — the best of both.

## Target Models (validation & benchmarking)

sconce should work on **any** HuggingFace/timm transformer. The models below are our primary validation targets — if the framework handles these well, it generalises.

### Vision Encoders (ViTs)

| Model | Source | Why |
|-------|--------|-----|
| InternViT (InternVL series) | `OpenGVLab/InternViT-*` | Large ViT, vision encoder for VLMs |
| EVA02 | timm `eva02_*` / HuggingFace | High-performance ViT, used in multiple VLM pipelines |
| SigLIP / SO400M | `google/siglip-*` | Contrastive ViT, PaLI-style VLMs |

Structured pruning on vision encoders must preserve interface compatibility (hidden-dim divisibility, output token count) so the compressed encoder still plugs into its downstream VLM.

### LLMs (decoder-only transformers)

| Family | Versions | Source |
|--------|----------|--------|
| Qwen | 2, 2.5, 3, 3.5 | `Qwen/*` |
| LLaMA | 2, 3, 3.1, 3.2 | `meta-llama/*` |

LLMs exercise GQA-aware head pruning, FFN width pruning, and depth pruning. These families cover both standard MHA (LLaMA-2) and GQA (LLaMA-3, Qwen2+), testing the framework's ability to auto-detect attention grouping.

### Ramp-up proxies (fast iteration)

- `vit_tiny_patch16_224` (timm) — ViT pruning logic
- `Qwen/Qwen2-0.5B` — LLM pruning logic

Validate code paths on these before scaling to full-size targets.

## Context

sconce is an E2E AutoML compression package. Originally CNN-centric (Conv2d, Linear, BN) — this program generalises it to **any transformer architecture** (ViT, LLM, encoder-decoder, VLM encoder) via HuggingFace/timm, with CWP + Wanda-style structured pruning as the core compression method.

### Repository Structure

```
sconce/
    __init__.py
    sconce.py           — main class: train/evaluate/compress orchestration
    pruner.py           — CWP+Wanda structured pruning, GMP baseline, sensitivity_scan
    quanter.py          — PTQ, QAT quantization
    perf.py             — latency/MACs/size profiling, model comparison
    model_analyzer.py   — layer analysis utilities
```

### Branches

| Branch | What it adds | Status |
|--------|-------------|--------|
| `main` | Base CNN compression pipeline (Conv2d, Linear, BN) | Stable |
| `ViT` | Vision Transformer support, `tests/` directory, attention pruning flags | WIP |
| `attention` | Attention-head-aware pruning in pruner.py, `attention_heads` flag | WIP |

### External: Octopus (sensitivity scan parallelization)

Octopus (`autoresearch/octopus`) provides GPU worker parallelization for model inference. Used for **parallel sensitivity scanning** — the bottleneck in sconce's compression pipeline.

```python
from octopus import Octopus

with Octopus(model=model, eval_fn=eval_fn, safety_net_gb=1.5) as o:
    for layer_config in layer_configs:
        o.submit(layer_config)
    results = o.gather()
```

Key features relevant to sconce:
- `submit/gather` pattern for parallel per-layer sensitivity eval
- `sensitivity_scan()` method for quantization sensitivity (enabling loop)
- Auto GPU discovery + VRAM profiling → max parallelism
- Dynamic scheduling with work-stealing for uneven layer costs
- PyTorch `nn.Module` adapter (auto-detected)

## Setup

To set up a new experiment run:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `apr14`). Branch `experiment/<tag>` must not exist.
2. **Create the branch**: `git checkout -b experiment/<tag>` from current `main`.
3. **Read in-scope files**: The repo is small. Read these for full context:
   - `README.rst` — project overview
   - `sconce/sconce.py` — main orchestrator (train, evaluate, compress)
   - `sconce/pruner.py` — pruning methods + sensitivity scan
   - `sconce/quanter.py` — quantization methods
   - `sconce/perf.py` — performance profiling
4. **Verify environment**: Check that PyTorch + CUDA are available. Install sconce: `pip install -e .`
5. **Install model deps**: `pip install transformers timm datasets evaluate`
6. **Initialize results.tsv**: Create `results.tsv` with just the header row.
7. **Confirm and go**: Confirm setup looks good.

## What You CAN Modify

- `sconce/sconce.py` — train/eval/compress orchestration, add transformer-aware data handling
- `sconce/pruner.py` — pruning strategies, sensitivity scan, attention-head pruning
- `sconce/quanter.py` — quantization strategies, bitwidth selection
- `sconce/perf.py` — profiling, comparison tables
- `sconce/model_analyzer.py` — layer analysis, transformer block detection
- `scripts/` — experiment runner scripts
- Test files under `tests/`

## What You CANNOT Modify

- External library source code (transformers, timm, torch_pruning, etc.)
- The evaluation harness once established for a model (consistency across experiments)

## The Goal

**Find the best compression configuration for each model that maximizes compression ratio while minimizing accuracy degradation.**

Metrics tracked per experiment:
- `accuracy` — task accuracy (top-1 for classification, perplexity for LLM)
- `model_size_mb` — compressed model size in MiB (non-zero params only)
- `compression_ratio` — original_size / compressed_size
- `latency_ms` — inference latency per sample
- `peak_vram_mb` — peak GPU memory during inference

## Supported Model Types

sconce auto-detects the model architecture and selects appropriate pruning groups. No model-specific flags required.

### Any HuggingFace transformer

```python
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForImageClassification
# All of these should just work:
model = AutoModel.from_pretrained('OpenGVLab/InternViT-6B-448px-V1-5')
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-7B')
model = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B')
```

### Any timm vision model

```python
import timm
model = timm.create_model('eva02_large_patch14_448', pretrained=True)
model = timm.create_model('vit_base_patch16_224', pretrained=True)
```

### What the framework auto-detects per model

- **Prunable layer types**: `nn.Linear`, `nn.Conv2d`, attention projections (Q/K/V/O)
- **Attention layout**: MHA vs GQA vs MQA → determines head-pruning grouping
- **FFN structure**: gate/up/down patterns (GLU variants), intermediate dim
- **Tied weights**: shared embeddings (input ↔ output), tied Q/K
- **Block boundaries**: which layers form a transformer block (for depth pruning)

### Structured pruning targets (generic)

| Target | What gets removed | Applicable to |
|--------|-------------------|---------------|
| Attention heads | Full Q/K/V/O head slices | All transformers |
| FFN neurons | Intermediate-dim neurons (gate+up+down together) | All transformers |
| Transformer blocks | Entire layers (depth pruning) | Deep models |
| Embedding channels | Hidden-dim slices across the model | Width pruning |

Importance scoring: CWP + Wanda (`|W| * ||X||`) by default. Fall back to magnitude-only when no calibration data is provided.

## Sensitivity Scan with Octopus

The sensitivity scan is the most expensive step — evaluating each layer independently. Use Octopus to parallelize:

```python
from octopus import Octopus

def eval_fn(model, batch):
    """Evaluate model accuracy on a calibration batch."""
    model.eval()
    with torch.no_grad():
        outputs = model(batch['input_ids'], attention_mask=batch['attention_mask'])
        # return metric (accuracy, perplexity, SQNR, etc.)
    return metric

with Octopus(
    model=model,
    eval_fn=eval_fn,
    safety_net_gb=1.5,
    scheduling="dynamic",
    stagger_init_s=5.0,
) as o:
    for layer_name in prunable_layers:
        o.submit(layer_name)
    sensitivity_scores = o.gather()
# sensitivity_scores: {layer_name: accuracy_at_sparsity}
```

For quantization sensitivity (per-layer bitwidth selection):

```python
with Octopus(model=quantized_sim, eval_fn=sqnr_eval_fn) as o:
    sqnr_map = o.sensitivity_scan(layers=quantizable_layers, mode="enabling")
# sqnr_map: {layer_name: sqnr_db}  → higher = less sensitive → can use lower bitwidth
```

## Output Format

Each experiment prints a summary:

```
---
model:              vit_base_patch16_224
task:               imagenet-1k-val
original_accuracy:  81.07
compressed_accuracy: 79.85
accuracy_drop:      -1.22
original_size_mb:   343.2
compressed_size_mb: 87.4
compression_ratio:  3.93x
latency_original_ms: 12.4
latency_compressed_ms: 5.1
speedup:            2.43x
peak_vram_mb:       1240.5
prune_mode:         CWP
sparsity:           0.50
quantization:       int8
sensitivity_scan_time_min: 3.2
total_time_min:     8.7
```

Extract key metric:
```bash
grep "^compressed_accuracy:" run.log
```

## Logging Results

Log each experiment to `results.tsv` (tab-separated):

```
commit	model	accuracy	accuracy_drop	size_mb	compression	latency_ms	speedup	prune_mode	quant	status	description
```

Example:

```
commit	model	accuracy	accuracy_drop	size_mb	compression	latency_ms	speedup	prune_mode	quant	status	description
a1b2c3d	vit_base	81.07	0.00	343.2	1.00x	12.4	1.00x	none	none	keep	baseline
b2c3d4e	vit_base	79.85	-1.22	87.4	3.93x	5.1	2.43x	CWP	int8	keep	CWP 50% + int8 quantization
c3d4e5f	vit_base	76.20	-4.87	45.2	7.59x	4.8	2.58x	GMP	int4	discard	too much accuracy loss at int4
d4e5f6g	vit_base	0.00	0.00	0.0	0.00x	0.0	0.00x	sparseGPT	int8	crash	OOM on attention pruning
```

Status values: `keep` (Pareto-improving), `discard` (dominated or regression), `crash`.

## The Experiment Loop

Runs on dedicated branch `experiment/<tag>`.

LOOP FOREVER:

1. **Check git state**: current branch/commit
2. **Pick next experiment**: choose a compression config to try:
   - Vary structured pruning: CWP+Wanda heads, CWP+Wanda FFN, depth pruning, width pruning
   - Use GMP only as a sensitivity baseline (not a final deliverable)
   - Vary pruning ratio: 0.1 → 0.9 in steps
   - Vary quantization: fp16, int8, int4, mixed-precision
   - Vary model: cycle through validation targets (EVA02, InternViT, Qwen2, LLaMA, etc.)
   - Combine: structured prune + fine-tune + quantize
3. **Modify code** to implement the config
4. **git commit**
5. **Run experiment**: `python scripts/run_experiment.py --config <config> > run.log 2>&1`
6. **Read results**: `grep "^compressed_accuracy:\|^compression_ratio:" run.log`
7. **Handle failures**: if grep empty → crash. `tail -n 50 run.log` for traceback. Fix if trivial, skip if fundamental.
8. **Log to TSV** (do NOT commit results.tsv)
9. **Keep or discard**:
   - If Pareto-improving (better accuracy at same compression OR better compression at same accuracy) → keep commit
   - Otherwise → `git reset` to previous good state
10. **Repeat**

## Experiment Prioritization

### Phase 1: Baseline Establishment
For each target model, run uncompressed baseline to record original accuracy/size/latency.

### Phase 2: Single-Axis Exploration
- GMP sweep (sensitivity baseline — informs which layers/heads to structurally prune)
- CWP+Wanda structured pruning: head, FFN-neuron, depth sweeps at varying ratios
- Quantization only: fp16, int8, int4
- Measure sensitivity scan time (with and without Octopus)

### Phase 3: Combined Compression
- Best structured pruning config + quantization
- Iterative: CWP+Wanda prune → fine-tune → quantize → measure
- Mixed-precision: use sensitivity scan to assign per-layer bitwidths

### Phase 4: Advanced Techniques
- Knowledge distillation from original to compressed model
- Depth pruning: remove full transformer blocks guided by layer sensitivity
- GQA-aware head pruning for grouped-query models (auto-detected, not model-specific)
- Width + depth combined: prune heads/neurons AND remove layers
- Wanda metric variants: explore `|W|^α * ||X||^β` tuning, per-block calibration
- Generalization testing: try on unseen model families (Mistral, Phi, DBRX, etc.)

## Simplicity Criterion

All else equal, simpler is better. A 0.1% accuracy gain that adds 50 lines of complexity → probably not worth it. Removing code while maintaining accuracy → always keep. A cleaner API that achieves same compression → keep.

## Timeouts & Crashes

- **Sensitivity scan timeout**: depends on model size. ViT-B ~5min, LLaMA-7B ~30min. If >2x expected → kill + treat as crash.
- **Training/fine-tune timeout**: scale with epochs. If >3x expected → kill.
- **OOM**: increase Octopus `safety_net_gb`, reduce batch size, or skip config.
- **Trivial fix** (typo, import): fix + re-run.
- **Fundamental issue** (model architecture incompatible): log crash, move on.

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask. Do NOT ask "should I keep going?". Run indefinitely until manually stopped. If out of ideas:
- Re-read pruner.py / quanter.py for unexplored code paths
- Try combining near-miss configs
- Try more radical approaches (layer dropping, progressive pruning)
- Switch to a different target model
- Read research papers referenced in the codebase for new angles

## Key Transformer-Specific Considerations

### Wanda-style Importance Scoring (core method)
```python
# Compute per-structure importance: |W| * ||X||
# Works for any nn.Linear (attention projections, FFN layers)
importance = {}
for name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        W = module.weight.data                        # [out, in]
        X_norm = activation_norms[name]               # [in] — ||X_j||_2 over calibration set
        score = (W.abs() * X_norm.unsqueeze(0))       # [out, in]
        # For structured pruning, aggregate per output neuron / head:
        importance[name] = score.sum(dim=1)            # [out] — importance of each output channel
```

### Structured Head Importance (Wanda on Q/K/V/O)
```python
# Aggregate Wanda scores across Q, K, V, O projections per head
head_importance = torch.zeros(num_layers, num_heads)
for layer_idx in range(num_layers):
    for proj in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        proj_importance = importance[f'layers.{layer_idx}.self_attn.{proj}']
        # Reshape to [num_heads, head_dim] and sum per head
        head_importance[layer_idx] += proj_importance.view(num_heads, -1).sum(dim=1)
# Prune heads with lowest aggregate importance
```

### Layer-wise Sensitivity for LLMs
```python
# Per-layer reconstruction error (OBC/sparseGPT style)
for name, module in model.named_modules():
    if isinstance(module, nn.Linear) and 'attention' in name:
        W = module.weight.data
        H = compute_hessian(module, calibration_data)
        pruned_W, error = obc_prune(W, H, sparsity=target)
        sensitivity[name] = error.item()
```

### Mixed-Precision Quantization via Sensitivity
```python
# Use Octopus sensitivity_scan to assign bitwidths
sqnr_scores = octopus.sensitivity_scan(layers=all_linear_layers, mode="enabling")
# High SQNR → robust to quantization → use lower bitwidth (4-bit)
# Low SQNR → sensitive → keep higher bitwidth (8-bit or fp16)
bitwidth_map = {layer: 4 if sqnr > 40 else 8 for layer, sqnr in sqnr_scores.items()}
```

## File Map (target state after cleanup)

```
sconce/
    __init__.py
    sconce.py              — main orchestrator: model loading (HF/timm), train, eval, compress
    pruner.py              — CWP+Wanda structured pruning, GMP baseline, sensitivity_scan
    quanter.py             — PTQ, QAT, mixed-precision, GPTQ-style weight-only
    perf.py                — latency, MACs, size profiling, Pareto analysis
    model_analyzer.py      — layer detection (Conv/Linear/Attention/FFN/Embedding)
    transforms.py          — HuggingFace/timm data pipeline adapters
scripts/
    run_experiment.py      — single experiment runner (config → compress → log)
    sweep.py               — multi-config sweep orchestrator
tests/
    test_vit.py            — ViT compression end-to-end
    test_llm.py            — LLM compression end-to-end
    test_sensitivity.py    — sensitivity scan with Octopus
```

## Testing

```bash
# Unit tests (no GPU required for most)
pytest tests/ -v

# Integration tests (requires CUDA)
pytest tests/ -v -m gpu

# Specific model tests
pytest tests/test_vit.py -v
pytest tests/test_llm.py -v
```
