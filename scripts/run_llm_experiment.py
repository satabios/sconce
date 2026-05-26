"""
LLM experiment runner for Qwen2-0.5B (and other HuggingFace causal LMs).

Compression pipeline:
  load checkpoint → prune (FFN/depth/attention) → finetune → quantize → measure → print

Usage:
  python scripts/run_llm_experiment.py --config scripts/configs/apr14/qwen_01_baseline.json
"""

import argparse
import json
import math
import os
import sys
import tempfile
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MiB = 1024 * 1024

# ---------------------------------------------------------------------------
# Perplexity evaluation (WikiText-2 sliding-window)
# ---------------------------------------------------------------------------

def load_wikitext2(tokenizer, split="test", max_tokens=None):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    enc = tokenizer(text, return_tensors="pt")
    ids = enc.input_ids[0]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    return ids


def evaluate_perplexity(model, tokenizer, device, seq_len=512, max_tokens=None):
    ids = load_wikitext2(tokenizer, split="test", max_tokens=max_tokens)
    model.eval()
    model.to(device)
    model_dtype = next(model.parameters()).dtype

    nlls = []
    n = ids.size(0)
    for begin in tqdm(range(0, n - 1, seq_len), desc="ppl", leave=False):
        end = min(begin + seq_len, n - 1)
        chunk = ids[begin:end + 1].unsqueeze(0).to(device)
        chunk = chunk.to(dtype=torch.long)
        labels = chunk.clone()
        with torch.no_grad():
            out = model(chunk, labels=labels)
        nlls.append(out.loss.float())
    ppl = math.exp(torch.stack(nlls).mean().item())
    return ppl


# ---------------------------------------------------------------------------
# Model size
# ---------------------------------------------------------------------------

def get_model_size_mb(model):
    if _uses_bnb(model):
        import bitsandbytes as bnb
        total_bytes = 0
        seen = set()
        for m in model.modules():
            if isinstance(m, bnb.nn.Linear4bit):
                # 4-bit: weight stored as packed uint8 (2 vals/byte) after quantization
                # quant_state carries scale/zero metadata (~small overhead, ignore)
                pid = id(m.weight)
                if pid not in seen:
                    seen.add(pid)
                    if m.weight.quant_state is not None:
                        # quantized: numel()/2 bytes for the packed 4-bit data
                        total_bytes += m.weight.numel() // 2
                    else:
                        total_bytes += m.weight.numel() * m.weight.element_size()
                if m.bias is not None:
                    bid = id(m.bias)
                    if bid not in seen:
                        seen.add(bid)
                        total_bytes += m.bias.numel() * m.bias.element_size()
            else:
                for p in m.parameters(recurse=False):
                    pid = id(p.data)
                    if pid in seen:
                        continue
                    seen.add(pid)
                    total_bytes += p.numel() * p.element_size()
        return total_bytes / MiB
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
        tmp = f.name
    torch.save(model.state_dict(), tmp)
    size = os.path.getsize(tmp) / MiB
    os.unlink(tmp)
    return size


# ---------------------------------------------------------------------------
# Latency: tokens/sec on a fixed prompt
# ---------------------------------------------------------------------------

def measure_latency_ms(model, tokenizer, device, prompt="The quick brown fox", n_new=50, n_warmup=5, n_runs=20):
    model.eval()
    model.to(device)
    inp = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    gen_kwargs = dict(max_new_tokens=n_new, do_sample=False, use_cache=True,
                      pad_token_id=tokenizer.eos_token_id)
    times = []
    with torch.no_grad():
        for i in range(n_warmup + n_runs):
            t0 = time.perf_counter()
            model.generate(inp, **gen_kwargs)
            t1 = time.perf_counter()
            if i >= n_warmup:
                times.append((t1 - t0) * 1000 / n_new)  # ms per token
    return sum(times) / len(times)


# ---------------------------------------------------------------------------
# Peak VRAM
# ---------------------------------------------------------------------------

def get_peak_vram_mb(model, tokenizer, device, prompt="The quick brown fox"):
    if not torch.cuda.is_available() or device == "cpu":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    inp = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        model(inp)
    return torch.cuda.max_memory_allocated(device) / MiB


# ---------------------------------------------------------------------------
# Structured pruning: FFN width
# ---------------------------------------------------------------------------

def prune_ffn_width(model, sparsity):
    """Remove a fraction of FFN intermediate neurons by L2 importance."""
    for layer in model.model.layers:
        mlp = layer.mlp
        intermediate = mlp.gate_proj.out_features
        n_keep = max(1, int(round(intermediate * (1.0 - sparsity))))
        # SwiGLU importance: combined L2 norm across gate_proj rows, up_proj rows, down_proj cols
        # neuron i output = down_proj[:, i] * silu(gate_proj[i] @ x) * (up_proj[i] @ x)
        gate_norm = mlp.gate_proj.weight.data.norm(dim=1)   # [intermediate]
        up_norm   = mlp.up_proj.weight.data.norm(dim=1)     # [intermediate]
        down_norm = mlp.down_proj.weight.data.norm(dim=0)   # [intermediate]
        importance = gate_norm * up_norm * down_norm
        keep_idx = torch.argsort(importance, descending=True)[:n_keep]
        keep_idx, _ = torch.sort(keep_idx)

        # gate_proj: [intermediate, hidden] → [n_keep, hidden]
        mlp.gate_proj = _prune_linear_rows(mlp.gate_proj, keep_idx)
        # up_proj:   [intermediate, hidden] → [n_keep, hidden]
        mlp.up_proj = _prune_linear_rows(mlp.up_proj, keep_idx)
        # down_proj: [hidden, intermediate] → [hidden, n_keep]
        mlp.down_proj = _prune_linear_cols(mlp.down_proj, keep_idx)

    return model


def _prune_linear_rows(linear, keep_idx):
    keep_idx = keep_idx.to(linear.weight.device)
    new = nn.Linear(
        linear.in_features,
        len(keep_idx),
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    new.weight = nn.Parameter(linear.weight.data[keep_idx].clone())
    if linear.bias is not None:
        new.bias = nn.Parameter(linear.bias.data[keep_idx].clone())
    return new


def _prune_linear_cols(linear, keep_idx):
    keep_idx = keep_idx.to(linear.weight.device)
    new = nn.Linear(
        len(keep_idx),
        linear.out_features,
        bias=linear.bias is not None,
        device=linear.weight.device,
        dtype=linear.weight.dtype,
    )
    new.weight = nn.Parameter(linear.weight.data[:, keep_idx].clone())
    if linear.bias is not None:
        new.bias = nn.Parameter(linear.bias.data.clone())
    return new


# ---------------------------------------------------------------------------
# Structured pruning: depth (layer removal)
# ---------------------------------------------------------------------------

def prune_depth(model, sparsity):
    """Remove a fraction of transformer layers (uniformly spaced, skip first/last)."""
    layers = list(model.model.layers)
    n_total = len(layers)
    n_remove = max(0, int(round(n_total * sparsity)))
    if n_remove == 0:
        return model
    # Remove uniformly-spaced middle layers (skip first and last which are most important)
    candidates = list(range(1, n_total - 1))  # exclude first and last
    step = len(candidates) / n_remove
    remove_idx = set(candidates[int(i * step)] for i in range(n_remove))
    kept = nn.ModuleList([layers[i] for i in range(n_total) if i not in remove_idx])
    model.model.layers = kept
    model.config.num_hidden_layers = len(kept)
    return model


# ---------------------------------------------------------------------------
# Structured pruning: attention heads (GQA-aware)
# ---------------------------------------------------------------------------

def prune_attention_heads(model, sparsity):
    """Remove a fraction of Q attention heads, keeping GQA group structure.

    For GQA models (e.g. Qwen2 with num_heads=14, num_kv_heads=2):
    - Each KV group serves num_heads//num_kv_heads Q heads.
    - We remove the same number of Q heads from every KV group (to stay balanced).
    - n_keep must be divisible by num_kv_heads.
    - Importance: Frobenius norm of q_proj block + o_proj block for each Q head.
    """
    cfg = model.config
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.hidden_size // num_heads
    grp_size = num_heads // num_kv_heads  # Q heads per KV group

    n_remove_total = max(0, int(round(num_heads * sparsity)))
    # Round to nearest multiple of num_kv_heads
    n_remove_total = round(n_remove_total / num_kv_heads) * num_kv_heads
    n_keep_total = num_heads - n_remove_total
    if n_keep_total <= 0:
        n_keep_total = num_kv_heads  # always keep at least 1 per KV group
        n_remove_total = num_heads - n_keep_total

    # Remove evenly from each KV group
    n_remove_per_grp = n_remove_total // num_kv_heads
    n_keep_per_grp = grp_size - n_remove_per_grp

    for layer in model.model.layers:
        attn = layer.self_attn
        q_weight = attn.q_proj.weight.data  # [num_heads * head_dim, hidden]
        o_weight = attn.o_proj.weight.data  # [hidden, num_heads * head_dim]

        keep_idx_all = []
        for g in range(num_kv_heads):
            start = g * grp_size
            group_heads = list(range(start, start + grp_size))
            # Importance per head in this group
            imp = []
            for h in group_heads:
                r0, r1 = h * head_dim, (h + 1) * head_dim
                imp.append(q_weight[r0:r1].norm().item() + o_weight[:, r0:r1].norm().item())
            # Keep top-n by importance
            sorted_local = sorted(range(grp_size), key=lambda i: imp[i], reverse=True)
            keep_local = sorted(sorted_local[:n_keep_per_grp])
            keep_idx_all.extend(group_heads[i] for i in keep_local)

        keep_idx_all.sort()
        row_idx = torch.tensor(
            [i for h in keep_idx_all for i in range(h * head_dim, (h + 1) * head_dim)],
            dtype=torch.long,
            device=q_weight.device,
        )

        # Update q_proj: rows correspond to output Q channels
        new_q = nn.Linear(attn.q_proj.in_features, len(row_idx),
                          bias=attn.q_proj.bias is not None,
                          device=attn.q_proj.weight.device,
                          dtype=attn.q_proj.weight.dtype)
        new_q.weight = nn.Parameter(q_weight[row_idx].clone())
        if attn.q_proj.bias is not None:
            new_q.bias = nn.Parameter(attn.q_proj.bias.data[row_idx].clone())
        attn.q_proj = new_q

        # Update o_proj: cols correspond to input Q channels
        new_o = nn.Linear(len(row_idx), attn.o_proj.out_features,
                          bias=attn.o_proj.bias is not None,
                          device=attn.o_proj.weight.device,
                          dtype=attn.o_proj.weight.dtype)
        new_o.weight = nn.Parameter(o_weight[:, row_idx].clone())
        if attn.o_proj.bias is not None:
            new_o.bias = nn.Parameter(attn.o_proj.bias.data.clone())
        attn.o_proj = new_o

        # Update attention layer metadata
        attn.num_heads = n_keep_total
        attn.num_key_value_groups = n_keep_total // num_kv_heads

    # Update global config
    cfg.num_attention_heads = n_keep_total
    return model


# ---------------------------------------------------------------------------
# Quantization
# ---------------------------------------------------------------------------

def ptq_int8(model):
    """Replace nn.Linear with bitsandbytes int8 layers (4x compression)."""
    import bitsandbytes as bnb

    def _replace(parent, prefix=""):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not isinstance(child, bnb.nn.Linear8bitLt):
                new = bnb.nn.Linear8bitLt(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    has_fp16_weights=False,
                    threshold=6.0,
                )
                new.weight = bnb.nn.Int8Params(
                    child.weight.data.float(),
                    requires_grad=False,
                    has_fp16_weights=False,
                )
                if child.bias is not None:
                    new.bias = nn.Parameter(child.bias.data.clone())
                setattr(parent, name, new)
            else:
                _replace(child, full)

    _replace(model)
    return model


def ptq_fp16(model):
    return model.half()


def ptq_nf4(model):
    """Replace nn.Linear with bitsandbytes NF4 4-bit layers (~8x compression on linears)."""
    import bitsandbytes as bnb

    def _replace(parent, prefix=""):
        for name, child in list(parent.named_children()):
            full = f"{prefix}.{name}" if prefix else name
            if isinstance(child, nn.Linear) and not isinstance(child, bnb.nn.Linear4bit):
                new = bnb.nn.Linear4bit(
                    child.in_features, child.out_features,
                    bias=child.bias is not None,
                    quant_type="nf4",
                    compute_dtype=torch.float16,
                )
                new.weight = bnb.nn.Params4bit(
                    child.weight.data,
                    requires_grad=False,
                    quant_type="nf4",
                )
                if child.bias is not None:
                    new.bias = nn.Parameter(child.bias.data.clone())
                setattr(parent, name, new)
            else:
                _replace(child, full)

    _replace(model)
    return model


# ---------------------------------------------------------------------------
# Fine-tuning (causal LM on WikiText-2 train)
# ---------------------------------------------------------------------------

class WikiText2Dataset(Dataset):
    def __init__(self, tokenizer, split="train", seq_len=512, max_tokens=200_000):
        ids = load_wikitext2(tokenizer, split=split, max_tokens=max_tokens)
        self.chunks = [ids[i:i + seq_len] for i in range(0, len(ids) - seq_len, seq_len)]

    def __len__(self):
        return len(self.chunks)

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {"input_ids": chunk, "labels": chunk.clone()}


def finetune(model, tokenizer, config, device):
    epochs = config.get("finetune_after_prune_epochs", 1)
    lr = config.get("finetune_lr", 2e-5)
    seq_len = config.get("seq_len", 512)
    batch_size = config.get("batch_size", 4)
    grad_accum = config.get("grad_accumulation_steps", 1)
    max_tokens = config.get("finetune_max_tokens", 200_000)

    dataset = WikiText2Dataset(tokenizer, split="train", seq_len=seq_len, max_tokens=max_tokens)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        num_workers=config.get("num_workers", 2))

    model.train()
    model.to(device)
    # Gradient checkpointing reduces activation memory for large models
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model_dtype = next(model.parameters()).dtype
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(loader, desc=f"finetune epoch {epoch+1}/{epochs}", leave=False)):
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)
            if model_dtype != torch.float32:
                pass
            out = model(input_ids, labels=labels)
            loss = out.loss / grad_accum
            loss.backward()
            total_loss += loss.item() * grad_accum
            if (step + 1) % grad_accum == 0 or (step + 1) == len(loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        print(f"  epoch {epoch+1} loss: {total_loss / len(loader):.4f}")

    return model


# ---------------------------------------------------------------------------
# Collect all metrics
# ---------------------------------------------------------------------------

def _uses_bnb(model):
    try:
        import bitsandbytes as bnb
        return any(isinstance(m, (bnb.nn.Linear8bitLt, bnb.nn.Linear4bit)) for m in model.modules())
    except ImportError:
        return False


def _uses_bnb4(model):
    try:
        import bitsandbytes as bnb
        return any(isinstance(m, bnb.nn.Linear4bit) for m in model.modules())
    except ImportError:
        return False


def collect_metrics(model, tokenizer, config, device):
    # bnb int8 requires CUDA; torch.ao int8 runs on CPU
    quant = config.get("quant", "none")
    eval_device = device if (quant != "int8" or _uses_bnb(model)) else "cpu"
    # Always measure latency on same device as PPL for fair comparison
    lat_device = eval_device
    ppl = evaluate_perplexity(model, tokenizer, eval_device,
                               seq_len=config.get("seq_len", 512),
                               max_tokens=config.get("eval_max_tokens", None))
    size_mb = get_model_size_mb(model)
    lat = measure_latency_ms(model, tokenizer, lat_device)
    vram = get_peak_vram_mb(model, tokenizer, device)
    return {"ppl": ppl, "size_mb": size_mb, "latency_ms": lat, "peak_vram_mb": vram}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_name = config.get("model", "Qwen/Qwen2-0.5B")
    dtype = torch.float16 if config.get("load_in_fp16", False) else torch.float32

    print(f"Loading {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load checkpoint if specified
    ckpt = config.get("checkpoint")
    if ckpt and os.path.exists(ckpt):
        print(f"Loading checkpoint {ckpt}")
        state = torch.load(ckpt, map_location="cpu")
        model.load_state_dict(state)

    print("Collecting baseline metrics...")
    baseline = collect_metrics(model, tokenizer, config, device)
    print(f"  baseline PPL: {baseline['ppl']:.2f}  size: {baseline['size_mb']:.3f} MB"
          f"  latency: {baseline['latency_ms']:.3f} ms/tok")

    # --- Pruning ---
    prune_mode = config.get("prune_mode", "none")
    if prune_mode == "ffn_width":
        print(f"Pruning FFN width (sparsity={config['ffn_sparsity']}) ...")
        model = prune_ffn_width(model, config["ffn_sparsity"])
    elif prune_mode == "depth":
        print(f"Pruning depth (sparsity={config['depth_sparsity']}) ...")
        model = prune_depth(model, config["depth_sparsity"])
    elif prune_mode == "depth_plus_ffn":
        print(f"Pruning depth (sparsity={config['depth_sparsity']}) + FFN (sparsity={config['ffn_sparsity']}) ...")
        model = prune_depth(model, config["depth_sparsity"])
        model = prune_ffn_width(model, config["ffn_sparsity"])
    elif prune_mode == "attention_heads":
        print(f"Pruning attention heads (sparsity={config['head_sparsity']}) ...")
        model = prune_attention_heads(model, config["head_sparsity"])
    elif prune_mode == "head_plus_depth":
        print(f"Pruning heads (sparsity={config['head_sparsity']}) + depth (sparsity={config['depth_sparsity']}) ...")
        model = prune_attention_heads(model, config["head_sparsity"])
        model = prune_depth(model, config["depth_sparsity"])

    # --- Fine-tune after prune ---
    if config.get("finetune_after_prune") and prune_mode != "none":
        print(f"Fine-tuning for {config.get('finetune_after_prune_epochs', 1)} epochs ...")
        model = finetune(model, tokenizer, config, device)

    # --- Quantization ---
    quant = config.get("quant", "none")
    if quant == "int8":
        print("Applying dynamic int8 quantization ...")
        model = ptq_int8(model)
    elif quant == "fp16":
        print("Applying fp16 quantization ...")
        model = ptq_fp16(model)
    elif quant == "nf4":
        print("Applying NF4 4-bit quantization ...")
        model = ptq_nf4(model)

    # --- Compressed metrics ---
    print("Collecting compressed metrics...")
    compressed = collect_metrics(model, tokenizer, config, device)

    # --- Report ---
    orig_ppl = baseline["ppl"]
    comp_ppl = compressed["ppl"]
    compression = baseline["size_mb"] / compressed["size_mb"]
    speedup = baseline["latency_ms"] / compressed["latency_ms"]

    print()
    print(f"model:                  {model_name}")
    print(f"task:                   wikitext2_ppl")
    print(f"original_ppl:           {orig_ppl:.4f}")
    print(f"compressed_ppl:         {comp_ppl:.4f}")
    print(f"ppl_delta:              {comp_ppl - orig_ppl:+.4f}")
    print(f"original_size_mb:       {baseline['size_mb']:.3f}")
    print(f"compressed_size_mb:     {compressed['size_mb']:.3f}")
    print(f"compression_ratio:      {compression:.3f}x")
    print(f"latency_original_ms:    {baseline['latency_ms']:.3f}")
    print(f"latency_compressed_ms:  {compressed['latency_ms']:.3f}")
    print(f"speedup:                {speedup:.3f}x")
    print(f"peak_vram_mb:           {compressed['peak_vram_mb']:.1f}")
    print(f"prune_mode:             {prune_mode}")
    print(f"quantization:           {quant}")
    if "ffn_sparsity" in config:
        print(f"ffn_sparsity:           {config['ffn_sparsity']}")
    if "depth_sparsity" in config:
        print(f"depth_sparsity:         {config['depth_sparsity']}")
    if "head_sparsity" in config:
        print(f"head_sparsity:          {config['head_sparsity']}")


if __name__ == "__main__":
    main()
