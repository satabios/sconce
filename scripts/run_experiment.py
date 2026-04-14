#!/usr/bin/env python3
"""Single-experiment runner: config → compress → log."""

import argparse
import copy
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

Byte = 8
KiB = 1024 * Byte
MiB = 1024 * KiB


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="sconce experiment runner")
    p.add_argument("--config", required=True, help="Path to JSON config file")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(cfg):
    model_name = cfg["model"]
    source = cfg.get("model_source", "timm")
    num_classes = cfg.get("num_classes", 10)
    pretrained = cfg.get("pretrained", True)

    if source == "timm":
        import timm
        model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=num_classes,
        )
    elif source == "hf":
        from transformers import AutoModelForImageClassification
        model = AutoModelForImageClassification.from_pretrained(
            model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )
    else:
        raise ValueError(f"Unknown model_source: {source}")

    return model


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(cfg):
    dataset_name = cfg.get("dataset", "cifar10")
    batch_size = cfg.get("batch_size", 128)
    img_size = cfg.get("img_size", 224)
    data_root = cfg.get("data_root", "/tmp/data")

    import torchvision
    from torchvision import transforms

    transform_train = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    transform_test = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if dataset_name == "cifar10":
        train_set = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True, transform=transform_test)
    elif dataset_name == "cifar100":
        train_set = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True, transform=transform_train)
        test_set = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True, transform=transform_test)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    num_workers = cfg.get("num_workers", 4)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    return {"train": train_loader, "test": test_loader}


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def forward_model(model, inputs):
    """Handle both plain tensors and HF model output objects."""
    outputs = model(inputs)
    if hasattr(outputs, "logits"):
        return outputs.logits
    return outputs


@torch.no_grad()
def evaluate(model, loader, device, max_batches=None):
    model.eval()
    model.to(device)
    correct, total = 0, 0
    for i, (images, labels) in enumerate(tqdm(loader, desc="eval", leave=False)):
        if max_batches is not None and i >= max_batches:
            break
        images, labels = images.to(device), labels.to(device)
        logits = forward_model(model, images)
        _, predicted = torch.max(logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    return 100.0 * correct / total


def measure_latency_ms(model, dummy_input, n_warmup=20, n_test=100):
    model.eval()
    model.to("cpu")
    dummy = dummy_input.to("cpu")
    with torch.no_grad():
        for _ in range(n_warmup):
            forward_model(model, dummy)
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_test):
            forward_model(model, dummy)
    return (time.perf_counter() - t0) / n_test * 1000


def get_model_size_mb(model):
    total_bits = sum(
        p.count_nonzero().item() * p.element_size() * Byte
        for p in model.parameters()
    )
    return total_bits / MiB


def get_peak_vram_mb(model, dummy_input, device):
    if not torch.cuda.is_available() or device == "cpu":
        return 0.0
    torch.cuda.reset_peak_memory_stats(device)
    model.to(device)
    with torch.no_grad():
        forward_model(model, dummy_input.to(device))
    return torch.cuda.max_memory_allocated(device) / MiB


# ---------------------------------------------------------------------------
# Fine-tuning (for head adaptation to new dataset)
# ---------------------------------------------------------------------------

def finetune(model, dataloader, device, cfg):
    """Fine-tune the full model or just the head."""
    ft_epochs = cfg.get("finetune_epochs", 3)
    lr = cfg.get("finetune_lr", 1e-3)
    freeze_backbone = cfg.get("freeze_backbone", True)

    if freeze_backbone:
        # Freeze all params, unfreeze the head
        for param in model.parameters():
            param.requires_grad_(False)
        # Unfreeze the classification head
        head_names = ("head", "classifier", "fc", "last_linear")
        for name, module in model.named_modules():
            if any(name.endswith(h) or name == h for h in head_names):
                for param in module.parameters():
                    param.requires_grad_(True)
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        print(f"Frozen backbone — training {sum(p.numel() for p in trainable_params):,} params")
    else:
        trainable_params = list(model.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ft_epochs)

    model.to(device)
    best_acc = 0.0
    for epoch in range(ft_epochs):
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(dataloader["train"], desc=f"epoch {epoch+1}/{ft_epochs}", leave=False):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = forward_model(model, images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        scheduler.step()
        val_acc = evaluate(model, dataloader["test"], device)
        print(f"  epoch {epoch+1}: loss={running_loss/len(dataloader['train']):.4f} val_acc={val_acc:.2f}%")
        if val_acc > best_acc:
            best_acc = val_acc

    # Re-enable all params for subsequent pruning
    for param in model.parameters():
        param.requires_grad_(True)

    return model


# ---------------------------------------------------------------------------
# GMP pruning (magnitude-based unstructured) for transformer Linear layers
# ---------------------------------------------------------------------------

def gmp_prune_transformer(model, sparsity):
    """Apply global magnitude pruning to all nn.Linear weight matrices."""
    import torch.nn.utils.prune as torch_prune
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            torch_prune.l1_unstructured(module, name="weight", amount=sparsity)
    return model


def gmp_remove_masks(model):
    """Make pruning permanent by removing reparameterizations."""
    import torch.nn.utils.prune as torch_prune
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            try:
                torch_prune.remove(module, "weight")
            except ValueError:
                pass
    return model


# ---------------------------------------------------------------------------
# Attention-head pruning (L1-norm based, simpler + more reliable)
# ---------------------------------------------------------------------------

def prune_attention_heads(model, dataloader, device, head_sparsity=0.25):
    """
    Score each attention head by L1-norm of its output-projection weights.
    Zero out the columns in the output projection for the least important heads.
    Works on timm ViT (model.blocks[i].attn).
    """
    if not hasattr(model, "blocks"):
        print("  [attention-head pruning] model has no .blocks, skipping")
        return model

    num_layers = len(model.blocks)
    first_attn = model.blocks[0].attn
    num_heads = first_attn.num_heads
    head_dim = first_attn.head_dim

    # Score: L1 norm of output-projection columns for each head
    head_importance = torch.zeros(num_layers, num_heads)
    for layer_idx, block in enumerate(model.blocks):
        proj_w = block.attn.proj.weight.data  # (d, d)
        for head_idx in range(num_heads):
            start = head_idx * head_dim
            end = start + head_dim
            head_importance[layer_idx, head_idx] = proj_w[:, start:end].abs().mean()

    # Prune bottom `head_sparsity` fraction globally
    flat = head_importance.view(-1)
    k = max(1, int(flat.numel() * head_sparsity))
    threshold = flat.kthvalue(k).values.item()

    pruned_count = 0
    for layer_idx, block in enumerate(model.blocks):
        for head_idx in range(num_heads):
            if head_importance[layer_idx, head_idx].item() <= threshold:
                start = head_idx * head_dim
                end = start + head_dim
                with torch.no_grad():
                    block.attn.proj.weight.data[:, start:end] = 0.0
                pruned_count += 1

    total = num_layers * num_heads
    print(f"  Pruned {pruned_count}/{total} attention heads ({100*pruned_count/total:.1f}%)")
    return model


# ---------------------------------------------------------------------------
# FFN neuron pruning (structured: zero lowest-L1 intermediate neurons in MLP)
# ---------------------------------------------------------------------------

def prune_ffn_neurons(model, ffn_sparsity=0.25):
    """
    Prune FFN neurons in each transformer block's MLP by zeroing the
    output weights of the least-important intermediate neurons.
    Works on timm ViT (model.blocks[i].mlp.fc1 / fc2).
    """
    if not hasattr(model, "blocks"):
        print("  [FFN pruning] model has no .blocks, skipping")
        return model

    for layer_idx, block in enumerate(model.blocks):
        mlp = block.mlp
        fc1 = mlp.fc1  # (hidden_dim → intermediate_dim)
        fc2 = mlp.fc2  # (intermediate_dim → hidden_dim)

        # Score each neuron by L1 norm of its fc2 input weight column
        neuron_importance = fc2.weight.data.abs().mean(dim=0)  # (intermediate_dim,)
        k = max(1, int(len(neuron_importance) * ffn_sparsity))
        _, prune_idx = neuron_importance.topk(k, largest=False)

        with torch.no_grad():
            fc1.weight.data[prune_idx, :] = 0.0
            if fc1.bias is not None:
                fc1.bias.data[prune_idx] = 0.0
            fc2.weight.data[:, prune_idx] = 0.0

    print(f"  FFN: zeroed {ffn_sparsity*100:.0f}% of intermediate neurons per block")
    return model


# ---------------------------------------------------------------------------
# PTQ int8 (CPU-only, via PyTorch static quant)
# ---------------------------------------------------------------------------

def ptq_int8(model, dataloader, num_calibration_batches=32):
    """Static post-training quantization via torch.ao.quantization."""
    import torch.ao.quantization as tq

    model_q = copy.deepcopy(model).to("cpu")

    # For transformer models, use per-tensor dynamic quant (works universally)
    model_q = torch.ao.quantization.quantize_dynamic(
        model_q,
        {nn.Linear},
        dtype=torch.qint8,
    )
    return model_q


# ---------------------------------------------------------------------------
# Metrics collection
# ---------------------------------------------------------------------------

def collect_metrics(model, dataloader, device, cfg, label):
    input_shape = [1] + list(next(iter(dataloader["test"]))[0].shape[1:])
    dummy = torch.randn(input_shape)

    acc = evaluate(model, dataloader["test"], device)
    size_mb = get_model_size_mb(model)
    lat_ms = measure_latency_ms(model, dummy)
    peak_vram = get_peak_vram_mb(model, dummy, device)

    return {
        "label": label,
        "accuracy": round(acc, 4),
        "size_mb": round(size_mb, 3),
        "latency_ms": round(lat_ms, 3),
        "peak_vram_mb": round(peak_vram, 2),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = json.load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    # ---- Load data ----
    print("Loading dataset...")
    dataloader = load_dataset(cfg)

    # ---- Load / restore model ----
    checkpoint_path = cfg.get("checkpoint")
    print(f"Loading model: {cfg['model']}")
    model = load_model(cfg)

    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"  Restoring checkpoint: {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    elif cfg.get("finetune_epochs", 0) > 0:
        print("Fine-tuning model on dataset...")
        model = finetune(model, dataloader, device, cfg)
        if checkpoint_path:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    # ---- Baseline metrics ----
    t_start = time.perf_counter()
    print("\nCollecting baseline metrics...")
    baseline = collect_metrics(model, dataloader, device, cfg, "baseline")
    original_acc = baseline["accuracy"]
    original_size = baseline["size_mb"]
    original_latency = baseline["latency_ms"]

    prune_mode = cfg.get("prune_mode", "none")
    quant = cfg.get("quant", "none")

    compressed_model = copy.deepcopy(model)

    # ---- Pruning ----
    sens_time_min = 0.0
    if prune_mode == "GMP":
        sparsity = cfg.get("sparsity", 0.5)
        print(f"\nGMP pruning — sparsity={sparsity}")
        compressed_model = gmp_prune_transformer(compressed_model, sparsity)
        compressed_model = gmp_remove_masks(compressed_model)

    elif prune_mode == "attention":
        head_sparsity = cfg.get("head_sparsity", 0.25)
        print(f"\nAttention-head pruning — head_sparsity={head_sparsity}")
        sens_t0 = time.perf_counter()
        compressed_model = prune_attention_heads(
            compressed_model, dataloader, device, head_sparsity)
        sens_time_min = (time.perf_counter() - sens_t0) / 60

    elif prune_mode == "ffn":
        ffn_sparsity = cfg.get("ffn_sparsity", 0.25)
        print(f"\nFFN neuron pruning — ffn_sparsity={ffn_sparsity}")
        compressed_model = prune_ffn_neurons(compressed_model, ffn_sparsity)

    elif prune_mode == "none":
        pass  # baseline only
    if prune_mode != "none" and cfg.get("finetune_after_prune", False):
        ft_cfg = {**cfg, "finetune_epochs": cfg.get("finetune_after_prune_epochs", 3),
                  "freeze_backbone": False}
        print("\nFine-tuning after pruning...")
        compressed_model = finetune(compressed_model, dataloader, device, ft_cfg)

    # ---- Quantization ----
    if quant == "int8":
        print("\nPTQ int8 (dynamic quantization)...")
        compressed_model = ptq_int8(compressed_model, dataloader)
    elif quant == "none":
        pass

    # ---- Compressed metrics ----
    print("\nCollecting compressed metrics...")
    # Dynamic/static quantized models must run on CPU
    eval_device = "cpu" if quant in ("int8", "int4") else device
    compressed = collect_metrics(compressed_model, dataloader, eval_device, cfg, "compressed")

    total_time_min = (time.perf_counter() - t_start) / 60
    compression_ratio = original_size / compressed["size_mb"] if compressed["size_mb"] > 0 else 0.0
    speedup = original_latency / compressed["latency_ms"] if compressed["latency_ms"] > 0 else 0.0
    accuracy_drop = compressed["accuracy"] - original_acc

    # ---- Print summary (grep-able) ----
    model_tag = cfg["model"].replace("/", "_")
    print(f"""
---
model:                  {model_tag}
task:                   {cfg.get("dataset", "cifar10")}
original_accuracy:      {original_acc:.4f}
compressed_accuracy:    {compressed["accuracy"]:.4f}
accuracy_drop:          {accuracy_drop:.4f}
original_size_mb:       {original_size:.3f}
compressed_size_mb:     {compressed["size_mb"]:.3f}
compression_ratio:      {compression_ratio:.3f}x
latency_original_ms:    {original_latency:.3f}
latency_compressed_ms:  {compressed["latency_ms"]:.3f}
speedup:                {speedup:.3f}x
peak_vram_mb:           {compressed["peak_vram_mb"]:.1f}
prune_mode:             {prune_mode}
sparsity:               {cfg.get("sparsity", cfg.get("head_sparsity", 0.0))}
quantization:           {quant}
sensitivity_scan_time_min: {sens_time_min:.2f}
total_time_min:         {total_time_min:.2f}
---""")

    return 0


if __name__ == "__main__":
    sys.exit(main())
