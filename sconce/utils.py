"""
sconce/utils.py
Shared utilities: constants, device helpers, GPU discovery, low-level pruning ops.
All other modules import from here to avoid duplication.
"""
from __future__ import annotations

import queue
import subprocess
import warnings

import torch
import torch.nn as nn
from collections import OrderedDict

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

Byte = 8
KiB  = 1024 * Byte
MiB  = 1024 * KiB
GiB  = 1024 * MiB

# ---------------------------------------------------------------------------
# Warning suppression
# ---------------------------------------------------------------------------

def _suppress_warnings() -> None:
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("default")
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=ImportWarning)


_suppress_warnings()

# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def get_device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if dev.type == "cuda":
        torch.cuda.synchronize()
    return dev

# ---------------------------------------------------------------------------
# GPU discovery helpers
# ---------------------------------------------------------------------------

def _nvidia_smi_free_gb() -> list[tuple[int, float]]:
    """Return [(gpu_id, free_gb), ...] via nvidia-smi. Empty list if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free",
             "--format=csv,noheader,nounits"],
            text=True, stderr=subprocess.DEVNULL,
        )
        result = []
        for line in out.strip().splitlines():
            parts = line.split(",")
            if len(parts) == 2:
                result.append((int(parts[0].strip()), int(parts[1].strip()) / 1024))
        return result
    except Exception:
        return []


def _measure_model_vram_gb(model: nn.Module, activation_multiplier: float = 1.5) -> float:
    """Estimate model VRAM footprint in GB × multiplier (parameter + buffer byte count)."""
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buf_bytes   = sum(b.numel() * b.element_size() for b in model.buffers())
    return (param_bytes + buf_bytes) / (1024 ** 3) * activation_multiplier


def _build_gpu_worker_pool(
    gpu_list: list[tuple[int, float]],
    model: nn.Module,
    safety_net_gb: float,
    activation_multiplier: float,
    verbose: bool,
    label: str,
) -> tuple[queue.Queue, int, float]:
    """Build a GPU worker queue from *gpu_list*.

    Allocates ``max(1, floor((free_gb - safety_net_gb) / model_vram_gb))`` workers
    per GPU.  Prints the allocation table when *verbose* is True.

    Returns:
        (gpu_queue, total_slots, model_vram_gb)
    """
    model_vram_gb = _measure_model_vram_gb(model, activation_multiplier)
    gpu_queue: queue.Queue = queue.Queue()
    alloc_rows = []
    for gpu_id, free_gb in gpu_list:
        n_workers = max(1, int((free_gb - safety_net_gb) / model_vram_gb))
        for _ in range(n_workers):
            gpu_queue.put(gpu_id)
        alloc_rows.append((gpu_id, free_gb, n_workers))
    total_slots = gpu_queue.qsize()
    if verbose:
        print(f"\n[{label}]  model_vram={model_vram_gb:.2f} GB  safety_net={safety_net_gb} GB")
        print(f"  {'GPU':>4}  {'free_GB':>8}  {'workers':>8}")
        for gpu_id, free_gb, nw in alloc_rows:
            print(f"  {gpu_id:>4}  {free_gb:>8.1f}  {nw:>8}")
    return gpu_queue, total_slots, model_vram_gb

# ---------------------------------------------------------------------------
# Low-level pruning utilities
# ---------------------------------------------------------------------------

# Module types eligible for structured (CWP-style) pruning.
PRUNABLE_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _get_example_inputs(dataloader, device=None):
    """Return a batch-size-1 example input from *dataloader*.

    Handles three common batch formats:
    - ``(Tensor, label)``  — standard classification / image
    - ``dict``             — HuggingFace-style (input_ids, attention_mask, …)
    - bare ``Tensor``      — unlabelled datasets

    Returns a ``Tensor`` or ``dict[str, Tensor]``.
    """
    batch = next(iter(dataloader))

    if isinstance(batch, dict):
        x = {k: v[:1] for k, v in batch.items() if isinstance(v, torch.Tensor)}
        if device is not None:
            x = {k: v.to(device) for k, v in x.items()}
        return x

    if isinstance(batch, (list, tuple)):
        x = batch[0]
    else:
        x = batch

    x = x[:1]
    if device is not None:
        x = x.to(device)
    return x


def _collect_conv_bn(
    model: nn.Module,
) -> tuple[list[nn.Conv2d], list[nn.BatchNorm2d]]:
    """Return (all_convs, all_bns) collected in traversal order from *model*."""
    all_convs: list[nn.Conv2d] = []
    all_bns: list[nn.BatchNorm2d] = []

    def _walk(obj):
        if isinstance(obj, nn.Conv2d):
            all_convs.append(obj)
        elif isinstance(obj, nn.BatchNorm2d):
            all_bns.append(obj)
        elif isinstance(obj, list):
            for child in obj:
                _walk(child)
        elif isinstance(obj, OrderedDict):
            for child in obj.values():
                _walk(child)
        elif hasattr(obj, "children"):
            for child in obj.children():
                _walk(child)

    _walk(model)
    return all_convs, all_bns


def _structured_zero_prune(module: nn.Module, sparsity: float) -> None:
    """Zero out the lowest-importance output rows/channels of a prunable module.

    Works for **any** ``nn.Linear``, ``nn.Conv1d/2d/3d`` — no dependency graph
    needed and no dimension changes, so the model's forward pass stays valid.
    Importance = L2 norm of each output unit across all input dimensions.
    """
    if not isinstance(module, PRUNABLE_MODULES):
        return
    with torch.no_grad():
        w = module.weight
        importance = w.view(w.shape[0], -1).norm(dim=1)
        n_prune = round(w.shape[0] * sparsity)
        if n_prune <= 0:
            return
        prune_idx = importance.argsort()[:n_prune]
        module.weight[prune_idx] = 0.0
        if getattr(module, 'bias', None) is not None:
            module.bias[prune_idx] = 0.0


def _cwp_prune_module(model, current_module, sparsity, example_inputs):
    """Apply structured pruning to *current_module* inside *model*.

    Tries ``torch_pruning.MetaPruner`` first (handles dependency propagation
    automatically for both CNN and Transformer graphs).  Falls back to
    ``_structured_zero_prune`` when torch_pruning is unavailable.
    """
    try:
        import torch_pruning as tp
        pruner = tp.pruner.MetaPruner(
            model,
            example_inputs,
            importance=tp.importance.MagnitudeImportance(p=2, group_reduction='mean'),
            pruning_ratio=0,
            pruning_ratio_dict={current_module: sparsity},
        )
        pruner.step()
    except ImportError:
        _structured_zero_prune(current_module, sparsity)
