#!/usr/bin/env python3
"""Sensitivity scanner: VRAM-aware parallel sparsity sweep across GPUs.

Uses octopus GPU discovery + compute_worker_allocation to determine how many
concurrent experiment workers each GPU can host, then spawns subprocesses with
a staggered launch to avoid CUDA init races.

All scan dimensions share a single GPU pool so VRAM is maximally utilized.
Use --scan-dim all to sweep ffn+depth+attn simultaneously.

Usage:
  # Sweep all three dims in one pool (maximizes GPU utilization):
  python scripts/run_sensitivity_scan.py \\
      --config scripts/configs/eva02/01_baseline.json \\
      --scan-dim all --steps 32 --output-dir results/

  # Single dim:
  python scripts/run_sensitivity_scan.py \\
      --config scripts/configs/eva02/01_baseline.json \\
      --scan-dim ffn --steps 8 --output-dir results/

  # Comma-sep subset:
  python scripts/run_sensitivity_scan.py \\
      --config scripts/configs/eva02/01_baseline.json \\
      --scan-dim ffn,depth --steps 16 --output-dir results/

  # Override detected GPUs and worker cap:
  python scripts/run_sensitivity_scan.py \\
      --config scripts/configs/eva02/01_baseline.json \\
      --scan-dim all --steps 32 \\
      --gpus 0,1,2,3 --max-workers 4 --safety-net-gb 2.0
"""

import argparse
import json
import math
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

# scripts/ is the cwd when running this script; octopus/ is a sibling package.
_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _SCRIPTS_DIR)

from octopus.discovery import discover_gpus, compute_worker_allocation  # noqa: E402

RUNNER = os.path.join(_SCRIPTS_DIR, "run_experiment.py")

# Seconds between consecutive worker launches on the *same* GPU.
# Prevents CUDA context-init races (BFCArena / CUBLAS_STATUS_ALLOC_FAILED).
# Different-GPU workers are launched in parallel (no stagger).
# From parallel_sensitivity.py: "WORKER_LAUNCH_STAGGER_SECONDS = 2.0"
LAUNCH_STAGGER_S: float = 2.0

SCAN_DIM_MAP = {
    "depth": ("depth_sparsity", "depth"),
    "ffn":   ("ffn_sparsity",   "ffn_structural"),
    "attn":  ("head_sparsity",  "attention_structural"),
}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="VRAM-aware sensitivity scanner")
    p.add_argument("--config",        required=True,              help="Base config JSON")
    p.add_argument(
        "--scan-dim",
        required=True,
        help=(
            "Dimension(s) to scan: comma-sep subset of {ffn,depth,attn} "
            "or 'all' to run all three in one shared GPU pool. "
            "E.g.: --scan-dim all  OR  --scan-dim ffn,depth"
        ),
    )
    p.add_argument("--steps",         type=int,   default=8,      help="Sparsity levels to sweep per dim")
    p.add_argument("--min-sparsity",  type=float, default=0.1)
    p.add_argument("--max-sparsity",  type=float, default=0.9)
    p.add_argument("--gpus",          default=None,               help="Comma-sep GPU IDs (default: auto)")
    p.add_argument("--max-workers",   type=int,   default=None,   help="Cap total concurrent workers")
    p.add_argument("--safety-net-gb", type=float, default=2.0,    help="VRAM headroom reserved per GPU (GB)")
    p.add_argument("--output-dir",    default=None,               help="Directory for TSV output files (default: cwd)")
    return p.parse_args()


def resolve_scan_dims(scan_dim_arg: str) -> list[str]:
    """Expand 'all' → all dims; validate comma-sep list."""
    if scan_dim_arg.strip().lower() == "all":
        return list(SCAN_DIM_MAP.keys())
    dims = [d.strip() for d in scan_dim_arg.split(",")]
    for d in dims:
        if d not in SCAN_DIM_MAP:
            raise ValueError(
                f"Unknown scan dim {d!r}. Valid: {list(SCAN_DIM_MAP.keys())} or 'all'"
            )
    return dims


# ---------------------------------------------------------------------------
# VRAM probe — measure peak VRAM of one experiment via dry run
# ---------------------------------------------------------------------------

def probe_vram_gb(base_cfg: dict, gpu_id: int) -> float:
    """Spawn a subprocess that loads the checkpoint, runs one forward pass,
    and prints VRAM usage.  Returns measured GB (default 4.0 on failure).

    Probe uses batch_size=1 to get minimum model-weight VRAM footprint;
    a 2× multiplier covers activation overhead across the full eval batch.
    """
    probe = f"""\
import sys, torch, os, traceback
sys.path.insert(0, {repr(_SCRIPTS_DIR)})
from run_experiment import load_model

try:
    cfg = {repr(base_cfg)}
    device = "cuda:0"
    model = load_model(cfg)
    model.eval()
    ckpt = cfg.get("checkpoint")
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=False)
    # Initialize CUDA context before resetting stats
    torch.cuda.set_device(0)
    _ = torch.zeros(1, device=device)
    # Reset BEFORE moving model to GPU so peak captures weights + activations
    torch.cuda.reset_peak_memory_stats(0)
    model = model.to(device)
    img_size = cfg.get("img_size", 224)
    # Use batch_size=1 for probe; multiply by 2 below to account for eval overhead
    dummy = torch.randn(1, 3, img_size, img_size, device=device,
                        dtype=next(model.parameters()).dtype)
    with torch.no_grad():
        _ = model(dummy)
    vram_gb = torch.cuda.max_memory_allocated(0) / (1024**3)
    # 2x multiplier: covers eval batch activations + pruning overhead
    vram_gb = vram_gb * 2.0
    print(f"PROBE_VRAM_GB={{vram_gb:.4f}}", flush=True)
except Exception:
    traceback.print_exc()
    print("PROBE_FAILED", flush=True)
"""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    try:
        result = subprocess.run(
            [sys.executable, "-c", probe],
            capture_output=True, text=True, env=env, timeout=180,
        )
        for line in result.stdout.splitlines():
            if line.startswith("PROBE_VRAM_GB="):
                gb = float(line.split("=", 1)[1])
                return gb
        print(f"  [probe] warning: PROBE_VRAM_GB not found in output; defaulting to 4.0 GB")
        if result.stdout.strip():
            print(f"  [probe] stdout: {result.stdout.strip()[:600]}")
        if result.stderr.strip():
            # Filter out noisy pydantic warnings, show real errors
            stderr_lines = [
                ln for ln in result.stderr.splitlines()
                if "UnsupportedFieldAttributeWarning" not in ln
                and "pydantic" not in ln
            ]
            if stderr_lines:
                print(f"  [probe] stderr: {chr(10).join(stderr_lines[:20])}")
    except subprocess.TimeoutExpired:
        print("  [probe] timed out — defaulting to 4.0 GB")
    except Exception as e:
        print(f"  [probe] failed ({e}) — defaulting to 4.0 GB")
    return 4.0


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def make_scan_config(base_cfg: dict, sparsity_key: str, prune_mode: str, sparsity: float) -> dict:
    cfg = {**base_cfg}
    cfg["prune_mode"]          = prune_mode
    cfg[sparsity_key]          = round(sparsity, 6)
    cfg["finetune_after_prune"] = False
    cfg["finetune_epochs"]      = 0
    cfg.pop("finetune_after_prune_epochs", None)
    return cfg


# ---------------------------------------------------------------------------
# Subprocess runner + result parser
# ---------------------------------------------------------------------------

def run_subprocess(gpu_id: int, cfg_path: str):
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
    result = subprocess.run(
        [sys.executable, RUNNER, "--config", cfg_path],
        capture_output=True, text=True, env=env,
    )
    return result.stdout, result.stderr, result.returncode


def parse_result_block(stdout: str) -> dict:
    in_block = False
    data = {}
    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == "---":
            in_block = not in_block
            continue
        if in_block and ":" in stripped:
            k, _, v = stripped.partition(":")
            data[k.strip()] = v.strip()
    return data


# ---------------------------------------------------------------------------
# Worker thread — picks GPU from pool, runs job, returns GPU to pool
# ---------------------------------------------------------------------------

def worker(gpu_queue: queue.Queue, dim: str, sp: float, cfg_path: str,
           results: list, lock: threading.Lock, print_lock: threading.Lock):
    gpu = gpu_queue.get()
    try:
        with print_lock:
            print(f"  [GPU {gpu}] {dim} sparsity={sp:.3f}  starting...")
        t0 = time.time()
        stdout, stderr, rc = run_subprocess(gpu, cfg_path)
        elapsed = time.time() - t0

        if rc != 0:
            with print_lock:
                print(f"  [GPU {gpu}] {dim} sparsity={sp:.3f}  FAILED (rc={rc}, {elapsed:.0f}s)")
                if stderr:
                    print("    stderr:", stderr[-400:].strip())
            data = None
        else:
            data = parse_result_block(stdout)
            with print_lock:
                print(
                    f"  [GPU {gpu}] {dim} sparsity={sp:.3f}  done — "
                    f"acc={data.get('compressed_accuracy', '?')}  "
                    f"drop={data.get('accuracy_drop', '?')}  "
                    f"({elapsed:.0f}s)"
                )
        with lock:
            results.append((dim, sp, data))
    finally:
        gpu_queue.put(gpu)
        try:
            os.unlink(cfg_path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Staggered launch (from parallel_sensitivity.py pattern)
# Workers on *different* GPUs launch in parallel.
# Workers on the *same* GPU are staggered by LAUNCH_STAGGER_S seconds to
# prevent CUDA init races (BFCArena / CUBLAS_STATUS_ALLOC_FAILED).
# ---------------------------------------------------------------------------

def launch_staggered(threads_by_gpu: dict[int, list[threading.Thread]]) -> list[threading.Thread]:
    """Launch threads in staggered order: all GPU-0-workers, sleep, GPU-1-workers, ...
    Actually: interleave by slot index so same-GPU workers are staggered,
    different-GPU workers launch together.

    E.g. with 3 GPUs, 2 workers each (slots 0,1):
      slot 0: GPU0-w0, GPU1-w0, GPU2-w0 (all start simultaneously)
      sleep LAUNCH_STAGGER_S
      slot 1: GPU0-w1, GPU1-w1, GPU2-w1 (all start simultaneously)
    """
    all_threads = []
    max_per_gpu = max(len(v) for v in threads_by_gpu.values()) if threads_by_gpu else 0

    for slot in range(max_per_gpu):
        slot_threads = [
            workers[slot]
            for workers in threads_by_gpu.values()
            if slot < len(workers)
        ]
        for t in slot_threads:
            t.start()
            all_threads.append(t)

        if slot < max_per_gpu - 1:
            print(
                f"  [stagger] sleeping {LAUNCH_STAGGER_S:.0f}s before slot {slot + 1} "
                f"(prevents CUDA init races on same-GPU workers) ..."
            )
            time.sleep(LAUNCH_STAGGER_S)

    return all_threads


# ---------------------------------------------------------------------------
# Table printing + TSV
# ---------------------------------------------------------------------------

def print_table(results: list, scan_dim: str) -> list[dict]:
    # results: list of (dim, sp, data)  — filter to this dim
    dim_results = [(sp, data) for (d, sp, data) in results if d == scan_dim]
    dim_results.sort(key=lambda x: x[0])
    hdr = (
        f"{'sparsity':>10}  {'orig_acc%':>10}  {'comp_acc%':>10}  "
        f"{'drop%':>8}  {'size_mb':>9}  {'lat_ms':>8}  {'speedup':>8}"
    )
    print(f"\nSensitivity scan  —  dim={scan_dim}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for sp, data in dim_results:
        if data is None:
            print(f"{sp:>10.3f}  FAILED")
            continue
        orig_acc = data.get("original_accuracy",     "")
        comp_acc = data.get("compressed_accuracy",   "")
        drop     = data.get("accuracy_drop",         "")
        size     = data.get("compressed_size_mb",    "")
        lat      = data.get("latency_compressed_ms", "")
        speedup  = data.get("speedup",               "")
        print(
            f"{sp:>10.3f}  {orig_acc:>10}  {comp_acc:>10}  "
            f"{drop:>8}  {size:>9}  {lat:>8}  {speedup:>8}"
        )
        rows.append({
            "sparsity":            sp,
            "original_accuracy":   orig_acc,
            "compressed_accuracy": comp_acc,
            "accuracy_drop":       drop,
            "compressed_size_mb":  size,
            "latency_ms":          lat,
            "speedup":             speedup,
        })
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        base_cfg = json.load(f)

    scan_dims = resolve_scan_dims(args.scan_dim)
    n = args.steps
    sparsities = [
        args.min_sparsity + i * (args.max_sparsity - args.min_sparsity) / max(n - 1, 1)
        for i in range(n)
    ]
    total_jobs = len(scan_dims) * n

    # ── 1. Discover GPUs via octopus ────────────────────────────────────────
    gpu_filter = [int(g) for g in args.gpus.split(",")] if args.gpus else None
    gpus = discover_gpus(device_ids=gpu_filter)
    print(f"\nDiscovered {len(gpus)} GPU(s):")
    for g in gpus:
        print(f"  GPU {g.device_id}: {g.name}  {g.available_vram_gb:.1f} GB free")

    # ── 2. Probe VRAM usage per experiment ───────────────────────────────────
    probe_gpu = gpus[0].device_id
    print(f"\nProbing VRAM usage on GPU {probe_gpu} ...")
    model_vram_gb = probe_vram_gb(base_cfg, probe_gpu)
    print(f"  Peak VRAM per experiment: {model_vram_gb:.2f} GB")

    # ── 3. Compute VRAM-aware worker allocation ──────────────────────────────
    plan = compute_worker_allocation(
        gpus,
        model_vram_gb=model_vram_gb,
        safety_net_gb=args.safety_net_gb,
        max_workers=args.max_workers,
    )

    print(f"\nWorker allocation  (safety_net={args.safety_net_gb:.1f} GB/GPU):")
    col = 10
    print(f"  {'GPU':>{col}}  {'free_GB':>{col}}  {'workers':>{col}}")
    print(f"  {'-'*col}  {'-'*col}  {'-'*col}")
    for alloc in plan.allocations:
        gpu_info = next(g for g in gpus if g.device_id == alloc.device_id)
        print(
            f"  {alloc.device_id:>{col}}  {gpu_info.available_vram_gb:>{col}.1f}  {alloc.num_workers:>{col}}"
        )
    print(
        f"  Total: {plan.total_workers} worker slot(s) for {total_jobs} jobs "
        f"({len(scan_dims)} dim(s) × {n} step(s))\n"
        f"  Estimated rounds: {math.ceil(total_jobs / plan.total_workers)}"
    )

    dims_str = "+".join(scan_dims)
    print(
        f"\nSensitivity scan: model={base_cfg['model']}  dims={dims_str}  "
        f"steps={n}  range=[{args.min_sparsity}, {args.max_sparsity}]"
    )

    # ── 4. Build GPU pool queue — interleaved round-robin across GPUs ─────────
    gpu_queue: queue.Queue = queue.Queue()
    _max_w = max((alloc.num_workers for alloc in plan.allocations), default=0)
    for _slot in range(_max_w):
        for alloc in plan.allocations:
            if _slot < alloc.num_workers:
                gpu_queue.put(alloc.device_id)

    results: list = []
    lock = threading.Lock()
    print_lock = threading.Lock()

    # ── 5. Write temp configs + build threads ────────────────────────────────
    # All dims share the same GPU pool: jobs interleaved across dims so load
    # spreads evenly from the start rather than exhausting one dim first.
    gpu_slots: list[int] = []
    for _slot in range(_max_w):
        for alloc in plan.allocations:
            if _slot < alloc.num_workers:
                gpu_slots.append(alloc.device_id)

    threads_by_gpu: dict[int, list[threading.Thread]] = {
        alloc.device_id: [] for alloc in plan.allocations
    }

    # Build flat job list interleaved by dim so different dims start together:
    # [(dim0,sp0),(dim1,sp0),(dim2,sp0),(dim0,sp1),(dim1,sp1),...]
    all_jobs: list[tuple[str, float]] = []
    for sp_idx in range(n):
        for dim in scan_dims:
            all_jobs.append((dim, sparsities[sp_idx]))

    for job_idx, (dim, sp) in enumerate(all_jobs):
        sparsity_key, prune_mode = SCAN_DIM_MAP[dim]
        cfg = make_scan_config(base_cfg, sparsity_key, prune_mode, sp)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
        json.dump(cfg, tmp)
        tmp.close()

        t = threading.Thread(
            target=worker,
            args=(gpu_queue, dim, sp, tmp.name, results, lock, print_lock),
            daemon=True,
        )
        assigned_gpu = gpu_slots[job_idx % len(gpu_slots)]
        threads_by_gpu[assigned_gpu].append(t)

    # ── 6. Staggered launch ──────────────────────────────────────────────────
    all_threads = launch_staggered(threads_by_gpu)
    for t in all_threads:
        t.join()

    # ── 7. Print table + save TSV per dim ────────────────────────────────────
    out_dir = args.output_dir or "."
    os.makedirs(out_dir, exist_ok=True)
    model_tag = base_cfg.get("model", "model").replace("/", "_")

    for dim in scan_dims:
        rows = print_table(results, dim)
        if rows:
            out_path = os.path.join(out_dir, f"sensitivity_{model_tag}_{dim}.tsv")
            with open(out_path, "w") as f:
                headers = list(rows[0].keys())
                f.write("\t".join(headers) + "\n")
                for row in rows:
                    f.write("\t".join(str(row.get(h, "")) for h in headers) + "\n")
            print(f"  Saved → {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
