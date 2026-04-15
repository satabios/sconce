from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

ShardingMode = Literal["tp", "pp", "none"]


@dataclass(frozen=True)
class GPUInfo:
    """Snapshot of one physical GPU at discovery time."""

    device_id: int
    name: str
    total_vram_gb: float
    available_vram_gb: float
    compute_capability: tuple[int, int]


@dataclass(frozen=True)
class VRAMProfile:
    """Result of a dry-run VRAM profiling pass."""

    peak_vram_bytes: int
    peak_vram_gb: float
    model_params_bytes: int
    activation_peak_bytes: int
    profiled_on_device: int


@dataclass(frozen=True)
class WorkerAllocation:
    """Per-GPU worker assignment."""

    device_id: int
    num_workers: int
    vram_per_worker_gb: float
    reserved_safety_gb: float


@dataclass
class PoolPlan:
    """Complete plan for how workers map to GPUs."""

    allocations: list[WorkerAllocation]
    total_workers: int
    sharding_required: bool
    sharding_strategy: Optional[Literal["tp", "pp"]]
    gpus_per_shard: int  # 1 when no sharding
