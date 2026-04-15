# octopus — GPU discovery and VRAM-aware allocation
# Copied from https://github.com/satabios/autoresearch/tree/octopus/octopus
from octopus.discovery import discover_gpus, compute_worker_allocation, poll_gpu_memory
from octopus._types import GPUInfo, PoolPlan, WorkerAllocation, VRAMProfile
from octopus.exceptions import InsufficientVRAMError, NoGPUsFoundError, WorkerOOMError

__all__ = [
    "discover_gpus",
    "compute_worker_allocation",
    "poll_gpu_memory",
    "GPUInfo",
    "PoolPlan",
    "WorkerAllocation",
    "VRAMProfile",
    "InsufficientVRAMError",
    "NoGPUsFoundError",
    "WorkerOOMError",
]
