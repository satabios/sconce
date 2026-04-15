class OctopusError(Exception):
    """Base for all octopus errors."""


class InsufficientVRAMError(OctopusError):
    """Model does not fit on any single GPU, and sharding is disabled."""


class ProfilingError(OctopusError):
    """Dry-run VRAM profiling failed."""


class WorkerOOMError(OctopusError):
    """A worker hit OOM during inference or model loading.

    Attributes:
        worker_id: which worker failed (Ray actor index or name).
        gpu_id: which physical GPU.
        oom_pattern: which OOM signature was detected in logs.
        suggested_sn_gb: recommended safety_net_gb to prevent recurrence.
    """

    OOM_PATTERNS = [
        "BFCArena",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "GPU session init failed",
        "out of memory",
        "CUDA out of memory",
    ]

    def __init__(
        self,
        msg: str,
        worker_id=None,
        gpu_id=None,
        current_sn: float = 0.0,
        oom_pattern: str = "",
    ) -> None:
        self.worker_id = worker_id
        self.gpu_id = gpu_id
        self.oom_pattern = oom_pattern
        self.suggested_sn_gb = current_sn + 1.0
        detail = (
            f"\n  Suggested: increase safety_net_gb to {self.suggested_sn_gb:.1f} GB"
            f" or reduce max_workers."
        )
        if worker_id is not None:
            detail = f" (worker={worker_id}, gpu={gpu_id})" + detail
        super().__init__(msg + detail)


def detect_oom_in_logs(log_text: str) -> str:
    """Scan log text for known OOM signatures. Returns matched pattern or ''."""
    for pattern in WorkerOOMError.OOM_PATTERNS:
        if pattern in log_text:
            return pattern
    return ""


class WorkerCrashedError(OctopusError):
    """A Ray actor died unexpectedly."""


class ShardingError(OctopusError):
    """Model cannot be sharded with the requested strategy."""


class NoGPUsFoundError(OctopusError):
    """No CUDA GPUs discovered."""
