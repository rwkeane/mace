###########################################################################################
# CUDA pass timing utilities
###########################################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class CudaPassTimer:
    """Measure GPU time for a training or validation pass using torch.cuda.Event."""

    device: torch.device
    enabled: bool
    elapsed_ms: Optional[float] = field(default=None, init=False)
    _start: Optional[torch.cuda.Event] = field(default=None, init=False, repr=False)
    _end: Optional[torch.cuda.Event] = field(default=None, init=False, repr=False)

    @property
    def active(self) -> bool:
        return self.enabled and self.device.type == "cuda"

    def __enter__(self) -> CudaPassTimer:
        if not self.active:
            return self
        torch.cuda.synchronize()
        self._start = torch.cuda.Event(enable_timing=True)
        self._end = torch.cuda.Event(enable_timing=True)
        self._start.record()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self.active:
            return
        self._end.record()
        self._end.synchronize()
        self.elapsed_ms = self._start.elapsed_time(self._end)


def reset_peak_memory(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mb(device: torch.device) -> Optional[float]:
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated(device) / (1024**2)
    return None
