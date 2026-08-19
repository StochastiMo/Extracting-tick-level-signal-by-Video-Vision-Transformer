"""Single-node distributed helpers shared by the ViViT training backends.

The helpers in this module keep the existing logical batch unchanged: every
rank receives a disjoint slice, and tensors are reconstructed before losses
whose definition depends on the complete batch or trading-day cross-section.
"""
from __future__ import annotations

import os
import random
import socket
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Sampler


VIVIT_DIR = Path(__file__).resolve().parent
VIT_DIR = VIVIT_DIR.parent / "ViT"
if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from vit_stage07_training import TradeDateBatchSampler  # noqa: E402


@dataclass(frozen=True)
class DistributedContext:
    """Process-local DDP identity."""

    rank: int
    world_size: int
    local_rank: int
    device: torch.device
    backend: str

    @property
    def is_primary(self) -> bool:
        return self.rank == 0


def parse_device_ids(value: str) -> tuple[int, ...]:
    """Parse a non-empty, duplicate-free CUDA device list."""

    try:
        devices = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as error:
        raise ValueError(f"非法 GPU 列表：{value!r}") from error
    if not devices or any(device < 0 for device in devices):
        raise ValueError("--device-ids 必须包含至少一个非负 GPU 编号")
    if len(set(devices)) != len(devices):
        raise ValueError("--device-ids 不能包含重复 GPU")
    return devices


def reserve_local_port() -> int:
    """Reserve an ephemeral loopback port for the next local process group."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])


def configure_master(port: int) -> None:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(int(port))


def initialize_process_group(
    rank: int,
    world_size: int,
    *,
    backend: str | None = None,
) -> DistributedContext:
    """Initialize one local rank and select its device."""

    selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")
    if selected_backend == "nccl":
        if not torch.cuda.is_available():
            raise RuntimeError("NCCL DDP 要求 CUDA")
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:
        device = torch.device("cpu")
    dist.init_process_group(
        backend=selected_backend,
        init_method="env://",
        rank=int(rank),
        world_size=int(world_size),
    )
    return DistributedContext(
        rank=int(rank),
        world_size=int(world_size),
        local_rank=int(rank),
        device=device,
        backend=selected_backend,
    )


def destroy_process_group() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def wrap_model(model: torch.nn.Module, context: DistributedContext) -> DistributedDataParallel:
    """Wrap a compiled or eager model without per-forward buffer broadcasts."""

    kwargs: dict[str, Any] = {"broadcast_buffers": False}
    if context.device.type == "cuda":
        kwargs.update(
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    return DistributedDataParallel(model, **kwargs)


class DistributedBatchSampler(Sampler[list[int]]):
    """Shard deterministic logical fixed-size batches without padding samples."""

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        *,
        rank: int,
        world_size: int,
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        if dataset_size < 1:
            raise ValueError("分布式 sampler 不能使用空数据集")
        if batch_size < world_size:
            raise ValueError("全局 batch size 不能小于 DDP world size")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.shuffle = bool(shuffle)
        self.drop_last = bool(drop_last)
        self.epoch = 0

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:
            order = list(range(self.dataset_size))
        usable = len(order)
        if self.drop_last:
            usable -= usable % self.batch_size
        for start in range(0, usable, self.batch_size):
            logical_batch = order[start:min(start + self.batch_size, usable)]
            if len(logical_batch) < self.world_size:
                raise ValueError(
                    "逻辑 batch 的样本数小于 DDP world size；请增大 batch 或减少 GPU"
                )
            yield logical_batch[self.rank :: self.world_size]


class DistributedTradeDateBatchSampler(TradeDateBatchSampler):
    """Give every rank a disjoint slice of every complete trading day."""

    def __init__(self, *args: Any, rank: int, world_size: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.rank = int(rank)
        self.world_size = int(world_size)
        too_small = np.flatnonzero(self.batch_sizes < self.world_size)
        if len(too_small):
            date_id = int(self.date_ids[int(too_small[0])])
            size = int(self.batch_sizes[int(too_small[0])])
            raise ValueError(
                f"date_id={date_id} 仅 {size} 个样本，小于 DDP world size={self.world_size}"
            )

    def __iter__(self) -> Iterator[range]:
        if self.shuffle:
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order: Sequence[int] = torch.randperm(
                len(self), generator=generator
            ).tolist()
        else:
            order = range(len(self))
        for batch_number in order:
            start = int(self.starts[batch_number])
            stop = int(self.stops[batch_number])
            yield range(start + self.rank, stop, self.world_size)


def _gather_lengths(tensor: torch.Tensor) -> list[int]:
    length = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)
    gathered = [torch.zeros_like(length) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered, length)
    return [int(item.item()) for item in gathered]


def _pad_first_dimension(tensor: torch.Tensor, target: int) -> torch.Tensor:
    if tensor.shape[0] == target:
        return tensor
    padding = tensor.new_zeros((target - tensor.shape[0], *tensor.shape[1:]))
    return torch.cat((tensor, padding), dim=0)


def differentiable_all_gather(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a variable first dimension while preserving autograd."""

    lengths = _gather_lengths(tensor)
    padded = _pad_first_dimension(tensor, max(lengths))
    gathered = dist_nn.all_gather(padded)
    return torch.cat(
        [part[:length] for part, length in zip(gathered, lengths)], dim=0
    )


@torch.no_grad()
def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Gather a variable first dimension for labels, metadata, or inference."""

    original_dtype = tensor.dtype
    communication_tensor = tensor
    if original_dtype == torch.int16:
        # NCCL does not implement collectives for Short metadata tensors.
        communication_tensor = tensor.to(torch.int32)
    lengths = _gather_lengths(communication_tensor)
    padded = _pad_first_dimension(communication_tensor, max(lengths))
    gathered = [torch.empty_like(padded) for _ in lengths]
    dist.all_gather(gathered, padded)
    result = torch.cat(
        [part[:length] for part, length in zip(gathered, lengths)], dim=0
    )
    return result.to(original_dtype)


def broadcast_object(value: Any, *, source: int = 0) -> Any:
    payload = [value]
    dist.broadcast_object_list(payload, src=source)
    return payload[0]


def all_gather_objects(value: Any) -> list[Any]:
    values: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, value)
    return values


def capture_rng_state(device: torch.device) -> dict[str, Any]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
    }


def gather_rng_states(device: torch.device) -> list[dict[str, Any]]:
    return all_gather_objects(capture_rng_state(device))


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
