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


VIVIT_DIR = Path(__file__).resolve().parent  # 定位vivit目录。
VIT_DIR = VIVIT_DIR.parent / "ViT"  # 定位vit目录。
if str(VIT_DIR) not in sys.path:  # 检查当前值是否属于允许集合。
    sys.path.insert(0, str(VIT_DIR))

from vit_stage07_training import TradeDateBatchSampler  # noqa: E402。


@dataclass(frozen=True)  # 自动生成配置对象的初始化与比较方法。
class DistributedContext:  # 保存当前进程的 DDP 身份与设备信息。
    """Process-local DDP identity."""

    rank: int  # 接收排名。
    world_size: int  # 接收world大小。
    local_rank: int  # 接收本地排名。
    device: torch.device  # 接收设备。
    backend: str  # 接收backend。

    @property
    def is_primary(self) -> bool:  # 判断当前进程是否为主 rank。
        return self.rank == 0  # 返回isprimary。


def parse_device_ids(value: str) -> tuple[int, ...]:  # 解析非空且无重复的 CUDA 设备列表。
    """Parse a non-empty, duplicate-free CUDA device list."""

    try:  # 执行可能失败的外部读取或计算步骤。
        devices = tuple(int(item.strip()) for item in value.split(","))  # 初始化devices容器。
    except ValueError as error:  # 捕获预期异常并执行对应恢复逻辑。
        raise ValueError(f"非法 GPU 列表：{value!r}") from error  # 输入数据不满足业务或数值约束时立即终止。
    if not devices or any(device < 0 for device in devices):  # 检查当前值是否属于允许集合。
        raise ValueError("--device-ids 必须包含至少一个非负 GPU 编号")  # 输入数据不满足业务或数值约束时立即终止。
    if len(set(devices)) != len(devices):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("--device-ids 不能包含重复 GPU")  # 输入数据不满足业务或数值约束时立即终止。
    return devices  # 返回devices。


def reserve_local_port() -> int:  # 申请本机临时端口供进程组通信。
    """Reserve an ephemeral loopback port for the next local process group."""

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as handle:  # 进入受控上下文并在退出时自动释放资源。
        handle.bind(("127.0.0.1", 0))
        return int(handle.getsockname()[1])  # 返回Reserve an ephemeral loopback port for the next 。


def configure_master(port: int) -> None:  # 配置本机 DDP 主节点地址与端口。
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(int(port))


def initialize_process_group(  # 初始化当前 rank 的进程组与 CUDA 设备。
    rank: int,  # 接收排名。
    world_size: int,  # 接收world大小。
    *,
    backend: str | None = None,  # 接收backend。
) -> DistributedContext:
    """Initialize one local rank and select its device."""

    selected_backend = backend or ("nccl" if torch.cuda.is_available() else "gloo")  # 构造筛选结果backend，限定后续参与计算的样本。
    if selected_backend == "nccl":  # 按当前模式、标识或状态选择处理分支。
        if not torch.cuda.is_available():  # 检查必需数据、配置或校验结果是否缺失。
            raise RuntimeError("NCCL DDP 要求 CUDA")  # 当前运行环境或任务状态不支持继续执行。
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
    else:  # 处理上述条件均不成立的情况。
        device = torch.device("cpu")
    dist.init_process_group(
        backend=selected_backend,  # 配置backend。
        init_method="env://",  # 配置initmethod。
        rank=int(rank),  # 记录当前 DDP 进程编号。
        world_size=int(world_size),  # 配置world大小。
    )
    return DistributedContext(  # 返回Initialize one local rank and select its device.。
        rank=int(rank),  # 记录当前 DDP 进程编号。
        world_size=int(world_size),  # 配置world大小。
        local_rank=int(rank),  # 把本机进程编号与当前 rank 对齐。
        device=device,  # 指定张量所在的计算设备。
        backend=selected_backend,  # 配置backend。
    )


def destroy_process_group() -> None:  # 安全销毁已初始化的分布式进程组。
    if dist.is_available() and dist.is_initialized():  # 根据当前数据状态选择处理分支。
        dist.destroy_process_group()


def wrap_model(model: torch.nn.Module, context: DistributedContext) -> DistributedDataParallel:  # 把模型包装为无逐次 buffer 广播的 DDP 模型。
    """Wrap a compiled or eager model without per-forward buffer broadcasts."""

    kwargs: dict[str, Any] = {"broadcast_buffers": False}  # 初始化kwargs容器。
    if context.device.type == "cuda":  # 按当前模式、标识或状态选择处理分支。
        kwargs.update(  # 用当前配置更新参数字典。
            device_ids=[context.local_rank],  # 配置设备ids。
            output_device=context.local_rank,  # 配置输出设备。
        )
    return DistributedDataParallel(model, **kwargs)  # 返回Wrap a compiled or eager model without per-forwa。


class DistributedBatchSampler(Sampler[list[int]]):  # 把固定大小逻辑批次无填充地切分到各 rank。
    """Shard deterministic logical fixed-size batches without padding samples."""

    def __init__(  # 初始化当前对象的配置、状态与子组件。
        self,
        dataset_size: int,  # 接收dataset大小。
        batch_size: int,  # 接收批次大小。
        *,
        rank: int,  # 接收排名。
        world_size: int,  # 接收world大小。
        seed: int,
        shuffle: bool,
        drop_last: bool,
    ) -> None:
        if dataset_size < 1:  # 检查当前数值是否越过业务或数值阈值。
            raise ValueError("分布式 sampler 不能使用空数据集")  # 输入数据不满足业务或数值约束时立即终止。
        if batch_size < world_size:  # 检查当前数值是否越过业务或数值阈值。
            raise ValueError("全局 batch size 不能小于 DDP world size")  # 输入数据不满足业务或数值约束时立即终止。
        self.dataset_size = int(dataset_size)  # 保存dataset大小，供该对象后续步骤复用。
        self.batch_size = int(batch_size)  # 保存批次大小，供该对象后续步骤复用。
        self.rank = int(rank)  # 保存当前 DDP 进程编号。
        self.world_size = int(world_size)  # 保存world大小，供该对象后续步骤复用。
        self.seed = int(seed)  # 保存随机种子，供该对象后续步骤复用。
        self.shuffle = bool(shuffle)  # 保存shuffle，供该对象后续步骤复用。
        self.drop_last = bool(drop_last)  # 保存droplast，供该对象后续步骤复用。
        self.epoch = 0  # 保存轮次，供该对象后续步骤复用。

    def __len__(self) -> int:  # 返回采样器包含的逻辑批次数。
        if self.drop_last:  # 根据当前数据状态选择处理分支。
            return self.dataset_size // self.batch_size  # 返回len。
        return (self.dataset_size + self.batch_size - 1) // self.batch_size  # 返回len。

    def set_epoch(self, epoch: int) -> None:  # set轮次。
        self.epoch = int(epoch)  # 保存轮次，供该对象后续步骤复用。

    def __iter__(self) -> Iterator[list[int]]:  # 按当前采样与分片规则逐批迭代。
        if self.shuffle:  # 根据当前数据状态选择处理分支。
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(self.dataset_size, generator=generator).tolist()
        else:  # 处理上述条件均不成立的情况。
            order = list(range(self.dataset_size))  # 初始化排序索引容器。
        usable = len(order)
        if self.drop_last:  # 根据当前数据状态选择处理分支。
            usable -= usable % self.batch_size
        for start in range(0, usable, self.batch_size):  # 按给定步长遍历样本区间，更新起点。
            logical_batch = order[start:min(start + self.batch_size, usable)]
            if len(logical_batch) < self.world_size:  # 检查当前样本数量是否满足计算要求。
                raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                    "逻辑 batch 的样本数小于 DDP world size；请增大 batch 或减少 GPU"
                )
            yield logical_batch[self.rank :: self.world_size]  # 产出当前批次并保留迭代状态。


class DistributedTradeDateBatchSampler(TradeDateBatchSampler):  # 把每个完整交易日截面互斥切分到各 rank。
    """Give every rank a disjoint slice of every complete trading day."""

    def __init__(self, *args: Any, rank: int, world_size: int, **kwargs: Any) -> None:  # 初始化当前对象的配置、状态与子组件。
        super().__init__(*args, **kwargs)
        self.rank = int(rank)  # 保存当前 DDP 进程编号。
        self.world_size = int(world_size)  # 保存world大小，供该对象后续步骤复用。
        too_small = np.flatnonzero(self.batch_sizes < self.world_size)  # 计算toosmall判定条件。
        if len(too_small):  # 检查当前样本数量是否满足计算要求。
            date_id = int(self.date_ids[int(too_small[0])])
            size = int(self.batch_sizes[int(too_small[0])])
            raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
                f"date_id={date_id} 仅 {size} 个样本，小于 DDP world size={self.world_size}"
            )

    def __iter__(self) -> Iterator[range]:  # 按当前采样与分片规则逐批迭代。
        if self.shuffle:  # 根据当前数据状态选择处理分支。
            generator = torch.Generator().manual_seed(self.seed + self.epoch)
            order: Sequence[int] = torch.randperm(
                len(self), generator=generator
            ).tolist()
        else:  # 处理上述条件均不成立的情况。
            order = range(len(self))
        for batch_number in order:  # 遍历排序索引，逐项处理批次number。
            start = int(self.starts[batch_number])
            stop = int(self.stops[batch_number])
            yield range(start + self.rank, stop, self.world_size)  # 产出当前批次并保留迭代状态。


def _gather_lengths(tensor: torch.Tensor) -> list[int]:  # tensor [N_local,...]。
    length = torch.tensor([tensor.shape[0]], dtype=torch.int64, device=tensor.device)  # [1]。
    gathered = [torch.zeros_like(length) for _ in range(dist.get_world_size())]  # world_size 个 [1]。
    dist.all_gather(gathered, length)
    return [int(item.item()) for item in gathered]  # 返回汇聚lengths。


def _pad_first_dimension(tensor: torch.Tensor, target: int) -> torch.Tensor:  # [N_local,...] -> [N_max,...]。
    if tensor.shape[0] == target:  # 检查张量或数组形状是否符合接口约定。
        return tensor  # 返回张量。
    padding = tensor.new_zeros((target - tensor.shape[0], *tensor.shape[1:]))  # [N_max-N_local,...]。
    return torch.cat((tensor, padding), dim=0)  # [N_local,...]+[N_max-N_local,...] -> [N_max,...]。


def differentiable_all_gather(tensor: torch.Tensor) -> torch.Tensor:  # [N_local,...] -> [N_global,...]。
    """Gather a variable first dimension while preserving autograd."""

    lengths = _gather_lengths(tensor)
    padded = _pad_first_dimension(tensor, max(lengths))  # [N_local,...] -> [N_max,...]。
    gathered = dist_nn.all_gather(padded)
    return torch.cat(  # 拼接各分块并返回完整结果。
        [part[:length] for part, length in zip(gathered, lengths)], dim=0
    )


@torch.no_grad()  # 验证或推理阶段关闭梯度记录。
def all_gather_tensor(tensor: torch.Tensor) -> torch.Tensor:  # [N_local,...] -> [N_global,...]。
    """Gather a variable first dimension for labels, metadata, or inference."""

    original_dtype = tensor.dtype
    communication_tensor = tensor
    if original_dtype == torch.int16:  # 按当前模式、标识或状态选择处理分支。
  # NCCL does not implement collectives for Short metadata tensors.。
        communication_tensor = tensor.to(torch.int32)  # 转换communication张量的数据类型或存储位置。
    lengths = _gather_lengths(communication_tensor)
    padded = _pad_first_dimension(communication_tensor, max(lengths))  # [N_local,...] -> [N_max,...]。
    gathered = [torch.empty_like(padded) for _ in lengths]  # world_size 个 [N_max,...]。
    dist.all_gather(gathered, padded)
    result = torch.cat(  # 各 rank 有效部分拼接为 [N_global,...]。
        [part[:length] for part, length in zip(gathered, lengths)], dim=0
    )
    return result.to(original_dtype)  # 返回Gather a variable first dimension for labels, me。


def broadcast_object(value: Any, *, source: int = 0) -> Any:  # 从主 rank 广播普通 Python 对象。
    payload = [value]  # 初始化载荷容器。
    dist.broadcast_object_list(payload, src=source)
    return payload[0]  # 返回broadcastobject。


def all_gather_objects(value: Any) -> list[Any]:  # all汇聚objects。
    values: list[Any] = [None for _ in range(dist.get_world_size())]
    dist.all_gather_object(values, value)
    return values  # 返回值。


def capture_rng_state(device: torch.device) -> dict[str, Any]:  # capturerng状态。
    return {  # 返回capturerng状态。
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": (
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        ),
    }


def gather_rng_states(device: torch.device) -> list[dict[str, Any]]:  # 汇聚rngstates。
    return all_gather_objects(capture_rng_state(device))  # 返回汇聚rngstates。


def restore_rng_state(state: dict[str, Any], device: torch.device) -> None:  # restorerng状态。
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"])
    if device.type == "cuda" and state.get("cuda_rng_state") is not None:  # 按当前模式、标识或状态选择处理分支。
        torch.cuda.set_rng_state(state["cuda_rng_state"], device)
