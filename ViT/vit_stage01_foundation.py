"""阶段 01：集中声明 ViT 数据路径、签名规则与原子写入工具。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# 读取 `Path(__file__).resolve().parent` 的当前值，并赋给 `VIT_DIR`。
VIT_DIR = Path(__file__).resolve().parent
# 读取 `VIT_DIR.parent` 的当前值，并赋给 `PROJECT_ROOT`。
PROJECT_ROOT = VIT_DIR.parent
# 计算 `PROJECT_ROOT / 'level2_pic'`，并将数值结果保存到 `LEVEL2_DIR`。
LEVEL2_DIR = PROJECT_ROOT / "level2_pic"
# 计算 `PROJECT_ROOT / 'Wind'`，并将数值结果保存到 `WIND_DIR`。
WIND_DIR = PROJECT_ROOT / "Wind"
# 计算 `WIND_DIR / 'ASHAREEODPRICES.parquet'`，并将数值结果保存到 `EOD_PATH`。
EOD_PATH = WIND_DIR / "ASHAREEODPRICES.parquet"
# 计算 `WIND_DIR / 'ASHAREEODDERIVATIVEINDICATOR.parquet'`，并将数值结果保存到 `DERIVATIVE_PATH`。
DERIVATIVE_PATH = WIND_DIR / "ASHAREEODDERIVATIVEINDICATOR.parquet"
# 计算 `WIND_DIR / 'ASHARECALENDAR.parquet'`，并将数值结果保存到 `CALENDAR_PATH`。
CALENDAR_PATH = WIND_DIR / "ASHARECALENDAR.parquet"
# 调用 `re.compile` 编译正则表达式供后续重复匹配，并将结果保存到 `IMAGE_PATTERN`。
IMAGE_PATTERN = re.compile(r"^stock_pixels_cn\.(\d{8})\.npz$")
# 构造去重集合，并保存到 `REQUIRED_IMAGE_KEYS`。
REQUIRED_IMAGE_KEYS = {"trade_date", "instrument_ids", "pixel_matrices", "valid_mask", "channel_names", "axis_order"}


# 定义 `get_data_root` 函数，封装这一阶段可独立调用的处理步骤。
def get_data_root() -> Path:
    """返回 ViT 数据根目录；测试可用 ``VIT_DATA_ROOT`` 覆盖。

    输入：
        无显式输入参数。
    输出：
        Path。
    """

    # 将 `Path(os.getenv('VIT_DATA_ROOT', str(VIT_DIR / 'data'))).expanduser()` 作为当前函数结果返回给调用方。
    return Path(os.getenv("VIT_DATA_ROOT", str(VIT_DIR / "data"))).expanduser()


# 定义 `read_bool_environment` 函数，封装这一阶段可独立调用的处理步骤。
def read_bool_environment(name: str, default: bool = False) -> bool:
    """读取布尔环境变量，未设置时返回 ``default``。

    输入：
        name: str。
        default: bool。
    输出：
        bool。
    """

    # 调用 `os.getenv` 读取环境变量中的运行配置，并将结果保存到 `value`。
    value = os.getenv(name)
    # 检查 `value is None`；根据条件结果选择对应处理分支。
    if value is None:
        # 将 `default` 作为当前函数结果返回给调用方。
        return default
    # 将 `value.strip().lower() in {'1', 'true', 'yes', 'y'}` 作为当前函数结果返回给调用方。
    return value.strip().lower() in {"1", "true", "yes", "y"}


# 定义 `normalize_dates` 函数，封装这一阶段可独立调用的处理步骤。
def normalize_dates(values: pd.Series) -> pd.Series:
    """把日期列统一转换为八位 ``YYYYMMDD`` 字符串。

    输入：
        values: pd.Series。
    输出：
        pd.Series。
    """

    # 将 `values.astype('string').str.replace('\\.0$', '', regex=True).str.zfill(8)` 作为当前函数结果返回给调用方。
    return values.astype("string").str.replace(
        r"\.0$", "", regex=True
    ).str.zfill(8)


# 定义 `stable_hash` 函数，封装这一阶段可独立调用的处理步骤。
def stable_hash(payload: object) -> str:
    """对可 JSON 序列化对象计算稳定 SHA-256。

    输入：
        payload: object。
    输出：
        str。
    """

    # 调用 `json.dumps` 把结构化对象序列化为 JSON 文本，并将结果保存到 `text`。
    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    # 将 `hashlib.sha256(text.encode('utf-8')).hexdigest()` 作为当前函数结果返回给调用方。
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# 定义 `file_signature` 函数，封装这一阶段可独立调用的处理步骤。
def file_signature(path: Path) -> dict[str, object]:
    """返回用于 lineage 的文件名、字节数和纳秒修改时间。

    输入：
        path: Path。
    输出：
        dict[str, object]。
    """

    # 调用 `path.stat` 解析当前 run 对应的数据与输出路径，并将结果保存到 `stat`。
    stat = path.stat()
    # 将 `{'name': path.name, 'size': stat.st_size, 'mtime_ns': stat.st_mtime_ns}` 作为当前函数结果返回给调用方。
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


# 定义 `read_json` 函数，封装这一阶段可独立调用的处理步骤。
def read_json(path: Path) -> dict[str, Any] | None:
    """读取 JSON；文件不存在时返回 ``None``。

    输入：
        path: Path。
    输出：
        dict[str, Any] | None。
    """

    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 将 `None` 作为当前函数结果返回给调用方。
        return None
    # 将 `json.loads(path.read_text(encoding='utf-8'))` 作为当前函数结果返回给调用方。
    return json.loads(path.read_text(encoding="utf-8"))


# 定义 `atomic_json` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_json(payload: dict[str, object], path: Path) -> None:
    """先写临时文件再原子替换 JSON 目标。

    输入：
        payload: dict[str, object]。
        path: Path。
    输出：
        None。
    """

    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    # 调用 `temporary.write_text` 将当前结果持久化到目标文件。
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)


# 定义 `atomic_parquet` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """以 Zstandard 压缩写入 Parquet，并原子替换目标。

    输入：
        frame: pd.DataFrame。
        path: Path。
    输出：
        None。
    """

    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    # 调用 `frame.to_parquet` 将当前结果持久化到目标文件。
    frame.to_parquet(temporary, index=False, compression="zstd")
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)


# 定义 `atomic_npz` 函数，封装这一阶段可独立调用的处理步骤。
def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """原子写入一组 NumPy 数组。

    输入：
        path: Path。
        arrays: np.ndarray。
    输出：
        None。
    """

    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    # 在 `temporary.open('wb')` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with temporary.open("wb") as file_object:
        # 调用 `np.savez` 把多个数组写入同一个 NPZ 容器。
        np.savez(file_object, **arrays)
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)
