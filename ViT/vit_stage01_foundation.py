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

VIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIT_DIR.parent
LEVEL2_DIR = PROJECT_ROOT / "level2_pic"
WIND_DIR = PROJECT_ROOT / "Wind"
EOD_PATH = WIND_DIR / "ASHAREEODPRICES.parquet"
DERIVATIVE_PATH = WIND_DIR / "ASHAREEODDERIVATIVEINDICATOR.parquet"
CALENDAR_PATH = WIND_DIR / "ASHARECALENDAR.parquet"
IMAGE_PATTERN = re.compile(r"^stock_pixels_cn\.(\d{8})\.npz$")
# 构造去重集合，并保存到 `REQUIRED_IMAGE_KEYS`。
REQUIRED_IMAGE_KEYS = {"trade_date", "instrument_ids", "pixel_matrices", "valid_mask", "channel_names", "axis_order"}


def get_data_root() -> Path:
    """返回 ViT 数据根目录；测试可用 ``VIT_DATA_ROOT`` 覆盖。"""

    return Path(os.getenv("VIT_DATA_ROOT", str(VIT_DIR / "data"))).expanduser()


def read_bool_environment(name: str, default: bool = False) -> bool:
    """读取布尔环境变量，未设置时返回 ``default``。"""

    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y"}


def normalize_dates(values: pd.Series) -> pd.Series:
    """把日期列统一转换为八位 ``YYYYMMDD`` 字符串。"""

    return values.astype("string").str.replace(
        r"\.0$", "", regex=True
    ).str.zfill(8)


def stable_hash(payload: object) -> str:
    """对可 JSON 序列化对象计算稳定 SHA-256。"""

    text = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_signature(path: Path) -> dict[str, object]:
    """返回用于 lineage 的文件名、字节数和纳秒修改时间。"""

    stat = path.stat()
    return {
        "name": path.name,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def read_json(path: Path) -> dict[str, Any] | None:
    """读取 JSON；文件不存在时返回 ``None``。"""

    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(payload: dict[str, object], path: Path) -> None:
    """先写临时文件再原子替换 JSON 目标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    """以 Zstandard 压缩写入 Parquet，并原子替换目标。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    """原子写入一组 NumPy 数组。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(
        f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    with temporary.open("wb") as file_object:
        np.savez(file_object, **arrays)
    temporary.replace(path)
