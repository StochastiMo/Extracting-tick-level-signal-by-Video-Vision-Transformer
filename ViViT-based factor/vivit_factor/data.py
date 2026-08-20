"""ViViT 基线因子的输入校验与数据准备。"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Sampler


IMAGE_PATTERN = re.compile(r"^stock_pixels_cn\.(\d{8})\.npz$")
# 每日图片必须同时提供这些键；下游依赖固定的股票顺序和有效性掩码。
IMAGE_KEYS = {
    "trade_date", "instrument_ids", "pixel_matrices", "valid_mask",
    "channel_names", "axis_order",
}
FORECAST_YEARS = tuple(range(2018, 2026))
SEEDS = (42, 43, 44)
SEQUENCE_LENGTH = 20


@dataclass(frozen=True)
class PipelinePaths:
    """集中管理流水线使用的输入路径和产物路径。"""

    project_dir: Path
    level2_dir: Path
    wind_dir: Path
    artifacts_dir: Path

    @classmethod
    def default(cls) -> "PipelinePaths":
        # 交付目录与 level2_pic、Wind 平行，避免依赖运行命令所在位置。
        project_dir = Path(__file__).resolve().parents[1]
        input_root = project_dir.parent
        wind_dir = input_root / "wind"
        # Linux 区分大小写；同时兼容实际交付目录 Wind/ 和约定写法 wind/。
        if not wind_dir.exists() and (input_root / "Wind").exists():
            wind_dir = input_root / "Wind"
        return cls(
            project_dir=project_dir,
            level2_dir=input_root / "level2_pic",
            wind_dir=wind_dir,
            artifacts_dir=project_dir / "artifacts",
        )

    @property
    def eod_path(self) -> Path:
        return self.wind_dir / "ASHAREEODPRICES.parquet"

    @property
    def derivative_path(self) -> Path:
        return self.wind_dir / "ASHAREEODDERIVATIVEINDICATOR.parquet"

    @property
    def calendar_path(self) -> Path:
        return self.wind_dir / "ASHARECALENDAR.parquet"

    @property
    def data_dir(self) -> Path:
        return self.artifacts_dir / "data"

    @property
    def universe_dir(self) -> Path:
        return self.data_dir / "universe"

    @property
    def labels_dir(self) -> Path:
        return self.data_dir / "labels"

    @property
    def feature_dir(self) -> Path:
        return self.data_dir / "feature_store"

    @property
    def split_dir(self) -> Path:
        return self.feature_dir / "rolling_splits"

    @property
    def features_path(self) -> Path:
        return self.feature_dir / "features.npy"

    @property
    def targets_path(self) -> Path:
        return self.feature_dir / "targets.npy"

    @property
    def label_valid_path(self) -> Path:
        return self.feature_dir / "label_valid.npy"

    @property
    def date_ids_path(self) -> Path:
        return self.feature_dir / "date_ids.npy"

    @property
    def history_path(self) -> Path:
        return self.feature_dir / "history_indices.npy"

    @property
    def sample_index_path(self) -> Path:
        return self.feature_dir / "sample_index.parquet"

    @property
    def factors_dir(self) -> Path:
        return self.artifacts_dir / "factors"

    @property
    def models_dir(self) -> Path:
        return self.artifacts_dir / "models"

    @property
    def logs_dir(self) -> Path:
        return self.artifacts_dir / "logs"


def stable_hash(payload: object) -> str:
    # 固定 JSON 字段顺序，使相同配置跨进程得到相同签名。
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {"name": path.name, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def atomic_json(payload: dict[str, object], path: Path) -> None:
    # 先写同目录临时文件，再原子替换，防止中断后留下半成品。
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def atomic_parquet(frame: pd.DataFrame, path: Path) -> None:
    # Parquet 同样采用原子替换，训练进程不会读到未写完的文件。
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    frame.to_parquet(temporary, index=False, compression="zstd")
    temporary.replace(path)


def atomic_numpy(array: np.ndarray, path: Path) -> None:
    # 显式关闭 pickle，保证数组产物只包含数值数据。
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    # 滚动切分体积较小，使用压缩 NPZ 一次性保存全部索引。
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def normalize_dates(values: pd.Series) -> pd.Series:
    # Wind 日期可能被读取成浮点字符串，统一还原为 YYYYMMDD。
    return values.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(8)


def list_image_paths(paths: PipelinePaths) -> dict[str, Path]:
    # 文件名中的日期是图片与行情数据对齐的唯一日期键。
    result: dict[str, Path] = {}
    for path in paths.level2_dir.glob("stock_pixels_cn.*.npz"):
        match = IMAGE_PATTERN.match(path.name)
        if match:
            trade_date = match.group(1)
            if trade_date in result:
                raise ValueError(f"duplicate Level2 image date: {trade_date}")
            result[trade_date] = path
    if not result:
        raise FileNotFoundError(f"no stock_pixels_cn.YYYYMMDD.npz under {paths.level2_dir}")
    return dict(sorted(result.items()))


def validate_inputs(paths: PipelinePaths) -> dict[str, object]:
    # 先校验三个必需的 Wind Parquet，失败时在耗时处理前立即退出。
    required = (paths.eod_path, paths.derivative_path, paths.calendar_path)
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"missing Wind parquet files: {missing}")
    image_paths = list_image_paths(paths)
    first_path = next(iter(image_paths.values()))
    with np.load(first_path, allow_pickle=False) as archive:
        # 抽查首个文件的数据契约；逐日读取时仍会复核日期和股票索引。
        missing_keys = IMAGE_KEYS.difference(archive.files)
        if missing_keys:
            raise KeyError(f"{first_path.name} missing keys: {sorted(missing_keys)}")
        shape = archive["pixel_matrices"].shape
        if shape[1:] != (17, 8, 8):
            raise ValueError(f"unexpected Level2 shape: {shape}")
    return {
        "image_count": len(image_paths),
        "first_image_date": min(image_paths),
        "last_image_date": max(image_paths),
        "wind_dir": str(paths.wind_dir),
    }


def load_calendars(paths: PipelinePaths) -> dict[str, np.ndarray]:
    # 沪深交易日分别推进，不能用自然日或统一日历替代 T+1。
    frame = pd.read_parquet(paths.calendar_path, columns=["trade_days", "s_info_exchmarket"])
    frame["trade_days"] = normalize_dates(frame["trade_days"])
    calendars: dict[str, np.ndarray] = {}
    for exchange, market in {"SH": "SSE", "SZ": "SZSE"}.items():
        dates = (
            frame.loc[frame["s_info_exchmarket"].astype(str).eq(market), "trade_days"]
            .drop_duplicates().sort_values().to_numpy(dtype=str)
        )
        if not len(dates):
            raise ValueError(f"calendar missing {market}")
        calendars[exchange] = dates
    return calendars


def calendar_shift(trade_date: str, exchange: str, calendars: dict[str, np.ndarray], steps: int) -> str:
    # searchsorted 同时验证 T 本身在对应交易所日历中。
    dates = calendars[exchange]
    position = int(np.searchsorted(dates, trade_date))
    if position >= len(dates) or dates[position] != trade_date or position + steps >= len(dates):
        raise ValueError(f"cannot shift {trade_date} by {steps} on {exchange}")
    return str(dates[position + steps])


def instrument_to_wind(instrument_id: str) -> str:
    # 图片代码为 SH.600000，Wind 行情键为 600000.SH。
    exchange, security_id = str(instrument_id).split(".", maxsplit=1)
    if exchange not in {"SH", "SZ"}:
        raise ValueError(f"unsupported instrument: {instrument_id}")
    return f"{security_id}.{exchange}"


def read_image_metadata(trade_date: str, path: Path) -> tuple[np.ndarray, np.ndarray]:
    # 此处只读元数据，不加载整批像素，降低构建 Universe 时的内存占用。
    with np.load(path, allow_pickle=False) as archive:
        missing = IMAGE_KEYS.difference(archive.files)
        if missing:
            raise KeyError(f"{path.name} missing keys: {sorted(missing)}")
        instruments = archive["instrument_ids"].astype(str)
        valid = archive["valid_mask"].astype(bool)
        if str(archive["trade_date"].item()) != trade_date:
            raise ValueError(f"date mismatch in {path.name}")
        if archive["pixel_matrices"].shape != (len(instruments), 17, 8, 8):
            raise ValueError(f"invalid pixel shape in {path.name}")
        if valid.shape != (len(instruments),) or len(np.unique(instruments)) != len(instruments):
            raise ValueError(f"invalid stock index contract in {path.name}")
    return instruments, valid


def classify_universe(candidates: pd.DataFrame) -> pd.DataFrame:
    """按 T/T+1 可交易性和 T 日图片规则标记样本，但保留全部候选行。"""
    output = candidates.copy()
    # 缺失状态一律视为不可用；这里没有额外过滤 T 日涨跌停。
    output["is_eligible"] = (
        output["feature_valid"].fillna(False).astype(bool)
        & output["tradestatus_t"].notna()
        & ~output["tradestatus_t"].astype("string").eq("停牌")
        & output["tradestatus_t1"].notna()
        & ~output["tradestatus_t1"].astype("string").eq("停牌")
        & output["limit_status_t1"].notna()
        & ~output["limit_status_t1"].isin([-1, 1])
    )
    return output


def read_market_status(
    paths: PipelinePaths, required_dates: set[str], required_codes: set[str]
) -> pd.DataFrame:
    # 仅按本年度候选股票和 T/T+1 日期下推读取，避免扫描完整行情表。
    filters = [("trade_dt", ">=", min(required_dates)), ("trade_dt", "<=", max(required_dates))]
    eod = pd.read_parquet(
        paths.eod_path,
        columns=["s_info_windcode", "trade_dt", "s_dq_tradestatus"],
        filters=filters,
    )
    derivative = pd.read_parquet(
        paths.derivative_path,
        columns=["s_info_windcode", "trade_dt", "up_down_limit_status"],
        filters=filters,
    )
    frames = []
    for frame in (eod, derivative):
        # 两张表先各自去重校验，再以股票和日期做一对一合并。
        frame["trade_dt"] = normalize_dates(frame["trade_dt"])
        frame["s_info_windcode"] = frame["s_info_windcode"].astype("string")
        frame = frame.loc[
            frame["trade_dt"].isin(required_dates)
            & frame["s_info_windcode"].isin(required_codes)
        ].copy()
        if frame.duplicated(["s_info_windcode", "trade_dt"]).any():
            raise ValueError("duplicate Wind stock/date status key")
        frames.append(frame)
    return frames[0].merge(
        frames[1], on=["s_info_windcode", "trade_dt"], how="outer", validate="one_to_one"
    )


def build_universe(paths: PipelinePaths, overwrite: bool = False) -> dict[str, object]:
    # 输入文件签名和筛选口径共同决定缓存是否仍然可复用。
    validate_inputs(paths)
    image_paths = list_image_paths(paths)
    calendars = load_calendars(paths)
    years = sorted({date[:4] for date in image_paths})
    output_paths = {year: paths.universe_dir / f"universe.{year}.parquet" for year in years}
    manifest_path = paths.universe_dir / "manifest.json"
    payload = {
        "schema_version": 1,
        "image_hash": stable_hash([file_signature(path) for path in image_paths.values()]),
        "eod": file_signature(paths.eod_path),
        "derivative": file_signature(paths.derivative_path),
        "calendar": file_signature(paths.calendar_path),
        "rule": "valid T image; T and T+1 not suspended; T+1 not limit up/down",
    }
    signature = stable_hash(payload)
    if not overwrite and manifest_path.exists() and all(path.exists() for path in output_paths.values()):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature:
            return existing

    total_rows = eligible_rows = 0
    for year in years:
        daily_frames = []
        for trade_date in (date for date in image_paths if date.startswith(year)):
            # 图片中的 stock_index 必须原样保留，后续据此从像素矩阵取行。
            instruments, feature_valid = read_image_metadata(trade_date, image_paths[trade_date])
            exchanges = np.char.partition(instruments, ".")[:, 0]
            t1_dates = np.full(len(instruments), "", dtype="<U8")
            for exchange in ("SH", "SZ"):
                mask = exchanges == exchange
                if mask.any():
                    # 每只股票按所属交易所推进一个交易日，不硬编码具体日期。
                    t1_dates[mask] = calendar_shift(trade_date, exchange, calendars, 1)
            if (t1_dates == "").any():
                raise ValueError(f"unsupported exchange on {trade_date}")
            daily_frames.append(pd.DataFrame({
                "wind_code": [instrument_to_wind(value) for value in instruments],
                "trade_date": trade_date,
                "t1_trade_date": t1_dates,
                "instrument_id": instruments,
                "stock_index": np.arange(len(instruments), dtype=np.int32),
                "feature_valid": feature_valid,
            }))
        candidates = pd.concat(daily_frames, ignore_index=True)
        # 一次读取本年所有 T 和 T+1 状态，再拆成两份回连候选表。
        required_dates = set(candidates["trade_date"]) | set(candidates["t1_trade_date"])
        status = read_market_status(paths, required_dates, set(candidates["wind_code"]))
        t_status = status[["s_info_windcode", "trade_dt", "s_dq_tradestatus"]].rename(columns={
            "s_info_windcode": "wind_code", "trade_dt": "trade_date",
            "s_dq_tradestatus": "tradestatus_t",
        })
        t1_status = status.rename(columns={
            "s_info_windcode": "wind_code", "trade_dt": "t1_trade_date",
            "s_dq_tradestatus": "tradestatus_t1",
            "up_down_limit_status": "limit_status_t1",
        })
        candidates = candidates.merge(
            t_status, on=["wind_code", "trade_date"], how="left", validate="many_to_one"
        ).merge(
            t1_status[["wind_code", "t1_trade_date", "tradestatus_t1", "limit_status_t1"]],
            on=["wind_code", "t1_trade_date"], how="left", validate="many_to_one",
        )
        # 保留未入选行便于审计，只通过 is_eligible 控制后续样本。
        candidates = classify_universe(candidates).sort_values(
            ["trade_date", "stock_index"], kind="stable"
        )
        if candidates.duplicated(["trade_date", "instrument_id"]).any():
            raise ValueError(f"duplicate universe key in {year}")
        atomic_parquet(candidates, output_paths[year])
        total_rows += len(candidates)
        eligible_rows += int(candidates["is_eligible"].sum())
        print(f"universe {year}: {len(candidates):,} rows, {int(candidates['is_eligible'].sum()):,} eligible")

    result = {
        "signature": signature, "payload": payload, "years": years,
        "candidate_rows": total_rows, "eligible_rows": eligible_rows,
    }
    atomic_json(result, manifest_path)
    return result


def calculate_forward_vwap_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """按股票直接位移，计算 T+1 到 T+11 的复权 VWAP 收益率。"""
    required = {"s_info_windcode", "trade_dt", "s_dq_avgprice", "s_dq_adjfactor"}
    missing = required.difference(prices.columns)
    if missing:
        raise KeyError(f"price data missing columns: {sorted(missing)}")
    frame = prices[list(required)].copy()
    frame["trade_dt"] = normalize_dates(frame["trade_dt"])
    frame["s_info_windcode"] = frame["s_info_windcode"].astype("string")
    frame = frame.sort_values(["s_info_windcode", "trade_dt"], kind="stable")
    if frame.duplicated(["s_info_windcode", "trade_dt"]).any():
        raise ValueError("duplicate Wind price key")
    # 复权 VWAP 用均价乘复权因子；非正数和无穷值不能参与收益率。
    adjusted_vwap = (
        pd.to_numeric(frame["s_dq_avgprice"], errors="coerce")
        * pd.to_numeric(frame["s_dq_adjfactor"], errors="coerce")
    )
    adjusted_vwap = adjusted_vwap.where(np.isfinite(adjusted_vwap) & adjusted_vwap.gt(0))
    grouped_price = adjusted_vwap.groupby(frame["s_info_windcode"], sort=False)
    grouped_date = frame["trade_dt"].groupby(frame["s_info_windcode"], sort=False)
    # 用户口径是不显式对齐日期，直接取每只股票未来第 1 和第 11 条交易记录。
    entry = grouped_price.shift(-1)
    exit_ = grouped_price.shift(-11)
    output = pd.DataFrame({
        "wind_code": frame["s_info_windcode"].to_numpy(),
        "trade_date": frame["trade_dt"].to_numpy(),
        "entry_adjusted_vwap": entry.to_numpy(dtype=np.float64),
        "exit_adjusted_vwap": exit_.to_numpy(dtype=np.float64),
        "label_end_date": grouped_date.shift(-11).to_numpy(),
    })
    output["forward_vwap_return_10d"] = (
        output["exit_adjusted_vwap"] / output["entry_adjusted_vwap"] - 1.0
    )
    return output


def mad_winsorize_zscore(values: pd.Series) -> pd.DataFrame:
    """先按中位数 ± 3×1.4826×MAD 去极值，再做 ddof=0 标准化。"""
    raw = pd.to_numeric(values, errors="coerce").to_numpy(dtype=np.float64)
    valid = np.isfinite(raw)
    clipped = np.full(len(raw), np.nan, dtype=np.float64)
    zscore = np.full(len(raw), np.nan, dtype=np.float64)
    lower = upper = np.nan
    # 少于两个有效样本时无法形成可靠横截面，保持结果为缺失。
    if valid.sum() >= 2:
        selected = raw[valid]
        median = float(np.median(selected))
        scaled_mad = 1.4826 * float(np.median(np.abs(selected - median)))
        if np.isfinite(scaled_mad) and scaled_mad > 0:
            # MAD 为零时不裁剪，随后标准差检查会阻止无效 z-score。
            lower, upper = median - 3.0 * scaled_mad, median + 3.0 * scaled_mad
            selected = np.clip(selected, lower, upper)
        clipped[valid] = selected
        standard_deviation = float(selected.std(ddof=0))
        if np.isfinite(standard_deviation) and standard_deviation > 0:
            zscore[valid] = (selected - selected.mean()) / standard_deviation
    return pd.DataFrame({
        "winsorized_forward_return": clipped,
        "target_zscore": zscore,
        "mad_lower": lower,
        "mad_upper": upper,
    }, index=values.index)


def _price_read_end(last_date: str, calendars: dict[str, np.ndarray]) -> str:
    # 多读 15 个交易日，为年末样本留出 T+11 标签窗口。
    ends = []
    for dates in calendars.values():
        position = int(np.searchsorted(dates, last_date, side="left"))
        ends.append(str(dates[min(position + 15, len(dates) - 1)]))
    return max(ends)


def build_labels(paths: PipelinePaths, overwrite: bool = False) -> dict[str, object]:
    universe_paths = sorted(paths.universe_dir.glob("universe.*.parquet"))
    if not universe_paths:
        raise FileNotFoundError("run build_universe first")
    paths.labels_dir.mkdir(parents=True, exist_ok=True)
    calendars = load_calendars(paths)
    years = [path.stem.split(".")[-1] for path in universe_paths]
    outputs = {year: paths.labels_dir / f"labels.{year}.parquet" for year in years}
    manifest_path = paths.labels_dir / "manifest.json"
    universe_manifest = json.loads((paths.universe_dir / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "schema_version": 1,
        "universe_signature": universe_manifest["signature"],
        "eod": file_signature(paths.eod_path),
        "return": "adjusted_vwap.shift(-11) / adjusted_vwap.shift(-1) - 1",
        "winsorization": "median +/- 3 * 1.4826 * MAD",
        "zscore_ddof": 0,
    }
    signature = stable_hash(payload)
    if not overwrite and manifest_path.exists() and all(path.exists() for path in outputs.values()):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature:
            return existing

    total = 0
    for universe_path, year in zip(universe_paths, years):
        # 只有 Universe 入选样本生成标签，未入选行仍保留在 Universe 产物中。
        universe = pd.read_parquet(universe_path)
        candidates = universe.loc[universe["is_eligible"]].copy()
        if candidates.empty:
            atomic_parquet(candidates, outputs[year])
            continue
        start_date = str(candidates["trade_date"].min())
        end_date = _price_read_end(str(candidates["trade_date"].max()), calendars)
        prices = pd.read_parquet(
            paths.eod_path,
            columns=["s_info_windcode", "trade_dt", "s_dq_avgprice", "s_dq_adjfactor"],
            filters=[("trade_dt", ">=", start_date), ("trade_dt", "<=", end_date)],
        )
        returns = calculate_forward_vwap_returns(prices)
        # 以股票和 T 日一对一回连，避免依赖 T+1/T+11 的外部日期映射。
        labels = candidates[["trade_date", "instrument_id", "wind_code", "stock_index"]].merge(
            returns, on=["wind_code", "trade_date"], how="left", validate="one_to_one"
        )
        normalized = []
        for _, indices in labels.groupby("trade_date", sort=True).groups.items():
            # MAD 和 z-score 都按每日横截面独立计算，杜绝跨日信息混用。
            normalized.append(mad_winsorize_zscore(labels.loc[indices, "forward_vwap_return_10d"]))
        normalized_frame = pd.concat(normalized).sort_index()
        labels[normalized_frame.columns] = normalized_frame
        labels = labels.loc[np.isfinite(labels["target_zscore"])].copy()
        float_columns = [
            "entry_adjusted_vwap", "exit_adjusted_vwap", "forward_vwap_return_10d",
            "winsorized_forward_return", "target_zscore", "mad_lower", "mad_upper",
        ]
        labels[float_columns] = labels[float_columns].astype(np.float32)
        if labels.duplicated(["trade_date", "instrument_id"]).any():
            raise ValueError(f"duplicate label key in {year}")
        atomic_parquet(labels, outputs[year])
        total += len(labels)
        print(f"labels {year}: {len(labels):,} valid")

    result = {"signature": signature, "payload": payload, "years": years, "label_rows": total}
    atomic_json(result, manifest_path)
    return result


def normalize_pixels(pixel_batch: np.ndarray) -> np.ndarray:
    # 每只股票、每个通道在 8×8 平面内独立做 min-max；常数通道置零。
    values = pixel_batch.astype(np.float32, copy=False)
    minimum = values.min(axis=(-2, -1), keepdims=True)
    value_range = values.max(axis=(-2, -1), keepdims=True) - minimum
    normalized = values - minimum
    np.divide(normalized, value_range, out=normalized, where=value_range > 0)
    return np.ascontiguousarray(np.where(value_range > 0, normalized, 0.0), dtype=np.float32)


def build_feature_store(paths: PipelinePaths, overwrite: bool = False) -> dict[str, object]:
    universe_paths = sorted(paths.universe_dir.glob("universe.*.parquet"))
    if not universe_paths or not (paths.labels_dir / "manifest.json").exists():
        raise FileNotFoundError("run universe and labels first")
    frames = []
    for universe_path in universe_paths:
        year = universe_path.stem.split(".")[-1]
        universe = pd.read_parquet(universe_path)
        selected = universe.loc[universe["is_eligible"]].copy()
        label_path = paths.labels_dir / f"labels.{year}.parquet"
        labels = pd.read_parquet(label_path, columns=[
            "trade_date", "instrument_id", "stock_index", "label_end_date", "target_zscore",
        ])
        # 左连接保留可预测但暂时无标签的样本，供对应 OOS 年生成因子。
        selected = selected.merge(
            labels, on=["trade_date", "instrument_id", "stock_index"],
            how="left", validate="one_to_one",
        )
        selected["label_valid"] = np.isfinite(selected["target_zscore"])
        frames.append(selected)
    samples = pd.concat(frames, ignore_index=True).sort_values(
        ["trade_date", "stock_index"], kind="stable"
    ).reset_index(drop=True)
    if samples.empty or samples.duplicated(["trade_date", "instrument_id"]).any():
        raise ValueError("invalid feature-store sample keys")
    feature_dates = np.sort(samples["trade_date"].astype(str).unique())
    # date_id 是连续交易日位置，历史窗口据此判断是否真正连续。
    samples["date_id"] = samples["trade_date"].map(
        {date: index for index, date in enumerate(feature_dates)}
    ).astype(np.int32)
    universe_manifest = json.loads((paths.universe_dir / "manifest.json").read_text(encoding="utf-8"))
    label_manifest = json.loads((paths.labels_dir / "manifest.json").read_text(encoding="utf-8"))
    manifest_path = paths.feature_dir / "manifest.json"
    payload = {
        "schema_version": 1,
        "universe_signature": universe_manifest["signature"],
        "label_signature": label_manifest["signature"],
        "sample_rule": "is_eligible",
        "normalization": "per-stock per-channel 8x8 min-max",
        "sample_count": len(samples),
    }
    signature = stable_hash(payload)
    required = (
        paths.features_path, paths.targets_path, paths.label_valid_path,
        paths.date_ids_path, paths.sample_index_path,
    )
    if not overwrite and manifest_path.exists() and all(path.exists() for path in required):
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("signature") == signature:
            return existing

    paths.feature_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths = {
        path: path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
        for path in required[:4]
    }
    arrays = {
        # 使用 .npy memmap 顺序写入，避免把全量图片一次装入内存。
        paths.features_path: np.lib.format.open_memmap(
            temporary_paths[paths.features_path], "w+", np.float32,
            shape=(len(samples), 17, 8, 8),
        ),
        paths.targets_path: np.lib.format.open_memmap(
            temporary_paths[paths.targets_path], "w+", np.float32, shape=(len(samples),),
        ),
        paths.label_valid_path: np.lib.format.open_memmap(
            temporary_paths[paths.label_valid_path], "w+", np.bool_, shape=(len(samples),),
        ),
        paths.date_ids_path: np.lib.format.open_memmap(
            temporary_paths[paths.date_ids_path], "w+", np.int32, shape=(len(samples),),
        ),
    }
    arrays[paths.targets_path][:] = np.nan
    cursor = 0
    index_frames = []
    for trade_date in feature_dates:
        daily = samples.loc[samples["trade_date"].eq(trade_date)].sort_values("stock_index", kind="stable")
        stock_indices = daily["stock_index"].to_numpy(dtype=np.int64)
        with np.load(paths.level2_dir / f"stock_pixels_cn.{trade_date}.npz", allow_pickle=False) as archive:
            # stock_index 和 instrument_id 双重校验，防止像素错配到其他股票。
            image_ids = archive["instrument_ids"][stock_indices].astype(str)
            if not np.array_equal(image_ids, daily["instrument_id"].to_numpy(dtype=str)):
                raise ValueError(f"image/universe row mismatch on {trade_date}")
            if not archive["valid_mask"][stock_indices].astype(bool).all():
                raise ValueError(f"invalid image entered feature store on {trade_date}")
            pixels = normalize_pixels(archive["pixel_matrices"][stock_indices])
        stop = cursor + len(daily)
        # 四个数组使用完全相同的样本顺序，sample_index 即其共同位置。
        arrays[paths.features_path][cursor:stop] = pixels
        arrays[paths.targets_path][cursor:stop] = daily["target_zscore"].to_numpy(np.float32)
        arrays[paths.label_valid_path][cursor:stop] = daily["label_valid"].to_numpy(bool)
        arrays[paths.date_ids_path][cursor:stop] = daily["date_id"].to_numpy(np.int32)
        index = daily[[
            "trade_date", "instrument_id", "wind_code", "stock_index", "date_id",
            "t1_trade_date", "label_end_date", "label_valid",
        ]].copy()
        index.insert(0, "sample_index", np.arange(cursor, stop, dtype=np.int64))
        index_frames.append(index)
        cursor = stop
    for array in arrays.values():
        # flush 后释放映射，再替换正式文件，保证其他进程只看到完整版本。
        array.flush()
    arrays.clear()
    for final_path, temporary_path in temporary_paths.items():
        temporary_path.replace(final_path)
    atomic_parquet(pd.concat(index_frames, ignore_index=True), paths.sample_index_path)
    result = {"signature": signature, "payload": payload, "sample_count": len(samples)}
    atomic_json(result, manifest_path)
    return result


def build_history_indices(sample_index: pd.DataFrame, output_path: Path) -> None:
    sample_numbers = sample_index["sample_index"].to_numpy(dtype=np.int64)
    if not np.array_equal(sample_numbers, np.arange(len(sample_index), dtype=np.int64)):
        raise ValueError("sample_index must be contiguous")
    history = np.full((len(sample_index), SEQUENCE_LENGTH), -1, dtype=np.int32)
    # 序列最后一格永远是 T 日自身；缺失历史保持 -1，不向前填充。
    history[:, -1] = sample_numbers.astype(np.int32)
    date_ids = sample_index["date_id"].to_numpy(dtype=np.int32)
    for positions in sample_index.groupby("instrument_id", sort=False).indices.values():
        rows = np.asarray(positions, dtype=np.int64)
        group_dates = date_ids[rows]
        group_samples = sample_numbers[rows]
        if len(group_dates) > 1 and np.any(np.diff(group_dates) <= 0):
            raise ValueError("instrument dates must be strictly increasing")
        for lag in range(1, SEQUENCE_LENGTH):
            # 只有 date_id 恰好等于 T-lag 才写入，停牌或缺图形成显式空槽。
            desired = group_dates - lag
            locations = np.searchsorted(group_dates, desired)
            in_bounds = locations < len(group_dates)
            matches = np.zeros(len(group_dates), dtype=bool)
            matches[in_bounds] = group_dates[locations[in_bounds]] == desired[in_bounds]
            history[group_samples[matches], SEQUENCE_LENGTH - 1 - lag] = (
                group_samples[locations[matches]].astype(np.int32)
            )
    atomic_numpy(history, output_path)


def build_rolling_splits(paths: PipelinePaths) -> pd.DataFrame:
    sample_index = pd.read_parquet(paths.sample_index_path)
    sample_index["trade_date"] = sample_index["trade_date"].astype(str)
    sample_index["label_end_date"] = sample_index["label_end_date"].astype("string")
    labeled = sample_index.loc[sample_index["label_valid"]].copy()
    date_max_end = labeled.groupby("trade_date", sort=True)["label_end_date"].max()
    records = []
    for forecast_year in FORECAST_YEARS:
        forecast_start = f"{forecast_year}0101"
        # 先剔除标签结束日越过预测年起点的日期，阻断未来收益泄漏。
        eligible_dates = date_max_end.loc[
            (date_max_end.index < forecast_start) & date_max_end.lt(forecast_start)
        ].index.to_numpy(dtype=str)
        if len(eligible_dates) < 2:
            raise ValueError(f"insufficient history for {forecast_year}")
        split_position = min(max(int(np.floor(len(eligible_dates) * 0.8)), 1), len(eligible_dates) - 1)
        raw_train_dates = eligible_dates[:split_position]
        validation_dates = eligible_dates[split_position:]
        validation_start = str(validation_dates[0])
        # 再剔除训练标签越过验证起点的日期，形成训练/验证 purge。
        train_dates = date_max_end.loc[raw_train_dates]
        train_dates = train_dates.loc[train_dates.lt(validation_start)].index.to_numpy(dtype=str)
        train = sample_index.loc[
            sample_index["label_valid"] & sample_index["trade_date"].isin(train_dates), "sample_index"
        ].to_numpy(np.int64)
        validation = sample_index.loc[
            sample_index["label_valid"] & sample_index["trade_date"].isin(validation_dates), "sample_index"
        ].to_numpy(np.int64)
        prediction = sample_index.loc[
            # 预测集取目标年份全部 Universe 样本，不要求标签有效。
            sample_index["trade_date"].str.startswith(str(forecast_year)), "sample_index"
        ].to_numpy(np.int64)
        if len(train) == 0 or len(validation) == 0:
            raise ValueError(f"empty train/validation split for {forecast_year}")
        atomic_npz(
            paths.split_dir / f"forecast_{forecast_year}.npz",
            train_indices=train, validation_indices=validation, prediction_indices=prediction,
        )
        records.append({
            "forecast_year": forecast_year,
            "train_samples": len(train), "validation_samples": len(validation),
            "prediction_samples": len(prediction),
            "max_train_label_end": str(sample_index.iloc[train]["label_end_date"].max()),
            "validation_start": validation_start,
            "max_validation_label_end": str(sample_index.iloc[validation]["label_end_date"].max()),
            "forecast_start": forecast_start,
        })
    summary = pd.DataFrame(records)
    if not (summary["max_train_label_end"] < summary["validation_start"]).all():
        raise AssertionError("train labels cross validation boundary")
    if not (summary["max_validation_label_end"] < summary["forecast_start"]).all():
        raise AssertionError("validation labels cross forecast boundary")
    atomic_parquet(summary, paths.feature_dir / "rolling_split_summary.parquet")
    return summary


def prepare_data(paths: PipelinePaths, overwrite: bool = False) -> dict[str, object]:
    # 依赖顺序固定：Universe → 标签 → 特征 → 历史索引 → 滚动切分。
    universe = build_universe(paths, overwrite)
    labels = build_labels(paths, overwrite)
    features = build_feature_store(paths, overwrite)
    sample_index = pd.read_parquet(paths.sample_index_path)
    build_history_indices(sample_index, paths.history_path)
    splits = build_rolling_splits(paths)
    return {
        "universe_rows": universe["eligible_rows"],
        "label_rows": labels["label_rows"],
        "sample_count": features["sample_count"],
        "forecast_years": splits["forecast_year"].tolist(),
    }


def load_split(paths: PipelinePaths, forecast_year: int) -> dict[str, np.ndarray]:
    split_path = paths.split_dir / f"forecast_{forecast_year}.npz"
    if not split_path.exists():
        raise FileNotFoundError(split_path)
    with np.load(split_path, allow_pickle=False) as archive:
        return {name: archive[name].copy() for name in archive.files}


class SampleIndexDataset(Dataset):
    def __init__(self, indices: np.ndarray) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> int:
        return int(self.indices[position])


class TradeDateBatchSampler(Sampler[list[int]]):
    """每个批次返回一个完整交易日横截面的数据集位置。"""

    def __init__(self, indices: np.ndarray, date_ids: np.ndarray, seed: int) -> None:
        self.indices = np.asarray(indices, dtype=np.int64)
        selected_dates = np.asarray(date_ids[self.indices], dtype=np.int64)
        if not len(self.indices) or np.any(selected_dates[1:] < selected_dates[:-1]):
            raise ValueError("training indices must be non-empty and date-sorted")
        boundaries = np.flatnonzero(selected_dates[1:] != selected_dates[:-1]) + 1
        # starts/stops 是 Dataset 的局部位置，不是全局 sample_index。
        self.starts = np.r_[0, boundaries]
        self.stops = np.r_[boundaries, len(self.indices)]
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __iter__(self) -> Iterator[list[int]]:
        # 每轮只打乱交易日顺序，日内股票仍保持固定顺序并完整入批。
        order = np.random.default_rng(self.seed + self.epoch).permutation(len(self.starts))
        for position in order:
            yield np.arange(self.starts[position], self.stops[position]).tolist()

    def __len__(self) -> int:
        return len(self.starts)


class VideoBatchCollator:
    def __init__(self, paths: PipelinePaths) -> None:
        self.paths = paths
        self.features = self.targets = self.date_ids = self.history = None

    def _open(self) -> None:
        if self.features is None:
            # DataLoader worker 首次使用时再各自打开只读 memmap，避免序列化大数组。
            self.features = np.load(self.paths.features_path, mmap_mode="r")
            self.targets = np.load(self.paths.targets_path, mmap_mode="r")
            self.date_ids = np.load(self.paths.date_ids_path, mmap_mode="r")
            self.history = np.load(self.paths.history_path, mmap_mode="r")

    def __call__(self, sample_indices: list[int]) -> dict[str, torch.Tensor]:
        self._open()
        indices = np.asarray(sample_indices, dtype=np.int64)
        sequence = np.array(self.history[indices], dtype=np.int64, copy=True)
        frame_mask = sequence >= 0
        if not frame_mask[:, -1].all():
            raise AssertionError("target frame must be valid")
        # 缺失历史帧以零像素占位，frame_mask 交给模型屏蔽对应 tubelet。
        frames = np.zeros((len(indices), SEQUENCE_LENGTH, 17, 8, 8), dtype=np.float32)
        frames[frame_mask] = np.array(self.features[sequence[frame_mask]], dtype=np.float32, copy=True)
        videos = np.ascontiguousarray(frames.transpose(0, 2, 1, 3, 4))
        return {
            "videos": torch.from_numpy(videos),
            "frame_mask": torch.from_numpy(frame_mask),
            "targets": torch.from_numpy(np.array(self.targets[indices], dtype=np.float32, copy=True)),
            "date_ids": torch.from_numpy(np.array(self.date_ids[indices], dtype=np.int32, copy=True)),
            "sample_indices": torch.from_numpy(indices),
        }


def make_data_loaders(
    paths: PipelinePaths,
    forecast_year: int,
    seed: int,
    workers: int = 8,
    inference_batch_size: int = 5000,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader, DataLoader | None]:
    split = load_split(paths, forecast_year)
    common: dict[str, object] = {
        "num_workers": workers, "pin_memory": pin_memory,
        "collate_fn": VideoBatchCollator(paths),
    }
    if workers:
        # worker 常驻并仅预取一个批次，降低大横截面视频占用的主存峰值。
        common.update({"persistent_workers": True, "prefetch_factor": 1})
    date_ids = np.load(paths.date_ids_path, mmap_mode="r")
    train_sampler = TradeDateBatchSampler(split["train_indices"], date_ids, seed)
    # 训练按完整日期批处理；验证和预测按固定样本数顺序读取。
    generator = torch.Generator().manual_seed(seed)
    train = DataLoader(
        SampleIndexDataset(split["train_indices"]), batch_sampler=train_sampler,
        generator=generator, **common,
    )
    validation = DataLoader(
        SampleIndexDataset(split["validation_indices"]), batch_size=inference_batch_size,
        shuffle=False, drop_last=False, **common,
    )
    prediction = None
    if len(split["prediction_indices"]):
        prediction = DataLoader(
            SampleIndexDataset(split["prediction_indices"]), batch_size=inference_batch_size,
            shuffle=False, drop_last=False, **common,
        )
    return train, validation, prediction
