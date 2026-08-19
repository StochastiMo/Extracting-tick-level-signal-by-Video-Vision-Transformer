"""阶段 02：读取 CNE5S、交易日历与复权价，并构造可复用的 Barra 原始特征。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

VIVIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIVIT_DIR.parent
DEFAULT_VIVIT_DATA_ROOT = VIVIT_DIR / "data"
DEFAULT_CNE5S_PATH = PROJECT_ROOT / "CNE5S.parquet"
DEFAULT_EOD_PATH = PROJECT_ROOT / "Wind" / "ASHAREEODPRICES.parquet"
DEFAULT_CALENDAR_PATH = PROJECT_ROOT / "Wind" / "ASHARECALENDAR.parquet"
DEFAULT_SOURCE_RUN_NAME = "sigmoid_daybatch_rankic"
DEFAULT_RESIDUAL_RUN_NAME = f"{DEFAULT_SOURCE_RUN_NAME}_barra_residual"
PREDICTION_PATTERN = "vivit_predictions.{year}.parquet"
SOURCE_FACTOR_COLUMN = "prediction_ensemble"
SOURCE_FACTOR_COLUMNS = (
    "prediction_ensemble",
    "prediction_ensemble_unprojected",
    "prediction_ensemble_projected",
    "prediction_ensemble_raw",
    "prediction_ensemble_alpha",
    "prediction_ensemble_style",
    "prediction_ensemble_total_return",
)
INDUSTRY_COLUMN = "cne5s_industry"
UNKNOWN_INDUSTRY = "CNE5S_UNKNOWN"
MINIMUM_MATCH_RATE = 0.999
DERIVATION_SCHEMA_VERSION = 5
HAC_MAX_LAG = 20
ORTHOGONALITY_TOLERANCE = 1e-8
REVERSAL_COLUMN = "reversal_20d"
REVERSAL_VALID_COLUMN = "reversal_20d_valid"
REVERSAL_LOOKBACK_DAYS = 20

STYLE_COLUMNS = (
    "cne5s_beta",
    "cne5s_btop",
    "cne5s_earnyild",
    "cne5s_growth",
    "cne5s_leverage",
    "cne5s_liquidty",
    "cne5s_momentum",
    "cne5s_size",
    "cne5s_sizenl",
)
STYLE_LABELS = {
    "cne5s_beta": "贝塔",
    "cne5s_btop": "账面市值比",
    "cne5s_earnyild": "盈利收益率",
    "cne5s_growth": "成长",
    "cne5s_leverage": "杠杆",
    "cne5s_liquidty": "流动性",
    "cne5s_momentum": "动量",
    "cne5s_size": "市值",
    "cne5s_sizenl": "非线性市值",
    REVERSAL_COLUMN: "20日反转",
}
CORRELATION_COLUMNS = (*STYLE_COLUMNS, REVERSAL_COLUMN)
REGRESSION_COLUMNS = CORRELATION_COLUMNS
VIVIT_MATRIX_COLUMN = "vivit_factor"
MATRIX_COLUMNS = (VIVIT_MATRIX_COLUMN, *CORRELATION_COLUMNS)
MATRIX_LABELS = {
    VIVIT_MATRIX_COLUMN: "ViViT因子",
    **STYLE_LABELS,
}
REQUIRED_CNE5S_COLUMNS = (
    "datadate",
    "assetid",
    INDUSTRY_COLUMN,
    *STYLE_COLUMNS,
)
INSTRUMENT_PATTERN = re.compile(r"^(SH|SZ|BJ)\.(\d{6})$")
CNE5S_ASSET_PATTERN = re.compile(r"^CN\d{6}$")


def _fingerprint_parquet(path: Path) -> dict[str, Any]:
    """提取文件身份信息，用于来源审计和复用判断。"""
    if not path.exists():
        raise FileNotFoundError(path)
    parquet = pq.ParquetFile(path)  # 读取 Parquet 元数据
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "mtime_ns": int(path.stat().st_mtime_ns),
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.num_row_groups),
        "columns": list(parquet.schema_arrow.names),
    }


def instrument_ids_to_assetids(instrument_ids: pd.Series) -> pd.Series:
    """把 SH.600000/SZ.000001 等项目代码转换为 Barra CN600000/CN000001。"""

    values = instrument_ids.astype(str)
    valid = values.str.fullmatch(INSTRUMENT_PATTERN)  # 校验证券代码格式
    if not bool(valid.all()):
        examples = values.loc[~valid].drop_duplicates().head(10).tolist()
        raise ValueError(f"instrument_id 格式非法：{examples}")
    return ("CN" + values.str[-6:]).rename("assetid")


def _normalize_date_strings(values: pd.Series) -> pd.Series:
    """把输入日期统一规范为八位 YYYYMMDD 字符串。"""
    if pd.api.types.is_datetime64_any_dtype(values):
        parsed = pd.to_datetime(values, errors="raise")
    else:
        normalized = values.astype(str).str.replace(r"\.0$", "", regex=True)
        parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="raise")
    return parsed.dt.strftime("%Y%m%d")


def _load_sse_calendar(calendar_path: Path) -> pd.DatetimeIndex:
    """加载该流程所需数据，并检查形状、业务键和来源一致性。"""
    calendar = pd.read_parquet(
        calendar_path,
        columns=["trade_days", "s_info_exchmarket"],
        filters=[("s_info_exchmarket", "==", "SSE")],
    )
    dates = pd.DatetimeIndex(  # 构造有序交易日索引
        pd.to_datetime(
            _normalize_date_strings(calendar["trade_days"]),
            format="%Y%m%d",
        ).sort_values().unique(),
        name="trade_date",
    )
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("SSE 交易日历为空、重复或未排序")
    return dates


def _instrument_ids_to_windcodes(instrument_ids: pd.Series) -> pd.Series:
    """批量把交易所前缀股票代码转换为 Wind 代码。"""
    values = instrument_ids.astype(str)
    parts = values.str.extract(  # 拆分证券代码字段
        r"^(?P<exchange>SH|SZ|BJ)\.(?P<number>\d{6})$"
    )
    if parts.isna().any(axis=None):
        examples = values.loc[parts.isna().any(axis=1)].head(10).tolist()
        raise ValueError(f"instrument_id 格式非法：{examples}")
    return (parts["number"] + "." + parts["exchange"]).rename("windcode")


def _attach_reversal_20d(
    prediction: pd.DataFrame,
    eod_path: Path,
    calendar_dates: pd.DatetimeIndex,
    year: int,
) -> pd.DataFrame:
    """按 SSE 精确 T-20 交易日计算负累计复权收益，不做填充。"""

    result = prediction.copy()
    signal_dates = pd.to_datetime(
        result["trade_date"], format="%Y%m%d", errors="raise"
    )
    positions = calendar_dates.get_indexer(signal_dates)  # 反转窗口以 SSE 交易日序号而非自然日计算。
    if (positions < 0).any():
        raise ValueError(f"{year} 预测日期无法映射到 SSE 交易日历")
    if (positions < REVERSAL_LOOKBACK_DAYS).any():
        raise ValueError(f"{year} 交易日历不足 T-{REVERSAL_LOOKBACK_DAYS}")
    lag_dates = calendar_dates[positions - REVERSAL_LOOKBACK_DAYS]  # 精确取 T-20，不使用最近可得价格替代。
    if not bool((lag_dates < signal_dates.to_numpy()).all()):
        raise AssertionError("20 日反转错误使用了当日或未来价格")

    result["_source_order"] = np.arange(len(result), dtype=np.int64)  # 合并后用该列恢复原预测顺序。
    result["_windcode"] = _instrument_ids_to_windcodes(
        result["instrument_id"]
    ).to_numpy()
    result["_lag_trade_date"] = lag_dates.strftime("%Y%m%d")  # 统一为 Wind 日期格式
    first_date = str(result["_lag_trade_date"].min())
    last_date = str(result["trade_date"].max())
    eod = pd.read_parquet(
        eod_path,
        columns=["s_info_windcode", "trade_dt", "s_dq_adjclose"],
        filters=[
            ("trade_dt", ">=", first_date),
            ("trade_dt", "<=", last_date),
        ],
    )
    eod["trade_date"] = _normalize_date_strings(eod["trade_dt"])  # 统一 Wind 日期为 YYYYMMDD。
    eod["_windcode"] = eod["s_info_windcode"].astype(str)
    eod = eod.loc[
        eod["_windcode"].isin(set(result["_windcode"])),
        ["trade_date", "_windcode", "s_dq_adjclose"],
    ].copy()
    if eod.duplicated(["trade_date", "_windcode"]).any():
        raise ValueError(f"{year} Wind 复权价业务键重复")
    price = eod["s_dq_adjclose"].to_numpy(np.float64)
    eod.loc[~(np.isfinite(price) & (price > 0)), "s_dq_adjclose"] = np.nan

    current = eod.rename(columns={"s_dq_adjclose": "_price_t"})
    lagged = eod.rename(
        columns={
            "trade_date": "_lag_trade_date",
            "s_dq_adjclose": "_price_t_minus_20",
        }
    )
    result = result.merge(  # 按股票与交易日业务键对齐
        current,
        on=["trade_date", "_windcode"],
        how="left",
        validate="many_to_one",
        sort=False,
    ).merge(
        lagged,
        on=["_lag_trade_date", "_windcode"],
        how="left",
        validate="many_to_one",
        sort=False,
    )
    result.sort_values("_source_order", inplace=True)
    result.reset_index(drop=True, inplace=True)
    valid = (  # 两端复权价同时存在才计算反转因子。
        result["_price_t"].notna()
        & result["_price_t_minus_20"].notna()
    )
    reversal = np.full(len(result), np.nan, dtype=np.float64)
    reversal[valid.to_numpy()] = -(  # 过去涨幅取负号，定义为 20 日反转。
        result.loc[valid, "_price_t"].to_numpy(np.float64)
        / result.loc[valid, "_price_t_minus_20"].to_numpy(np.float64)
        - 1.0
    )
    if not np.isfinite(reversal[valid.to_numpy()]).all():
        raise ValueError(f"{year} 20 日反转含 NaN 或 Inf")
    result[REVERSAL_COLUMN] = reversal
    result[REVERSAL_VALID_COLUMN] = valid.to_numpy(bool)
    return result.drop(
        columns=[
            "_source_order", "_windcode", "_lag_trade_date",
            "_price_t", "_price_t_minus_20",
        ]
    )


def _standardized_reversal(daily: pd.DataFrame) -> np.ndarray:
    """执行横截面标准化，并显式处理缺失值和常数截面。"""
    valid = daily[REVERSAL_VALID_COLUMN].to_numpy(bool)
    raw = daily[REVERSAL_COLUMN].to_numpy(np.float64)
    if not np.array_equal(valid, np.isfinite(raw)):
        raise ValueError("20 日反转有效标记与数值缺失不一致")
    if int(valid.sum()) < 3:
        raise ValueError("单日有效 20 日反转不足 3 个观测")
    mean = float(raw[valid].mean())
    std = float(raw[valid].std(ddof=0))
    if not np.isfinite(std) or std <= np.finfo(np.float64).eps:
        raise ValueError("单日 20 日反转为常数")
    standardized = np.zeros(len(daily), dtype=np.float64)  # 缺失反转暴露填零并保留独立 valid 标记。
    standardized[valid] = (raw[valid] - mean) / std
    return standardized


def _load_cne5s_year(cne5s_path: Path, year: int) -> pd.DataFrame:
    """加载该流程所需数据，并检查形状、业务键和来源一致性。"""
    dataset = ds.dataset(cne5s_path, format="parquet")  # 打开分区 Parquet 数据集
    table = dataset.to_table(  # 下推日期过滤，仅读取目标分区
        columns=list(REQUIRED_CNE5S_COLUMNS),
        filter=(
            (ds.field("datadate") >= pd.Timestamp(f"{year}-01-01"))
            & (ds.field("datadate") < pd.Timestamp(f"{year + 1}-01-01"))
        ),
    )
    frame = table.to_pandas()
    if frame.empty:
        raise ValueError(f"CNE5S 没有 {year} 数据")
    frame["trade_date"] = pd.to_datetime(frame["datadate"]).dt.strftime(  # 统一为 Wind 日期格式
        "%Y%m%d"
    )
    frame["assetid"] = frame["assetid"].astype(str)
    valid_assets = frame["assetid"].str.fullmatch(CNE5S_ASSET_PATTERN)  # 校验证券代码格式
    if not bool(valid_assets.all()):
        examples = frame.loc[~valid_assets, "assetid"].drop_duplicates()  # 业务键去重
        raise ValueError(
            f"{year} CNE5S assetid 格式非法：{examples.head(10).tolist()}"
        )
    if frame.duplicated(["trade_date", "assetid"]).any():
        duplicates = (
            frame.loc[
                frame.duplicated(["trade_date", "assetid"], keep=False),
                ["trade_date", "assetid"],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"{year} CNE5S 业务键重复：{duplicates}")
    style_values = frame.loc[:, STYLE_COLUMNS].to_numpy(np.float64)
    if not np.isfinite(style_values).all():
        raise ValueError(f"{year} CNE5S 风格暴露含 NaN 或 Inf")
    if frame[INDUSTRY_COLUMN].isna().any():
        raise ValueError(f"{year} CNE5S 行业字段含缺失")
    return frame[
        ["trade_date", "assetid", INDUSTRY_COLUMN, *STYLE_COLUMNS]
    ].copy()


def _join_prediction_and_cne5s(
    prediction: pd.DataFrame, barra: pd.DataFrame, year: int
) -> pd.DataFrame:
    """处理预测文件、预测列或对应的样本数量审计。"""
    source = prediction.copy()
    source["_source_order"] = np.arange(len(source), dtype=np.int64)  # 对齐 Barra 后恢复因子原始行序。
    joined = source.merge(  # 按业务键对齐，避免依赖原始行序
        barra,
        on=["trade_date", "assetid"],
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    joined.sort_values("_source_order", inplace=True)
    joined.reset_index(drop=True, inplace=True)  # 按股票与交易日业务键对齐
    if not np.array_equal(
        joined["_source_order"].to_numpy(),
        np.arange(len(source), dtype=np.int64),
    ):
        raise AssertionError(f"{year} 合并后源预测顺序改变")
    joined["barra_matched"] = joined["_merge"].eq("both")  # 按股票与交易日业务键对齐
    return joined


def _match_record(
    joined: pd.DataFrame, year: int, cne5s_rows: int
) -> dict[str, Any]:
    """统计或检查数据连接覆盖率。"""
    matched = int(joined["barra_matched"].sum())
    reversal_valid = int(joined[REVERSAL_VALID_COLUMN].sum())
    correlation_valid = int(
        (joined["barra_matched"] & joined[REVERSAL_VALID_COLUMN]).sum()
    )
    rows = len(joined)
    return {
        "year": year,
        "prediction_rows": rows,
        "cne5s_rows": int(cne5s_rows),
        "matched_rows": matched,
        "unmatched_rows": rows - matched,
        "match_rate": float(matched / rows) if rows else np.nan,
        "reversal_valid_rows": reversal_valid,
        "reversal_missing_rows": rows - reversal_valid,
        "reversal_coverage": (
            float(reversal_valid / rows) if rows else np.nan
        ),
        "reversal_correlation_rows": correlation_valid,
        "dates": int(joined["trade_date"].nunique()),
        "first_date": str(joined["trade_date"].min()) if rows else None,
        "last_date": str(joined["trade_date"].max()) if rows else None,
        "status": "passed",
    }


def _validate_cne5s_schema(cne5s_path: Path) -> dict[str, Any]:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    fingerprint = _fingerprint_parquet(cne5s_path)  # 记录源文件指纹
    missing = [
        column
        for column in REQUIRED_CNE5S_COLUMNS
        if column not in fingerprint["columns"]
    ]
    if missing:
        raise ValueError(f"CNE5S 缺少字段：{missing}")
    return fingerprint


def _validate_price_sources(
    eod_path: Path, calendar_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    eod_fingerprint = _fingerprint_parquet(eod_path)  # 记录源文件指纹
    calendar_fingerprint = _fingerprint_parquet(calendar_path)  # 记录源文件指纹
    required_eod = {"s_info_windcode", "trade_dt", "s_dq_adjclose"}  # 复权价最小字段契约。
    required_calendar = {"trade_days", "s_info_exchmarket"}  # 交易日历最小字段契约。
    missing_eod = required_eod.difference(eod_fingerprint["columns"])  # 检查输入缺失字段
    missing_calendar = required_calendar.difference(  # 检查输入缺失字段
        calendar_fingerprint["columns"]
    )
    if missing_eod:
        raise ValueError(f"Wind 复权价缺少字段：{sorted(missing_eod)}")
    if missing_calendar:
        raise ValueError(f"Wind 交易日历缺少字段：{sorted(missing_calendar)}")
    return eod_fingerprint, calendar_fingerprint


def _check_match_rate(
    record: dict[str, Any], minimum_match_rate: float
) -> None:
    """统计或检查数据连接覆盖率。"""
    if float(record["match_rate"]) < minimum_match_rate:
        raise AssertionError(
            f"{record['year']} CNE5S 匹配率 "
            f"{record['match_rate']:.6%} < {minimum_match_rate:.6%}"
        )
