"""阶段 02：读取 CNE5S、交易日历与复权价，并构造可复用的 Barra 原始特征。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# 读取 `Path(__file__).resolve().parent` 的当前值，并赋给 `VIVIT_DIR`。
VIVIT_DIR = Path(__file__).resolve().parent
# 读取 `VIVIT_DIR.parent` 的当前值，并赋给 `PROJECT_ROOT`。
PROJECT_ROOT = VIVIT_DIR.parent
# 计算 `VIVIT_DIR / 'data'`，并将数值结果保存到 `DEFAULT_VIVIT_DATA_ROOT`。
DEFAULT_VIVIT_DATA_ROOT = VIVIT_DIR / "data"
# 计算 `PROJECT_ROOT / 'CNE5S.parquet'`，并将数值结果保存到 `DEFAULT_CNE5S_PATH`。
DEFAULT_CNE5S_PATH = PROJECT_ROOT / "CNE5S.parquet"
# 计算 `PROJECT_ROOT / 'Wind' / 'ASHAREEODPRICES.parquet'`，并将数值结果保存到 `DEFAULT_EOD_PATH`。
DEFAULT_EOD_PATH = PROJECT_ROOT / "Wind" / "ASHAREEODPRICES.parquet"
# 计算 `PROJECT_ROOT / 'Wind' / 'ASHARECALENDAR.parquet'`，并将数值结果保存到 `DEFAULT_CALENDAR_PATH`。
DEFAULT_CALENDAR_PATH = PROJECT_ROOT / "Wind" / "ASHARECALENDAR.parquet"
# 设置 `DEFAULT_SOURCE_RUN_NAME` 的固定配置值，供后续流程统一使用。
DEFAULT_SOURCE_RUN_NAME = "sigmoid_daybatch_rankic"
# 计算 `f'{DEFAULT_SOURCE_RUN_NAME}_barra_residual'`，并将结果写入 `DEFAULT_RESIDUAL_RUN_NAME`。
DEFAULT_RESIDUAL_RUN_NAME = f"{DEFAULT_SOURCE_RUN_NAME}_barra_residual"
# 设置 `PREDICTION_PATTERN` 的固定配置值，供后续流程统一使用。
PREDICTION_PATTERN = "vivit_predictions.{year}.parquet"
# 设置 `SOURCE_FACTOR_COLUMN` 的固定配置值，供后续流程统一使用。
SOURCE_FACTOR_COLUMN = "prediction_ensemble"
# 组织一组有序值，并保存到 `SOURCE_FACTOR_COLUMNS`。
SOURCE_FACTOR_COLUMNS = (
    "prediction_ensemble",
    "prediction_ensemble_unprojected",
    "prediction_ensemble_projected",
    "prediction_ensemble_raw",
    "prediction_ensemble_alpha",
    "prediction_ensemble_style",
    "prediction_ensemble_total_return",
)
# 设置 `INDUSTRY_COLUMN` 的固定配置值，供后续流程统一使用。
INDUSTRY_COLUMN = "cne5s_industry"
# 设置 `UNKNOWN_INDUSTRY` 的固定配置值，供后续流程统一使用。
UNKNOWN_INDUSTRY = "CNE5S_UNKNOWN"
# 设置 `MINIMUM_MATCH_RATE` 的固定配置值，供后续流程统一使用。
MINIMUM_MATCH_RATE = 0.999
# 设置 `DERIVATION_SCHEMA_VERSION` 的固定配置值，供后续流程统一使用。
DERIVATION_SCHEMA_VERSION = 5
# 设置 `HAC_MAX_LAG` 的固定配置值，供后续流程统一使用。
HAC_MAX_LAG = 20
# 设置 `ORTHOGONALITY_TOLERANCE` 的固定配置值，供后续流程统一使用。
ORTHOGONALITY_TOLERANCE = 1e-8
# 设置 `REVERSAL_COLUMN` 的固定配置值，供后续流程统一使用。
REVERSAL_COLUMN = "reversal_20d"
# 设置 `REVERSAL_VALID_COLUMN` 的固定配置值，供后续流程统一使用。
REVERSAL_VALID_COLUMN = "reversal_20d_valid"
# 设置 `REVERSAL_LOOKBACK_DAYS` 的固定配置值，供后续流程统一使用。
REVERSAL_LOOKBACK_DAYS = 20

# 组织一组有序值，并保存到 `STYLE_COLUMNS`。
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
# 构造键值映射，并保存到 `STYLE_LABELS`，供后续按键查询。
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
# 组织一组有序值，并保存到 `CORRELATION_COLUMNS`。
CORRELATION_COLUMNS = (*STYLE_COLUMNS, REVERSAL_COLUMN)
# 读取 `CORRELATION_COLUMNS` 的当前值，并赋给 `REGRESSION_COLUMNS`。
REGRESSION_COLUMNS = CORRELATION_COLUMNS
# 设置 `VIVIT_MATRIX_COLUMN` 的固定配置值，供后续流程统一使用。
VIVIT_MATRIX_COLUMN = "vivit_factor"
# 组织一组有序值，并保存到 `MATRIX_COLUMNS`。
MATRIX_COLUMNS = (VIVIT_MATRIX_COLUMN, *CORRELATION_COLUMNS)
# 构造键值映射，并保存到 `MATRIX_LABELS`，供后续按键查询。
MATRIX_LABELS = {
    VIVIT_MATRIX_COLUMN: "ViViT因子",
    **STYLE_LABELS,
}
# 组织一组有序值，并保存到 `REQUIRED_CNE5S_COLUMNS`。
REQUIRED_CNE5S_COLUMNS = (
    "datadate",
    "assetid",
    INDUSTRY_COLUMN,
    *STYLE_COLUMNS,
)
# 调用 `re.compile` 编译正则表达式供后续重复匹配，并将结果保存到 `INSTRUMENT_PATTERN`。
INSTRUMENT_PATTERN = re.compile(r"^(SH|SZ|BJ)\.(\d{6})$")
# 调用 `re.compile` 编译正则表达式供后续重复匹配，并将结果保存到 `CNE5S_ASSET_PATTERN`。
CNE5S_ASSET_PATTERN = re.compile(r"^CN\d{6}$")


# 定义内部辅助函数 `_fingerprint_parquet`，封装仅供本模块复用的计算步骤。
def _fingerprint_parquet(path: Path) -> dict[str, Any]:
    """提取文件身份信息，用于来源审计和复用判断。
    
    输入：
        path: Path。
    输出：
        dict[str, Any]。
        """
    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(path)
    # 调用 `pq.ParquetFile` 打开 Parquet 元数据以便分批读取，并将结果保存到 `parquet`。
    parquet = pq.ParquetFile(path)
    # 将 `{'path': str(path.resolve()), 'size': int(path.stat().st_size), 'mtime_ns': int(path.stat().st_…` 作为当前函数结果返回给调用方。
    return {
        "path": str(path.resolve()),
        "size": int(path.stat().st_size),
        "mtime_ns": int(path.stat().st_mtime_ns),
        "rows": int(parquet.metadata.num_rows),
        "row_groups": int(parquet.num_row_groups),
        "columns": list(parquet.schema_arrow.names),
    }


# 定义 `instrument_ids_to_assetids` 函数，封装这一阶段可独立调用的处理步骤。
def instrument_ids_to_assetids(instrument_ids: pd.Series) -> pd.Series:
    """把 SH.600000/SZ.000001 等项目代码转换为 Barra CN600000/CN000001。
    
    输入：
        instrument_ids: pd.Series。
    输出：
        pd.Series。
        """

    # 调用 `instrument_ids.astype` 统一设备或数据类型，并将结果保存到 `values`。
    values = instrument_ids.astype(str)
    # 调用 `values.str.fullmatch` 检查证券代码是否完整符合目标格式，并将结果保存到 `valid`。
    valid = values.str.fullmatch(INSTRUMENT_PATTERN)
    # 检查 `not bool(valid.all())`；根据条件结果选择对应处理分支。
    if not bool(valid.all()):
        # 调用 `values.loc[~valid].drop_duplicates().head(10).tolist` 把张量或数组转换为 Python 列表，并将结果保存到 `examples`。
        examples = values.loc[~valid].drop_duplicates().head(10).tolist()
        # 抛出 `ValueError(f'instrument_id 格式非法：{examples}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"instrument_id 格式非法：{examples}")
    # 将 `('CN' + values.str[-6:]).rename('assetid')` 作为当前函数结果返回给调用方。
    return ("CN" + values.str[-6:]).rename("assetid")


# 定义内部辅助函数 `_normalize_date_strings`，封装仅供本模块复用的计算步骤。
def _normalize_date_strings(values: pd.Series) -> pd.Series:
    """把输入日期统一规范为八位 YYYYMMDD 字符串。
    
    输入：
        values: pd.Series。
    输出：
        pd.Series。
        """
    # 检查 `pd.api.types.is_datetime64_any_dtype(values)`；根据条件结果选择对应处理分支。
    if pd.api.types.is_datetime64_any_dtype(values):
        # 调用 `pd.to_datetime` 把日期字段转换为 Pandas 时间类型，并将结果保存到 `parsed`。
        parsed = pd.to_datetime(values, errors="raise")
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `values.astype(str).str.replace` 把完整临时产物替换为正式输出，并将结果保存到 `normalized`。
        normalized = values.astype(str).str.replace(r"\.0$", "", regex=True)
        # 调用 `pd.to_datetime` 把日期字段转换为 Pandas 时间类型，并将结果保存到 `parsed`。
        parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="raise")
    # 将 `parsed.dt.strftime('%Y%m%d')` 作为当前函数结果返回给调用方。
    return parsed.dt.strftime("%Y%m%d")


# 定义内部辅助函数 `_load_sse_calendar`，封装仅供本模块复用的计算步骤。
def _load_sse_calendar(calendar_path: Path) -> pd.DatetimeIndex:
    """加载该流程所需数据，并检查形状、业务键和来源一致性。
    
    输入：
        calendar_path: Path。
    输出：
        pd.DatetimeIndex。
        """
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `calendar`。
    calendar = pd.read_parquet(
        calendar_path,
        columns=["trade_days", "s_info_exchmarket"],
        filters=[("s_info_exchmarket", "==", "SSE")],
    )
    # 调用 `pd.DatetimeIndex` 构造有序日期索引以完成交易日对齐，并将结果保存到 `dates`。
    dates = pd.DatetimeIndex(
        pd.to_datetime(
            _normalize_date_strings(calendar["trade_days"]),
            format="%Y%m%d",
        ).sort_values().unique(),
        name="trade_date",
    )
    # 检查 `dates.empty or dates.has_duplicates or (not dates.is_monotonic_increasing)`；根据条件结果选择对应处理分支。
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        # 抛出 `ValueError('SSE 交易日历为空、重复或未排序')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("SSE 交易日历为空、重复或未排序")
    # 将 `dates` 作为当前函数结果返回给调用方。
    return dates


# 定义内部辅助函数 `_instrument_ids_to_windcodes`，封装仅供本模块复用的计算步骤。
def _instrument_ids_to_windcodes(instrument_ids: pd.Series) -> pd.Series:
    """批量把交易所前缀股票代码转换为 Wind 代码。
    
    输入：
        instrument_ids: pd.Series。
    输出：
        pd.Series。
        """
    # 调用 `instrument_ids.astype` 统一设备或数据类型，并将结果保存到 `values`。
    values = instrument_ids.astype(str)
    # 调用 `values.str.extract` 按正则捕获组拆分字符串字段，并将结果保存到 `parts`。
    parts = values.str.extract(
        r"^(?P<exchange>SH|SZ|BJ)\.(?P<number>\d{6})$"
    )
    # 检查 `parts.isna().any(axis=None)`；根据条件结果选择对应处理分支。
    if parts.isna().any(axis=None):
        # 调用 `values.loc[parts.isna().any(axis=1)].head(10).tolist` 把张量或数组转换为 Python 列表，并将结果保存到 `examples`。
        examples = values.loc[parts.isna().any(axis=1)].head(10).tolist()
        # 抛出 `ValueError(f'instrument_id 格式非法：{examples}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"instrument_id 格式非法：{examples}")
    # 将 `(parts['number'] + '.' + parts['exchange']).rename('windcode')` 作为当前函数结果返回给调用方。
    return (parts["number"] + "." + parts["exchange"]).rename("windcode")


# 定义内部辅助函数 `_attach_reversal_20d`，封装仅供本模块复用的计算步骤。
def _attach_reversal_20d(
    prediction: pd.DataFrame,
    eod_path: Path,
    calendar_dates: pd.DatetimeIndex,
    year: int,
) -> pd.DataFrame:
    """按 SSE 精确 T-20 交易日计算负累计复权收益，不做填充。
    
    输入：
        prediction: pd.DataFrame。
        eod_path: Path。
        calendar_dates: pd.DatetimeIndex。
        year: int。
    输出：
        pd.DataFrame。
        """

    # 复制 `prediction.copy`，避免后续更新污染原对象，并保存到 `result`。
    result = prediction.copy()
    # 调用 `pd.to_datetime` 把日期字段转换为 Pandas 时间类型，并将结果保存到 `signal_dates`。
    signal_dates = pd.to_datetime(
        result["trade_date"], format="%Y%m%d", errors="raise"
    )
    # 调用 `calendar_dates.get_indexer` 把目标标签映射为索引位置，并将结果保存到 `positions`。
    positions = calendar_dates.get_indexer(signal_dates)
    # 检查 `(positions < 0).any()`；根据条件结果选择对应处理分支。
    if (positions < 0).any():
        # 抛出 `ValueError(f'{year} 预测日期无法映射到 SSE 交易日历')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} 预测日期无法映射到 SSE 交易日历")
    # 检查 `(positions < REVERSAL_LOOKBACK_DAYS).any()`；根据条件结果选择对应处理分支。
    if (positions < REVERSAL_LOOKBACK_DAYS).any():
        # 抛出 `ValueError(f'{year} 交易日历不足 T-{REVERSAL_LOOKBACK_DAYS}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} 交易日历不足 T-{REVERSAL_LOOKBACK_DAYS}")
    # 读取 `calendar_dates[positions - REVERSAL_LOOKBACK_DAYS]` 的当前值，并赋给 `lag_dates`。
    lag_dates = calendar_dates[positions - REVERSAL_LOOKBACK_DAYS]
    # 检查 `not bool((lag_dates < signal_dates.to_numpy()).all())`；根据条件结果选择对应处理分支。
    if not bool((lag_dates < signal_dates.to_numpy()).all()):
        # 抛出 `AssertionError('20 日反转错误使用了当日或未来价格')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("20 日反转错误使用了当日或未来价格")

    # 调用 `np.arange` 按目标形状初始化数值容器 `result['_source_order']`。
    result["_source_order"] = np.arange(len(result), dtype=np.int64)
    # 调用 `_instrument_ids_to_windcodes(result['instrument_id']).to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `result['_windcode']`。
    result["_windcode"] = _instrument_ids_to_windcodes(
        result["instrument_id"]
    ).to_numpy()
    # 调用 `lag_dates.strftime` 把日期格式化为 Wind 数据使用的 YYYYMMDD 字符串，并将结果保存到 `result['_lag_trade_date']`。
    result["_lag_trade_date"] = lag_dates.strftime("%Y%m%d")
    # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `first_date`。
    first_date = str(result["_lag_trade_date"].min())
    # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `last_date`。
    last_date = str(result["trade_date"].max())
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `eod`。
    eod = pd.read_parquet(
        eod_path,
        columns=["s_info_windcode", "trade_dt", "s_dq_adjclose"],
        filters=[
            ("trade_dt", ">=", first_date),
            ("trade_dt", "<=", last_date),
        ],
    )
    # 调用 `_normalize_date_strings` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `eod['trade_date']`。
    eod["trade_date"] = _normalize_date_strings(eod["trade_dt"])
    # 调用 `eod['s_info_windcode'].astype` 统一设备或数据类型，并将结果保存到 `eod['_windcode']`。
    eod["_windcode"] = eod["s_info_windcode"].astype(str)
    # 调用 `copy` 复制当前对象以避免后续原地修改污染源数据，并将结果保存到 `eod`。
    eod = eod.loc[
        eod["_windcode"].isin(set(result["_windcode"])),
        ["trade_date", "_windcode", "s_dq_adjclose"],
    ].copy()
    # 检查 `eod.duplicated(['trade_date', '_windcode']).any()`；根据条件结果选择对应处理分支。
    if eod.duplicated(["trade_date", "_windcode"]).any():
        # 抛出 `ValueError(f'{year} Wind 复权价业务键重复')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} Wind 复权价业务键重复")
    # 调用 `eod['s_dq_adjclose'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `price`。
    price = eod["s_dq_adjclose"].to_numpy(np.float64)
    # 读取 `np.nan` 的当前值，并赋给 `eod.loc[~(np.isfinite(price) & (price > 0)), 's_dq_adjclose']`。
    eod.loc[~(np.isfinite(price) & (price > 0)), "s_dq_adjclose"] = np.nan

    # 调用 `eod.rename` 把完整临时产物替换为正式输出，并将结果保存到 `current`。
    current = eod.rename(columns={"s_dq_adjclose": "_price_t"})
    # 调用 `eod.rename` 把完整临时产物替换为正式输出，并将结果保存到 `lagged`。
    lagged = eod.rename(
        columns={
            "trade_date": "_lag_trade_date",
            "s_dq_adjclose": "_price_t_minus_20",
        }
    )
    # 调用 `result.merge(current, on=['trade_date', '_windcode'], how='left…` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `result`。
    result = result.merge(
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
    # 调用 `result.sort_values` 根据分数计算截面排序或排序索引。
    result.sort_values("_source_order", inplace=True)
    # 调用 `result.reset_index` 重建连续行索引并丢弃旧索引。
    result.reset_index(drop=True, inplace=True)
    # 计算 `result['_price_t'].notna() & result['_price_t_minus_20'].notna()`，并将数值结果保存到 `valid`。
    valid = (
        result["_price_t"].notna()
        & result["_price_t_minus_20"].notna()
    )
    # 调用 `np.full` 按目标形状初始化数值容器 `reversal`。
    reversal = np.full(len(result), np.nan, dtype=np.float64)
    # 计算 `-(result.loc[valid, '_price_t'].to_numpy(np.float64) / result.loc[valid, '_price_t_minus_20'].t…`，并将数值结果保存到 `reversal[valid.to_numpy()]`。
    reversal[valid.to_numpy()] = -(
        result.loc[valid, "_price_t"].to_numpy(np.float64)
        / result.loc[valid, "_price_t_minus_20"].to_numpy(np.float64)
        - 1.0
    )
    # 检查 `not np.isfinite(reversal[valid.to_numpy()]).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(reversal[valid.to_numpy()]).all():
        # 抛出 `ValueError(f'{year} 20 日反转含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} 20 日反转含 NaN 或 Inf")
    # 读取 `reversal` 的当前值，并赋给 `result[REVERSAL_COLUMN]`。
    result[REVERSAL_COLUMN] = reversal
    # 调用 `valid.to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `result[REVERSAL_VALID_COLUMN]`。
    result[REVERSAL_VALID_COLUMN] = valid.to_numpy(bool)
    # 将 `result.drop(columns=['_source_order', '_windcode', '_lag_trade_date', '_price_t', '_price_t_min…` 作为当前函数结果返回给调用方。
    return result.drop(
        columns=[
            "_source_order", "_windcode", "_lag_trade_date",
            "_price_t", "_price_t_minus_20",
        ]
    )


# 定义内部辅助函数 `_standardized_reversal`，封装仅供本模块复用的计算步骤。
def _standardized_reversal(daily: pd.DataFrame) -> np.ndarray:
    """执行横截面标准化，并显式处理缺失值和常数截面。
    
    输入：
        daily: pd.DataFrame。
    输出：
        np.ndarray。
        """
    # 调用 `daily[REVERSAL_VALID_COLUMN].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `valid`。
    valid = daily[REVERSAL_VALID_COLUMN].to_numpy(bool)
    # 调用 `daily[REVERSAL_COLUMN].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `raw`。
    raw = daily[REVERSAL_COLUMN].to_numpy(np.float64)
    # 检查 `not np.array_equal(valid, np.isfinite(raw))`；根据条件结果选择对应处理分支。
    if not np.array_equal(valid, np.isfinite(raw)):
        # 抛出 `ValueError('20 日反转有效标记与数值缺失不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("20 日反转有效标记与数值缺失不一致")
    # 检查 `int(valid.sum()) < 3`；根据条件结果选择对应处理分支。
    if int(valid.sum()) < 3:
        # 抛出 `ValueError('单日有效 20 日反转不足 3 个观测')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日有效 20 日反转不足 3 个观测")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `mean`。
    mean = float(raw[valid].mean())
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `std`。
    std = float(raw[valid].std(ddof=0))
    # 检查 `not np.isfinite(std) or std <= np.finfo(np.float64).eps`；根据条件结果选择对应处理分支。
    if not np.isfinite(std) or std <= np.finfo(np.float64).eps:
        # 抛出 `ValueError('单日 20 日反转为常数')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("单日 20 日反转为常数")
    # 调用 `np.zeros` 按目标形状初始化数值容器 `standardized`。
    standardized = np.zeros(len(daily), dtype=np.float64)
    # 计算 `(raw[valid] - mean) / std`，并将数值结果保存到 `standardized[valid]`。
    standardized[valid] = (raw[valid] - mean) / std
    # 将 `standardized` 作为当前函数结果返回给调用方。
    return standardized


# 定义内部辅助函数 `_load_cne5s_year`，封装仅供本模块复用的计算步骤。
def _load_cne5s_year(cne5s_path: Path, year: int) -> pd.DataFrame:
    """加载该流程所需数据，并检查形状、业务键和来源一致性。
    
    输入：
        cne5s_path: Path。
        year: int。
    输出：
        pd.DataFrame。
        """
    # 调用 `ds.dataset` 把分区 Parquet 目录打开为 Arrow Dataset，并将结果保存到 `dataset`。
    dataset = ds.dataset(cne5s_path, format="parquet")
    # 调用 `dataset.to_table` 按字段和日期过滤条件读取 Arrow 表，并将结果保存到 `table`。
    table = dataset.to_table(
        columns=list(REQUIRED_CNE5S_COLUMNS),
        filter=(
            (ds.field("datadate") >= pd.Timestamp(f"{year}-01-01"))
            & (ds.field("datadate") < pd.Timestamp(f"{year + 1}-01-01"))
        ),
    )
    # 调用 `table.to_pandas` 把 Arrow 表转换为 Pandas DataFrame，并将结果保存到 `frame`。
    frame = table.to_pandas()
    # 检查 `frame.empty`；根据条件结果选择对应处理分支。
    if frame.empty:
        # 抛出 `ValueError(f'CNE5S 没有 {year} 数据')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"CNE5S 没有 {year} 数据")
    # 调用 `pd.to_datetime(frame['datadate']).dt.strftime` 把日期格式化为 Wind 数据使用的 YYYYMMDD 字符串，并将结果保存到 `frame['trade_date']`。
    frame["trade_date"] = pd.to_datetime(frame["datadate"]).dt.strftime(
        "%Y%m%d"
    )
    # 调用 `frame['assetid'].astype` 统一设备或数据类型，并将结果保存到 `frame['assetid']`。
    frame["assetid"] = frame["assetid"].astype(str)
    # 调用 `frame['assetid'].str.fullmatch` 检查证券代码是否完整符合目标格式，并将结果保存到 `valid_assets`。
    valid_assets = frame["assetid"].str.fullmatch(CNE5S_ASSET_PATTERN)
    # 检查 `not bool(valid_assets.all())`；根据条件结果选择对应处理分支。
    if not bool(valid_assets.all()):
        # 调用 `frame.loc[~valid_assets, 'assetid'].drop_duplicates` 去除重复值并保留唯一记录，并将结果保存到 `examples`。
        examples = frame.loc[~valid_assets, "assetid"].drop_duplicates()
        # 抛出 `ValueError(f'{year} CNE5S assetid 格式非法：{examples.head(10).tolist()}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"{year} CNE5S assetid 格式非法：{examples.head(10).tolist()}"
        )
    # 检查 `frame.duplicated(['trade_date', 'assetid']).any()`；根据条件结果选择对应处理分支。
    if frame.duplicated(["trade_date", "assetid"]).any():
        # 调用 `to_dict` 把表格记录转换为便于写入报告的字典列表，并将结果保存到 `duplicates`。
        duplicates = (
            frame.loc[
                frame.duplicated(["trade_date", "assetid"], keep=False),
                ["trade_date", "assetid"],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        # 抛出 `ValueError(f'{year} CNE5S 业务键重复：{duplicates}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} CNE5S 业务键重复：{duplicates}")
    # 调用 `frame.loc[:, STYLE_COLUMNS].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `style_values`。
    style_values = frame.loc[:, STYLE_COLUMNS].to_numpy(np.float64)
    # 检查 `not np.isfinite(style_values).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(style_values).all():
        # 抛出 `ValueError(f'{year} CNE5S 风格暴露含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} CNE5S 风格暴露含 NaN 或 Inf")
    # 检查 `frame[INDUSTRY_COLUMN].isna().any()`；根据条件结果选择对应处理分支。
    if frame[INDUSTRY_COLUMN].isna().any():
        # 抛出 `ValueError(f'{year} CNE5S 行业字段含缺失')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"{year} CNE5S 行业字段含缺失")
    # 将 `frame[['trade_date', 'assetid', INDUSTRY_COLUMN, *STYLE_COLUMNS]].copy()` 作为当前函数结果返回给调用方。
    return frame[
        ["trade_date", "assetid", INDUSTRY_COLUMN, *STYLE_COLUMNS]
    ].copy()


# 定义内部辅助函数 `_join_prediction_and_cne5s`，封装仅供本模块复用的计算步骤。
def _join_prediction_and_cne5s(
    prediction: pd.DataFrame, barra: pd.DataFrame, year: int
) -> pd.DataFrame:
    """处理预测文件、预测列或对应的样本数量审计。
    
    输入：
        prediction: pd.DataFrame。
        barra: pd.DataFrame。
        year: int。
    输出：
        pd.DataFrame。
        """
    # 复制 `prediction.copy`，避免后续更新污染原对象，并保存到 `source`。
    source = prediction.copy()
    # 调用 `np.arange` 按目标形状初始化数值容器 `source['_source_order']`。
    source["_source_order"] = np.arange(len(source), dtype=np.int64)
    # 调用 `source.merge` 按键合并数据，并将对齐后的结果保存到 `joined`。
    joined = source.merge(
        barra,
        on=["trade_date", "assetid"],
        how="left",
        validate="one_to_one",
        sort=False,
        indicator=True,
    )
    # 调用 `joined.sort_values` 根据分数计算截面排序或排序索引。
    joined.sort_values("_source_order", inplace=True)
    # 调用 `joined.reset_index` 按股票和交易日键对齐来自不同来源的数据。
    joined.reset_index(drop=True, inplace=True)
    # 检查 `not np.array_equal(joined['_source_order'].to_numpy(), np.arange(len(source), dtype=np.int64))`；根据条件结果选择对应处理分支。
    if not np.array_equal(
        joined["_source_order"].to_numpy(),
        np.arange(len(source), dtype=np.int64),
    ):
        # 抛出 `AssertionError(f'{year} 合并后源预测顺序改变')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(f"{year} 合并后源预测顺序改变")
    # 调用 `joined['_merge'].eq` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `joined['barra_matched']`。
    joined["barra_matched"] = joined["_merge"].eq("both")
    # 将 `joined` 作为当前函数结果返回给调用方。
    return joined


# 定义内部辅助函数 `_match_record`，封装仅供本模块复用的计算步骤。
def _match_record(
    joined: pd.DataFrame, year: int, cne5s_rows: int
) -> dict[str, Any]:
    """统计或检查数据连接覆盖率。
    
    输入：
        joined: pd.DataFrame。
        year: int。
        cne5s_rows: int。
    输出：
        dict[str, Any]。
        """
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `matched`。
    matched = int(joined["barra_matched"].sum())
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `reversal_valid`。
    reversal_valid = int(joined[REVERSAL_VALID_COLUMN].sum())
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `correlation_valid`。
    correlation_valid = int(
        (joined["barra_matched"] & joined[REVERSAL_VALID_COLUMN]).sum()
    )
    # 调用 `len` 汇总当前数据，并将统计量保存到 `rows`。
    rows = len(joined)
    # 将 `{'year': year, 'prediction_rows': rows, 'cne5s_rows': int(cne5s_rows), 'matched_rows': matched,…` 作为当前函数结果返回给调用方。
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


# 定义内部辅助函数 `_validate_cne5s_schema`，封装仅供本模块复用的计算步骤。
def _validate_cne5s_schema(cne5s_path: Path) -> dict[str, Any]:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        cne5s_path: Path。
    输出：
        dict[str, Any]。
        """
    # 调用 `_fingerprint_parquet` 计算源文件指纹以支持缓存一致性检查，并将结果保存到 `fingerprint`。
    fingerprint = _fingerprint_parquet(cne5s_path)
    # 构造当前步骤所需的列表，并保存到 `missing`。
    missing = [
        column
        for column in REQUIRED_CNE5S_COLUMNS
        if column not in fingerprint["columns"]
    ]
    # 检查 `missing`；根据条件结果选择对应处理分支。
    if missing:
        # 抛出 `ValueError(f'CNE5S 缺少字段：{missing}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"CNE5S 缺少字段：{missing}")
    # 将 `fingerprint` 作为当前函数结果返回给调用方。
    return fingerprint


# 定义内部辅助函数 `_validate_price_sources`，封装仅供本模块复用的计算步骤。
def _validate_price_sources(
    eod_path: Path, calendar_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        eod_path: Path。
        calendar_path: Path。
    输出：
        tuple[dict[str, Any], dict[str, Any]]。
        """
    # 调用 `_fingerprint_parquet` 计算源文件指纹以支持缓存一致性检查，并将结果保存到 `eod_fingerprint`。
    eod_fingerprint = _fingerprint_parquet(eod_path)
    # 调用 `_fingerprint_parquet` 计算源文件指纹以支持缓存一致性检查，并将结果保存到 `calendar_fingerprint`。
    calendar_fingerprint = _fingerprint_parquet(calendar_path)
    # 构造去重集合，并保存到 `required_eod`。
    required_eod = {"s_info_windcode", "trade_dt", "s_dq_adjclose"}
    # 构造去重集合，并保存到 `required_calendar`。
    required_calendar = {"trade_days", "s_info_exchmarket"}
    # 调用 `required_eod.difference` 计算两个集合的差集以找出缺失字段，并将结果保存到 `missing_eod`。
    missing_eod = required_eod.difference(eod_fingerprint["columns"])
    # 调用 `required_calendar.difference` 计算两个集合的差集以找出缺失字段，并将结果保存到 `missing_calendar`。
    missing_calendar = required_calendar.difference(
        calendar_fingerprint["columns"]
    )
    # 检查 `missing_eod`；根据条件结果选择对应处理分支。
    if missing_eod:
        # 抛出 `ValueError(f'Wind 复权价缺少字段：{sorted(missing_eod)}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"Wind 复权价缺少字段：{sorted(missing_eod)}")
    # 检查 `missing_calendar`；根据条件结果选择对应处理分支。
    if missing_calendar:
        # 抛出 `ValueError(f'Wind 交易日历缺少字段：{sorted(missing_calendar)}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(f"Wind 交易日历缺少字段：{sorted(missing_calendar)}")
    # 将 `(eod_fingerprint, calendar_fingerprint)` 作为当前函数结果返回给调用方。
    return eod_fingerprint, calendar_fingerprint


# 定义内部辅助函数 `_check_match_rate`，封装仅供本模块复用的计算步骤。
def _check_match_rate(
    record: dict[str, Any], minimum_match_rate: float
) -> None:
    """统计或检查数据连接覆盖率。
    
    输入：
        record: dict[str, Any]。
        minimum_match_rate: float。
    输出：
        None。
        """
    # 检查 `float(record['match_rate']) < minimum_match_rate`；根据条件结果选择对应处理分支。
    if float(record["match_rate"]) < minimum_match_rate:
        # 抛出 `AssertionError(f'{record['year']} CNE5S 匹配率 {record['match_rate']:.6%} < {minimum_match_rate:.6…`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            f"{record['year']} CNE5S 匹配率 "
            f"{record['match_rate']:.6%} < {minimum_match_rate:.6%}"
        )
