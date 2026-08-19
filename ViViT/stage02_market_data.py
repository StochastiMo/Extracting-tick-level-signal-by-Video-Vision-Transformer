"""阶段 02：读取 CNE5S、交易日历与复权价，并构造可复用的 Barra 原始特征。"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

VIVIT_DIR = Path(__file__).resolve().parent  # 定位vivit目录。
PROJECT_ROOT = VIVIT_DIR.parent  # 定位项目根目录。
DEFAULT_VIVIT_DATA_ROOT = VIVIT_DIR / "data"  # 定位默认vivit数据根目录。
DEFAULT_CNE5S_PATH = PROJECT_ROOT / "CNE5S.parquet"  # 定位默认cne5s路径。
DEFAULT_EOD_PATH = PROJECT_ROOT / "Wind" / "ASHAREEODPRICES.parquet"  # 定位默认日行情路径。
DEFAULT_CALENDAR_PATH = PROJECT_ROOT / "Wind" / "ASHARECALENDAR.parquet"  # 定位默认交易日历路径。
DEFAULT_SOURCE_RUN_NAME = "sigmoid_daybatch_rankic"  # 设置默认来源执行名称。
DEFAULT_RESIDUAL_RUN_NAME = f"{DEFAULT_SOURCE_RUN_NAME}_barra_residual"  # 设置默认残差执行名称。
PREDICTION_PATTERN = "vivit_predictions.{year}.parquet"  # 声明逐年预测文件名模板。
SOURCE_FACTOR_COLUMN = "prediction_ensemble"  # 设置来源因子列。
SOURCE_FACTOR_COLUMNS = (  # 集中声明来源因子列。
    "prediction_ensemble",
    "prediction_ensemble_unprojected",
    "prediction_ensemble_projected",
    "prediction_ensemble_raw",
    "prediction_ensemble_alpha",
    "prediction_ensemble_style",
    "prediction_ensemble_total_return",
)
INDUSTRY_COLUMN = "cne5s_industry"  # 设置行业列。
UNKNOWN_INDUSTRY = "CNE5S_UNKNOWN"  # 设置未知行业行业。
MINIMUM_MATCH_RATE = 0.999  # 设置业务键连接允许的最低覆盖率。
DERIVATION_SCHEMA_VERSION = 5  # 固定产物 schema 版本，用于判断缓存与复用兼容性。
HAC_MAX_LAG = 20  # 设置hacmaxlag。
ORTHOGONALITY_TOLERANCE = 1e-8  # 设置 Barra 投影正交性的数值验收阈值。
REVERSAL_COLUMN = "reversal_20d"  # 设置reversal列。
REVERSAL_VALID_COLUMN = "reversal_20d_valid"  # 设置reversal有效列。
REVERSAL_LOOKBACK_DAYS = 20  # 设置reversallookbackdays。

STYLE_COLUMNS = (  # 集中声明风格列。
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
STYLE_LABELS = {  # 设置风格标签。
    "cne5s_beta": "贝塔",  # 把 cne5s_beta 映射为"贝塔"。
    "cne5s_btop": "账面市值比",  # 把 cne5s_btop 映射为"账面市值比"。
    "cne5s_earnyild": "盈利收益率",  # 把 cne5s_earnyild 映射为"盈利收益率"。
    "cne5s_growth": "成长",  # 把 cne5s_growth 映射为"成长"。
    "cne5s_leverage": "杠杆",  # 把 cne5s_leverage 映射为"杠杆"。
    "cne5s_liquidty": "流动性",  # 把 cne5s_liquidty 映射为"流动性"。
    "cne5s_momentum": "动量",  # 把 cne5s_momentum 映射为"动量"。
    "cne5s_size": "市值",  # 把 cne5s_size 映射为"市值"。
    "cne5s_sizenl": "非线性市值",  # 把 cne5s_sizenl 映射为"非线性市值"。
    REVERSAL_COLUMN: "20日反转",  # 为该字段配置报告显示名称。
}
CORRELATION_COLUMNS = (*STYLE_COLUMNS, REVERSAL_COLUMN)  # 集中声明相关系数列。
REGRESSION_COLUMNS = CORRELATION_COLUMNS  # 集中声明regression列。
VIVIT_MATRIX_COLUMN = "vivit_factor"  # 设置vivit矩阵列。
MATRIX_COLUMNS = (VIVIT_MATRIX_COLUMN, *CORRELATION_COLUMNS)  # 集中声明矩阵列。
MATRIX_LABELS = {  # 设置矩阵标签。
    VIVIT_MATRIX_COLUMN: "ViViT因子",  # 为该字段配置报告显示名称。
    **STYLE_LABELS,
}
REQUIRED_CNE5S_COLUMNS = (  # 集中声明必需cne5s列。
    "datadate",
    "assetid",
    INDUSTRY_COLUMN,
    *STYLE_COLUMNS,
)
INSTRUMENT_PATTERN = re.compile(r"^(SH|SZ|BJ)\.(\d{6})$")  # 编译股票pattern匹配规则。
CNE5S_ASSET_PATTERN = re.compile(r"^CN\d{6}$")  # 编译cne5sassetpattern匹配规则。


def _fingerprint_parquet(path: Path) -> dict[str, Any]:  # 提取文件身份信息，用于来源审计和复用判断。
    """提取文件身份信息，用于来源审计和复用判断。"""
    if not path.exists():  # 检查目标文件或目录是否满足存在性要求。
        raise FileNotFoundError(path)  # 目标文件不存在时立即终止。
    parquet = pq.ParquetFile(path)  # 读取 Parquet 元数据。
    return {  # 返回提取文件身份信息，用于来源审计和复用判断。
        "path": str(path.resolve()),  # 记录路径。
        "size": int(path.stat().st_size),  # 记录大小。
        "mtime_ns": int(path.stat().st_mtime_ns),  # 记录修改时间纳秒。
        "rows": int(parquet.metadata.num_rows),  # 记录行数。
        "row_groups": int(parquet.num_row_groups),  # 记录行分组。
        "columns": list(parquet.schema_arrow.names),  # 记录字段。
    }


def instrument_ids_to_assetids(instrument_ids: pd.Series) -> pd.Series:  # 把 SH.600000/SZ.000001 等项目代码转换为 Barra CN600000/CN000001。
    """把 SH.600000/SZ.000001 等项目代码转换为 Barra CN600000/CN000001。"""

    values = instrument_ids.astype(str)  # 转换值的数据类型或存储位置。
    valid = values.str.fullmatch(INSTRUMENT_PATTERN)  # 校验证券代码格式。
    if not bool(valid.all()):  # 检查当前掩码或数据是否满足有效性要求。
        examples = values.loc[~valid].drop_duplicates().head(10).tolist()
        raise ValueError(f"instrument_id 格式非法：{examples}")  # 输入数据不满足业务或数值约束时立即终止。
    return ("CN" + values.str[-6:]).rename("assetid")  # 返回完成上述格式转换后的结果。


def _normalize_date_strings(values: pd.Series) -> pd.Series:  # 把输入日期统一规范为八位 YYYYMMDD 字符串。
    """把输入日期统一规范为八位 YYYYMMDD 字符串。"""
    if pd.api.types.is_datetime64_any_dtype(values):  # 根据当前数据状态选择处理分支。
        parsed = pd.to_datetime(values, errors="raise")  # 将输入统一解析为交易日。
    else:  # 处理上述条件均不成立的情况。
        normalized = values.astype(str).str.replace(r"\.0$", "", regex=True)  # 按规则替换或转义字符串内容。
        parsed = pd.to_datetime(normalized, format="%Y%m%d", errors="raise")  # 将输入统一解析为交易日。
    return parsed.dt.strftime("%Y%m%d")  # 返回完成上述格式转换后的结果。


def _load_sse_calendar(calendar_path: Path) -> pd.DatetimeIndex:  # 加载SSE交易日历。
    """加载该流程所需数据，并检查形状、业务键和来源一致性。"""
    calendar = pd.read_parquet(  # 读取交易日历。
        calendar_path,
        columns=["trade_days", "s_info_exchmarket"],  # 限定本次读取或输出使用的字段。
        filters=[("s_info_exchmarket", "==", "SSE")],  # 下推数据筛选条件，减少无关数据读取。
    )
    dates = pd.DatetimeIndex(  # 构造有序交易日索引。
        pd.to_datetime(
            _normalize_date_strings(calendar["trade_days"]),
            format="%Y%m%d",  # 指定输入日期或文本格式。
        ).sort_values().unique(),
        name="trade_date",  # 设置结果对象名称。
    )
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:  # 检查结果表是否为空。
        raise ValueError("SSE 交易日历为空、重复或未排序")  # 输入数据不满足业务或数值约束时立即终止。
    return dates  # 返回交易日。


def _instrument_ids_to_windcodes(instrument_ids: pd.Series) -> pd.Series:  # 批量把交易所前缀股票代码转换为 Wind 代码。
    """批量把交易所前缀股票代码转换为 Wind 代码。"""
    values = instrument_ids.astype(str)  # 转换值的数据类型或存储位置。
    parts = values.str.extract(  # 拆分证券代码字段。
        r"^(?P<exchange>SH|SZ|BJ)\.(?P<number>\d{6})$"
    )
    if parts.isna().any(axis=None):  # 根据当前数据状态选择处理分支。
        examples = values.loc[parts.isna().any(axis=1)].head(10).tolist()
        raise ValueError(f"instrument_id 格式非法：{examples}")  # 输入数据不满足业务或数值约束时立即终止。
    return (parts["number"] + "." + parts["exchange"]).rename("windcode")  # 返回批量把交易所前缀股票代码转换为 Wind 代码。


def _attach_reversal_20d(  # 按 SSE 精确 T-20 交易日计算负累计复权收益，不做填充。
    prediction: pd.DataFrame,  # 接收预测。
    eod_path: Path,  # 接收日行情路径。
    calendar_dates: pd.DatetimeIndex,  # 接收交易日历交易日。
    year: int,  # 接收年份。
) -> pd.DataFrame:
    """按 SSE 精确 T-20 交易日计算负累计复权收益，不做填充。"""

    result = prediction.copy()  # 转换结果的数据类型或存储位置。
    signal_dates = pd.to_datetime(  # 解析并统一signal交易日。
        result["trade_date"], format="%Y%m%d", errors="raise"
    )
    positions = calendar_dates.get_indexer(signal_dates)  # 反转窗口以 SSE 交易日序号而非自然日计算。
    if (positions < 0).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError(f"{year} 预测日期无法映射到 SSE 交易日历")  # 输入数据不满足业务或数值约束时立即终止。
    if (positions < REVERSAL_LOOKBACK_DAYS).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError(f"{year} 交易日历不足 T-{REVERSAL_LOOKBACK_DAYS}")  # 输入数据不满足业务或数值约束时立即终止。
    lag_dates = calendar_dates[positions - REVERSAL_LOOKBACK_DAYS]  # 精确取 T-20，不使用最近可得价格替代。
    if not bool((lag_dates < signal_dates.to_numpy()).all()):  # 检查当前掩码或数据是否满足有效性要求。
        raise AssertionError("20 日反转错误使用了当日或未来价格")  # 关键数据契约被破坏时立即终止。

    result["_source_order"] = np.arange(len(result), dtype=np.int64)  # 合并后用该列恢复原预测顺序。
    result["_windcode"] = _instrument_ids_to_windcodes(
        result["instrument_id"]
    ).to_numpy()
    result["_lag_trade_date"] = lag_dates.strftime("%Y%m%d")  # 统一为 Wind 日期格式。
    first_date = str(result["_lag_trade_date"].min())
    last_date = str(result["trade_date"].max())
    eod = pd.read_parquet(  # 读取日行情。
        eod_path,
        columns=["s_info_windcode", "trade_dt", "s_dq_adjclose"],  # 限定本次读取或输出使用的字段。
        filters=[  # 初始化filters容器。
            ("trade_dt", ">=", first_date),
            ("trade_dt", "<=", last_date),
        ],
    )
    eod["trade_date"] = _normalize_date_strings(eod["trade_dt"])  # 统一 Wind 日期为 YYYYMMDD。
    eod["_windcode"] = eod["s_info_windcode"].astype(str)  # 转换日行情的数据类型或存储位置。
    eod = eod.loc[
        eod["_windcode"].isin(set(result["_windcode"])),
        ["trade_date", "_windcode", "s_dq_adjclose"],
    ].copy()
    if eod.duplicated(["trade_date", "_windcode"]).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError(f"{year} Wind 复权价业务键重复")  # 输入数据不满足业务或数值约束时立即终止。
    price = eod["s_dq_adjclose"].to_numpy(np.float64)  # 转换并统一price的数据表示。
    eod.loc[~(np.isfinite(price) & (price > 0)), "s_dq_adjclose"] = np.nan

    current = eod.rename(columns={"s_dq_adjclose": "_price_t"})
    lagged = eod.rename(
        columns={  # 初始化列容器。
            "trade_date": "_lag_trade_date",  # 把 trade_date 映射为"_lag_trade_date"。
            "s_dq_adjclose": "_price_t_minus_20",  # 把 s_dq_adjclose 映射为"_price_t_minus_20"。
        }
    )
    result = result.merge(  # 按股票与交易日业务键对齐。
        current,
        on=["trade_date", "_windcode"],  # 指定数据连接使用的业务键。
        how="left",  # 指定数据连接方式。
        validate="many_to_one",  # 校验数据连接的基数关系。
        sort=False,  # 保持源数据顺序，不额外排序。
    ).merge(
        lagged,
        on=["_lag_trade_date", "_windcode"],  # 指定数据连接使用的业务键。
        how="left",  # 指定数据连接方式。
        validate="many_to_one",  # 校验数据连接的基数关系。
        sort=False,  # 保持源数据顺序，不额外排序。
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
    if not np.isfinite(reversal[valid.to_numpy()]).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError(f"{year} 20 日反转含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。
    result[REVERSAL_COLUMN] = reversal
    result[REVERSAL_VALID_COLUMN] = valid.to_numpy(bool)  # 转换并统一结果的数据表示。
    return result.drop(  # 返回按当前规则组织后的结果。
        columns=[  # 初始化列容器。
            "_source_order", "_windcode", "_lag_trade_date",
            "_price_t", "_price_t_minus_20",
        ]
    )


def _standardized_reversal(daily: pd.DataFrame) -> np.ndarray:  # 执行横截面标准化，并显式处理缺失值和常数截面。
    """执行横截面标准化，并显式处理缺失值和常数截面。"""
    valid = daily[REVERSAL_VALID_COLUMN].to_numpy(bool)  # 转换并统一有效的数据表示。
    raw = daily[REVERSAL_COLUMN].to_numpy(np.float64)  # 转换并统一原始的数据表示。
    if not np.array_equal(valid, np.isfinite(raw)):  # 检查参与计算的数值是否全部有限。
        raise ValueError("20 日反转有效标记与数值缺失不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if int(valid.sum()) < 3:  # 检查当前数值是否越过业务或数值阈值。
        raise ValueError("单日有效 20 日反转不足 3 个观测")  # 输入数据不满足业务或数值约束时立即终止。
    mean = float(raw[valid].mean())  # 计算均值。
    std = float(raw[valid].std(ddof=0))  # 计算标准差。
    if not np.isfinite(std) or std <= np.finfo(np.float64).eps:  # 检查参与计算的数值是否全部有限。
        raise ValueError("单日 20 日反转为常数")  # 输入数据不满足业务或数值约束时立即终止。
    standardized = np.zeros(len(daily), dtype=np.float64)  # 缺失反转暴露填零并保留独立 valid 标记。
    standardized[valid] = (raw[valid] - mean) / std  # 构造standardized，限定后续参与计算的样本。
    return standardized  # 返回standardized。


def _load_cne5s_year(cne5s_path: Path, year: int) -> pd.DataFrame:  # 加载cne5s年份。
    """加载该流程所需数据，并检查形状、业务键和来源一致性。"""
    dataset = ds.dataset(cne5s_path, format="parquet")  # 打开分区 Parquet 数据集。
    table = dataset.to_table(  # 下推日期过滤，仅读取目标分区。
        columns=list(REQUIRED_CNE5S_COLUMNS),  # 限定本次读取或输出使用的字段。
        filter=(
            (ds.field("datadate") >= pd.Timestamp(f"{year}-01-01"))
            & (ds.field("datadate") < pd.Timestamp(f"{year + 1}-01-01"))
        ),
    )
    frame = table.to_pandas()
    if frame.empty:  # 检查结果表是否为空。
        raise ValueError(f"CNE5S 没有 {year} 数据")  # 输入数据不满足业务或数值约束时立即终止。
    frame["trade_date"] = pd.to_datetime(frame["datadate"]).dt.strftime(  # 统一为 Wind 日期格式。
        "%Y%m%d"
    )
    frame["assetid"] = frame["assetid"].astype(str)  # 转换帧的数据类型或存储位置。
    valid_assets = frame["assetid"].str.fullmatch(CNE5S_ASSET_PATTERN)  # 校验证券代码格式。
    if not bool(valid_assets.all()):  # 检查当前掩码或数据是否满足有效性要求。
        examples = frame.loc[~valid_assets, "assetid"].drop_duplicates()  # 业务键去重。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"{year} CNE5S assetid 格式非法：{examples.head(10).tolist()}"
        )
    if frame.duplicated(["trade_date", "assetid"]).any():  # 检查当前掩码或数据是否满足有效性要求。
        duplicates = (
            frame.loc[
                frame.duplicated(["trade_date", "assetid"], keep=False),
                ["trade_date", "assetid"],
            ]
            .drop_duplicates()
            .head(10)
            .to_dict("records")
        )
        raise ValueError(f"{year} CNE5S 业务键重复：{duplicates}")  # 输入数据不满足业务或数值约束时立即终止。
    style_values = frame.loc[:, STYLE_COLUMNS].to_numpy(np.float64)  # 转换并统一风格值的数据表示。
    if not np.isfinite(style_values).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError(f"{year} CNE5S 风格暴露含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。
    if frame[INDUSTRY_COLUMN].isna().any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError(f"{year} CNE5S 行业字段含缺失")  # 输入数据不满足业务或数值约束时立即终止。
    return frame[  # 返回加载并校验后的结果。
        ["trade_date", "assetid", INDUSTRY_COLUMN, *STYLE_COLUMNS]
    ].copy()


def _join_prediction_and_cne5s(  # 处理预测文件、预测列或对应的样本数量审计。
    prediction: pd.DataFrame, barra: pd.DataFrame, year: int
) -> pd.DataFrame:
    """处理预测文件、预测列或对应的样本数量审计。"""
    source = prediction.copy()  # 转换来源的数据类型或存储位置。
    source["_source_order"] = np.arange(len(source), dtype=np.int64)  # 对齐 Barra 后恢复因子原始行序。
    joined = source.merge(  # 按业务键对齐，避免依赖原始行序。
        barra,
        on=["trade_date", "assetid"],  # 指定数据连接使用的业务键。
        how="left",  # 指定数据连接方式。
        validate="one_to_one",  # 校验数据连接的基数关系。
        sort=False,  # 保持源数据顺序，不额外排序。
        indicator=True,  # 保留连接来源标记以审计匹配率。
    )
    joined.sort_values("_source_order", inplace=True)
    joined.reset_index(drop=True, inplace=True)  # 按股票与交易日业务键对齐。
    if not np.array_equal(  # 检查必需数据、配置或校验结果是否缺失。
        joined["_source_order"].to_numpy(),
        np.arange(len(source), dtype=np.int64),
    ):
        raise AssertionError(f"{year} 合并后源预测顺序改变")  # 关键数据契约被破坏时立即终止。
    joined["barra_matched"] = joined["_merge"].eq("both")  # 按股票与交易日业务键对齐。
    return joined  # 返回joined。


def _match_record(  # 统计或检查数据连接覆盖率。
    joined: pd.DataFrame, year: int, cne5s_rows: int
) -> dict[str, Any]:
    """统计或检查数据连接覆盖率。"""
    matched = int(joined["barra_matched"].sum())  # 计算匹配数量。
    reversal_valid = int(joined[REVERSAL_VALID_COLUMN].sum())  # 计算reversal有效。
    correlation_valid = int(  # 构造相关系数有效，限定后续参与计算的样本。
        (joined["barra_matched"] & joined[REVERSAL_VALID_COLUMN]).sum()
    )
    rows = len(joined)
    return {  # 返回统计或检查数据连接覆盖率。
        "year": year,  # 记录年份。
        "prediction_rows": rows,  # 记录预测行数。
        "cne5s_rows": int(cne5s_rows),  # 记录cne5s行数。
        "matched_rows": matched,  # 记录匹配行数。
        "unmatched_rows": rows - matched,  # 记录未匹配行数。
        "match_rate": float(matched / rows) if rows else np.nan,  # 记录match比例。
        "reversal_valid_rows": reversal_valid,  # 记录反转有效行数。
        "reversal_missing_rows": rows - reversal_valid,  # 记录反转缺失值行数。
        "reversal_coverage": (  # 记录反转覆盖率。
            float(reversal_valid / rows) if rows else np.nan
        ),
        "reversal_correlation_rows": correlation_valid,  # 记录反转相关系数行数。
        "dates": int(joined["trade_date"].nunique()),  # 记录交易日。
        "first_date": str(joined["trade_date"].min()) if rows else None,  # 记录首个交易日。
        "last_date": str(joined["trade_date"].max()) if rows else None,  # 记录最后一个交易日。
        "status": "passed",  # 把 status 映射为"passed"。
    }


def _validate_cne5s_schema(cne5s_path: Path) -> dict[str, Any]:  # 校验输入形状、业务键、有限性和签名约束。
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    fingerprint = _fingerprint_parquet(cne5s_path)  # 记录源文件指纹。
    missing = [  # 初始化missing容器。
        column
        for column in REQUIRED_CNE5S_COLUMNS  # 遍历必需cne5s列，逐项处理列。
        if column not in fingerprint["columns"]  # 检查当前值是否属于允许集合。
    ]
    if missing:  # 根据当前数据状态选择处理分支。
        raise ValueError(f"CNE5S 缺少字段：{missing}")  # 输入数据不满足业务或数值约束时立即终止。
    return fingerprint  # 返回文件指纹。


def _validate_price_sources(  # 校验输入形状、业务键、有限性和签名约束。
    eod_path: Path, calendar_path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    eod_fingerprint = _fingerprint_parquet(eod_path)  # 记录源文件指纹。
    calendar_fingerprint = _fingerprint_parquet(calendar_path)  # 记录源文件指纹。
    required_eod = {"s_info_windcode", "trade_dt", "s_dq_adjclose"}  # 复权价最小字段契约。
    required_calendar = {"trade_days", "s_info_exchmarket"}  # 交易日历最小字段契约。
    missing_eod = required_eod.difference(eod_fingerprint["columns"])  # 检查输入缺失字段。
    missing_calendar = required_calendar.difference(  # 检查输入缺失字段。
        calendar_fingerprint["columns"]
    )
    if missing_eod:  # 根据当前数据状态选择处理分支。
        raise ValueError(f"Wind 复权价缺少字段：{sorted(missing_eod)}")  # 输入数据不满足业务或数值约束时立即终止。
    if missing_calendar:  # 根据当前数据状态选择处理分支。
        raise ValueError(f"Wind 交易日历缺少字段：{sorted(missing_calendar)}")  # 输入数据不满足业务或数值约束时立即终止。
    return eod_fingerprint, calendar_fingerprint  # 返回校验输入形状、业务键、有限性和签名约束。


def _check_match_rate(  # 统计或检查数据连接覆盖率。
    record: dict[str, Any], minimum_match_rate: float
) -> None:
    """统计或检查数据连接覆盖率。"""
    if float(record["match_rate"]) < minimum_match_rate:  # 检查当前数值是否越过业务或数值阈值。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            f"{record['year']} CNE5S 匹配率 "
            f"{record['match_rate']:.6%} < {minimum_match_rate:.6%}"
        )
