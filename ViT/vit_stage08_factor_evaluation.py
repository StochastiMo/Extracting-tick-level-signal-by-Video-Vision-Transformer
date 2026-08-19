"""阶段 08：执行 ViT 样本外因子评价并生成实验报告。"""
from __future__ import annotations

from typing import Any

import argparse
import gc
import os
import time
import warnings
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
from IPython.display import display

from vit_stage01_foundation import (
    CALENDAR_PATH, EOD_PATH, VIT_DIR, atomic_json, atomic_parquet,
    get_data_root, normalize_dates, read_json,
)


DEFAULT_A2_RUN_NAME = "date_batch"
BACKTEST_MODES = (
    "mode1_t_with_t1_filter",
    "mode2_t_no_filter",
    "mode3_t1_no_filter",
)
DEFAULT_BACKTEST_MODE = "mode2_t_no_filter"
EQUAL_WEIGHT_MARKET_COLUMN = "Equal-weight market"
RELATIVE_PORTFOLIOS = ("Q5 long-only", "Q5-Q1 long-short")
ANNUALIZATION_DAYS = 252
BACKTEST_MODE_CONFIGS = {
    "mode1_t_with_t1_filter": {
        "price_start_offset": 0,
        "apply_can_enter_t1": True,
        "price_start": "T_no_shift",
        "portfolio_scope": "preassigned_quantile_then_can_enter_t1",
        "execution_mask": "can_enter_t1",
    },
    "mode2_t_no_filter": {
        "price_start_offset": 0,
        "apply_can_enter_t1": False,
        "price_start": "T_no_shift",
        "portfolio_scope": "full_t_cross_section_no_t1_filter",
        "execution_mask": "none",
    },
    "mode3_t1_no_filter": {
        "price_start_offset": 1,
        "apply_can_enter_t1": False,
        "price_start": "T_plus_1",
        "portfolio_scope": "full_t_cross_section_no_t1_filter",
        "execution_mask": "none",
    },
}


def validate_run_name(value: str) -> str:
    """校验 run 名称，避免路径穿越和非法目录字符。"""
    run_name = str(value).strip()
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    if (
        not run_name
        or run_name in {".", ".."}
        or any(character not in allowed for character in run_name)
    ):
        raise ValueError("运行名含非法字符")
    return run_name


def resolve_backtest_mode(backtest_mode: str) -> dict[str, object]:
    """校验回测模式，并返回配置副本。"""

    mode = str(backtest_mode).strip()
    if mode not in BACKTEST_MODE_CONFIGS:
        raise ValueError(
            f"未知回测模式：{backtest_mode!r}；可选 {BACKTEST_MODES}"
        )
    return {"name": mode, **BACKTEST_MODE_CONFIGS[mode]}


def price_point_label(offset: int) -> str:
    """返回回测价格时点的可读标签。"""
    return "T" if offset == 0 else f"T+{offset}"


def forward_return_formula(price_start_offset: int) -> str:
    """返回指定持有期对应的远期收益公式说明。"""
    if price_start_offset == 0:
        return "adj_close(T+k) / adj_close(T) - 1"
    return "adj_close(T+1+k) / adj_close(T+1) - 1"


def geometric_annual_return(
    returns: pd.Series | np.ndarray, annualization: int = ANNUALIZATION_DAYS
) -> float:
    """以日收益复利计算几何年化收益。"""

    values = np.asarray(returns, dtype=np.float64)
    if len(values) == 0 or not np.isfinite(values).all():
        raise ValueError("年化收益输入必须是非空有限日收益")
    if (values <= -1).any():
        raise ValueError("日收益必须大于 -100%")
    growth = float(np.prod(1.0 + values))
    if not np.isfinite(growth) or growth <= 0:
        raise ValueError("累计净值必须为正且有限")
    return growth ** (annualization / len(values)) - 1.0


def compute_relative_performance(
    portfolio_returns: pd.DataFrame,
    benchmark_column: str = EQUAL_WEIGHT_MARKET_COLUMN,
) -> pd.DataFrame:
    """计算 Q5 与 Q5-Q1 相对全市场等权基准的绩效。"""

    required = (*RELATIVE_PORTFOLIOS, benchmark_column)
    missing = set(required).difference(portfolio_returns.columns)  # 检查输入缺失字段
    if missing:
        raise KeyError(f"相对绩效缺少收益列：{sorted(missing)}")
    values = portfolio_returns[list(required)].to_numpy(np.float64)
    if len(portfolio_returns) < 2 or not np.isfinite(values).all():
        raise ValueError("相对绩效至少需要两日有限收益")

    benchmark = portfolio_returns[benchmark_column].to_numpy(np.float64)
    benchmark_annual_return = geometric_annual_return(benchmark)  # 由日收益复利年化
    records = []
    for portfolio in RELATIVE_PORTFOLIOS:
        returns = portfolio_returns[portfolio].to_numpy(np.float64)
        active = returns - benchmark
        active_scale = max(1.0, float(np.max(np.abs(active))))
        if float(np.ptp(active)) <= np.finfo(np.float64).eps * active_scale * 16:
            raise ValueError(f"{portfolio} 主动收益方差为零，信息比无定义")
        active_std = float(active.std(ddof=1))
        if not np.isfinite(active_std) or active_std <= 0:
            raise ValueError(f"{portfolio} 主动收益方差为零，信息比无定义")
        portfolio_annual_return = geometric_annual_return(returns)  # 由日收益复利年化
        records.append(
            {
                "portfolio": portfolio,
                "benchmark": benchmark_column,
                "observations": len(active),
                "portfolio_annual_return": portfolio_annual_return,
                "benchmark_annual_return": benchmark_annual_return,
                "annual_excess_return": (
                    portfolio_annual_return - benchmark_annual_return
                ),
                "annual_tracking_error": (
                    active_std * np.sqrt(ANNUALIZATION_DAYS)
                ),
                "information_ratio": (
                    float(active.mean()) / active_std
                    * np.sqrt(ANNUALIZATION_DAYS)
                ),
            }
        )
    return pd.DataFrame(records)

def build_return_price_view(
    prices: pd.DataFrame, backtest_mode: str = DEFAULT_BACKTEST_MODE
) -> pd.DataFrame:
    """让 Alphalens 在信号日 T 索引上按所选起点计算远期收益。"""

    offset = int(resolve_backtest_mode(backtest_mode)["price_start_offset"])
    return prices if offset == 0 else prices.shift(-offset)


def select_execution_factor_data(
    factor_data: pd.DataFrame,
    can_enter_t1: pd.Series | np.ndarray,
    backtest_mode: str = DEFAULT_BACKTEST_MODE,
) -> pd.DataFrame:
    """仅口径 1 在已固定分位后应用 T+1 可进入掩码。"""

    mode = resolve_backtest_mode(backtest_mode)
    mask = np.asarray(can_enter_t1, dtype=bool)
    if len(mask) != len(factor_data):
        raise ValueError("can_enter_t1 与因子数据长度不一致")
    if bool(mode["apply_can_enter_t1"]):
        return factor_data.iloc[np.flatnonzero(mask)].copy()
    return factor_data


def compute_equal_weight_market_returns(
    factor_data: pd.DataFrame, return_column: str = "1D"
) -> pd.Series:
    """在完整、价格有效的 T 日因子覆盖池内计算等权市场收益。"""

    if return_column not in factor_data.columns:
        raise KeyError(f"缺少基准收益列：{return_column}")
    values = factor_data[return_column].to_numpy(np.float64)
    if not np.isfinite(values).all():
        raise ValueError("全市场等权基准输入含 NaN 或 Inf")
    return (
        factor_data[return_column]
        .groupby(level="date", sort=True)
        .mean()
        .rename(EQUAL_WEIGHT_MARKET_COLUMN)
    )


# ====================== 04｜Alphalens 样本外评价 ======================
def run_oos_backtest(
    run_name: str = DEFAULT_A2_RUN_NAME,
    data_root: str | Path | None = None,
    backtest_mode: str = DEFAULT_BACKTEST_MODE,
) -> pd.DataFrame:
    """按指定价格起点与执行过滤口径计算样本外收益。"""

    run_name = validate_run_name(run_name)
    mode = resolve_backtest_mode(backtest_mode)
    price_start_offset = int(mode["price_start_offset"])
    warnings.filterwarnings(
        "ignore", message=r'Module "zipline\.assets" not found.*'
    )
    try:
        import alphalens as al
        import matplotlib.pyplot as plt
        import pyfolio as pf
        from matplotlib.ticker import PercentFormatter
        from scipy import stats
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "请使用 Jupyter 内核 `ViT Factor Eval (Alphalens/Pyfolio)`。"
        ) from error

    # factor-eval 环境已安装该字体；显式置顶，避免 Matplotlib 默认
    # DejaVu Sans 缺少中文字符而在 Notebook 中产生方框和大量 warning。
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    factor_column = "prediction_ensemble"
    periods = (1, 5, 10)
    quantiles = 5
    max_loss = 0.001
    price_column = "s_dq_adjclose"
    root = (
        Path(data_root).resolve()
        if data_root is not None
        else get_data_root().resolve()
    )
    if not root.is_dir():
        raise FileNotFoundError(root)
    prediction_dir = root / "predictions" / "runs" / run_name
    universe_dir = root / "universe"
    quality_dir = root / "quality" / "runs" / run_name
    evaluation_dir = quality_dir / "factor_evaluation"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    training_manifest = read_json(quality_dir / "a2_training_manifest.json")
    universe_manifest = read_json(universe_dir / "universe_manifest.json")
    dataset_manifest = read_json(root / "feature_store" / "dataset_manifest.json")
    if training_manifest is None or universe_manifest is None or dataset_manifest is None:
        raise FileNotFoundError("训练、Universe 或 feature store manifest 不完整")
    if training_manifest["universe_signature"] != universe_manifest["signature"]:
        raise AssertionError("训练使用的 Universe 与当前 Universe 签名不一致")
    if training_manifest["dataset_signature"] != dataset_manifest["signature"]:
        raise AssertionError("训练使用的 feature store 与当前数据签名不一致")

    prediction_years = tuple(
        year
        for year in training_manifest["years"]
        if (prediction_dir / f"vit_predictions.{year}.parquet").exists()
    )
    if not prediction_years:
        raise FileNotFoundError("没有可评价的年度预测")
    prediction_parts: list[pd.DataFrame] = []
    audit_records: list[dict[str, object]] = []
    for year in prediction_years:
        prediction_path = prediction_dir / f"vit_predictions.{year}.parquet"
        prediction = pd.read_parquet(
            prediction_path,
            columns=["trade_date", "instrument_id", factor_column],
        )
        prediction["trade_date"] = prediction["trade_date"].astype(str)
        prediction["instrument_id"] = prediction["instrument_id"].astype(str)
        if prediction.duplicated(["trade_date", "instrument_id"]).any():
            raise ValueError(f"{year} prediction 业务键重复")
        if not prediction["trade_date"].str.startswith(str(year)).all():
            raise ValueError(f"{year} prediction 含跨年度日期")
        if not np.isfinite(prediction[factor_column].to_numpy(np.float64)).all():
            raise ValueError(f"{year} 因子含 NaN 或 Inf")

        universe = pd.read_parquet(
            universe_dir / f"universe.{year}.parquet",
            columns=[
                "trade_date", "instrument_id", "feature_valid",
                "in_universe_t", "can_enter_t1",
            ],
        )
        expected = universe.loc[
            universe["feature_valid"] & universe["in_universe_t"],
            ["trade_date", "instrument_id", "can_enter_t1"],
        ].copy()
        expected["trade_date"] = expected["trade_date"].astype(str)
        expected["instrument_id"] = expected["instrument_id"].astype(str)
        joined = prediction.merge(  # 按业务键对齐，避免依赖原始行序
            expected,
            on=["trade_date", "instrument_id"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not joined["_merge"].eq("both").all():
            raise AssertionError(f"{year} prediction 与 T 日 Universe 键不一致")
        joined = joined.drop(columns="_merge")  # 按股票与交易日业务键对齐
        if joined["can_enter_t1"].isna().any():
            raise AssertionError(f"{year} can_enter_t1 含缺失值")
        joined["can_enter_t1"] = joined["can_enter_t1"].astype(bool)
        can_enter_t1_rows = int(joined["can_enter_t1"].sum())
        audit_records.append(
            {
                "year": year,
                "prediction_rows": len(joined),
                "can_enter_t1_rows": can_enter_t1_rows,
                "t1_excluded_rows": len(joined) - can_enter_t1_rows,
                "dates": joined["trade_date"].nunique(),
                "first_date": joined["trade_date"].min(),
                "last_date": joined["trade_date"].max(),
            }
        )
        prediction_parts.append(joined)

    prediction_audit = pd.DataFrame(audit_records)
    panel = pd.concat(prediction_parts, ignore_index=True)
    del prediction_parts, prediction, universe, expected, joined
    if panel.duplicated(["trade_date", "instrument_id"]).any():
        raise ValueError("跨年度因子业务键重复")
    t_grouping_rows_before_price_filter = len(panel)
    panel["date"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d")
    factor_panel = (  # 固定日期与股票排序
        panel.set_index(["date", "instrument_id"])[
            [factor_column, "can_enter_t1"]
        ]
        .sort_index()
    )
    factor_panel.index = factor_panel.index.set_names(["date", "asset"])
    factor = factor_panel[factor_column].rename("factor")
    can_enter_t1 = factor_panel["can_enter_t1"].astype(bool)
    del panel, factor_panel

    # 分位组只能使用 T 日已经可知的因子截面。先对完整 T 日预测池分组，
    # 后续价格缺失与 T+1 可进入状态都只能删行，不能重新改变组别边界。
    groups = factor.groupby(level="date", sort=False)
    ordinal_rank = groups.rank(method="first", ascending=True)
    group_size = groups.transform("size")
    factor_quantile = pd.Series(
        np.floor(
            (ordinal_rank.to_numpy(np.float64) - 1) * quantiles
            / group_size.to_numpy(np.float64)
        ).astype(np.int64)
        + 1,
        index=factor.index,
        name="factor_quantile",
    )
    t_grouping_quantile_counts = (
        factor_quantile.groupby(
            [pd.Grouper(level="date"), factor_quantile], observed=True
        ).size().unstack("factor_quantile")
    )
    if t_grouping_quantile_counts.isna().any(axis=None) or (
        t_grouping_quantile_counts.max(axis=1)
        - t_grouping_quantile_counts.min(axis=1)
    ).gt(1).any():
        raise AssertionError("T 日完整截面五分位分组不均衡")

    calendar = pd.read_parquet(
        CALENDAR_PATH,
        columns=["trade_days", "s_info_exchmarket"],
        filters=[("s_info_exchmarket", "==", "SSE")],
    )
    calendar_dates = pd.DatetimeIndex(  # 构造有序交易日索引
        pd.to_datetime(normalize_dates(calendar["trade_days"]), format="%Y%m%d")
        .sort_values().unique(),
        name="date",
    )
    factor_dates = pd.DatetimeIndex(  # 构造有序交易日索引
        factor.index.get_level_values("date").unique().sort_values()
    )
    positions = pd.Series(np.arange(len(calendar_dates)), index=calendar_dates)
    factor_positions = factor_dates.map(positions)
    if factor_positions.isna().any():
        raise AssertionError("因子日期无法映射到交易日历")
    first_price_date = factor_dates.min()
    final_price_position = (
        int(factor_positions[-1]) + price_start_offset + max(periods)
    )
    if final_price_position >= len(calendar_dates):
        raise ValueError("交易日历不足以覆盖最大收益期限")
    last_price_date = calendar_dates[final_price_position]
    price_calendar = calendar_dates[
        calendar_dates.get_loc(first_price_date) : final_price_position + 1
    ]
    assets = pd.Index(
        sorted(factor.index.get_level_values("asset").unique()), name="asset"
    )
    wind_codes = pd.Series(assets).str.extract(  # 拆分证券代码字段
        r"^(?P<exchange>SH|SZ)\.(?P<number>\d{6})$"
    )
    if wind_codes.isna().any(axis=None):
        raise ValueError("因子股票代码格式异常")
    needed_wind_codes = set(wind_codes["number"] + "." + wind_codes["exchange"])
    eod = pd.read_parquet(
        EOD_PATH,
        columns=["s_info_windcode", "trade_dt", price_column],
        filters=[
            ("trade_dt", ">=", first_price_date.strftime("%Y%m%d")),
            ("trade_dt", "<=", last_price_date.strftime("%Y%m%d")),
        ],
    )
    eod["trade_dt"] = normalize_dates(eod["trade_dt"])  # 仅使用当日截面统计量
    eod["s_info_windcode"] = eod["s_info_windcode"].astype(str)
    eod = eod.loc[eod["s_info_windcode"].isin(needed_wind_codes)].copy()
    parts = eod["s_info_windcode"].str.extract(  # 拆分证券代码字段
        r"^(?P<number>\d{6})\.(?P<exchange>SH|SZ)$"
    )
    eod["asset"] = parts["exchange"] + "." + parts["number"]
    eod["date"] = pd.to_datetime(eod["trade_dt"], format="%Y%m%d")
    if eod.duplicated(["date", "asset"]).any():
        raise ValueError("Wind 复权收盘价业务键重复")
    valid_price = (
        np.isfinite(eod[price_column].to_numpy(np.float64))
        & eod[price_column].gt(0).to_numpy()
    )
    eod.loc[~valid_price, price_column] = np.nan
    prices = (
        eod.pivot(index="date", columns="asset", values=price_column)
        .reindex(index=price_calendar, columns=assets)
        .astype(np.float64)
    )
    prices.index.name, prices.columns.name = "date", "asset"

    factor_keys = factor.index.to_frame(index=False)
    start_rows = prices.index.get_indexer(factor_keys["date"])
    asset_columns = prices.columns.get_indexer(factor_keys["asset"])
    if (start_rows < 0).any() or (asset_columns < 0).any():
        raise AssertionError("因子键无法映射到 T 日价格矩阵")
    price_values = prices.to_numpy(copy=False)
    all_prices = np.ones(len(factor), dtype=bool)
    coverage_records = []
    required_offsets = (
        price_start_offset,
        *(price_start_offset + period for period in periods),
    )
    for offset in required_offsets:
        selected = price_values[start_rows + offset, asset_columns]
        available = np.isfinite(selected) & (selected > 0)
        all_prices &= available
        coverage_records.append(
            {
                "price_point": price_point_label(offset),
                "available_rows": int(available.sum()),
                "missing_rows": int((~available).sum()),
                "coverage": float(available.mean()),
            }
        )
    price_coverage = pd.DataFrame(coverage_records)
    if 1 - float(all_prices.mean()) > max_loss:
        raise AssertionError("复权收盘价缺失比例超过 MAX_LOSS=0.001")
    valid_price_positions = np.flatnonzero(all_prices)
    factor = factor.iloc[valid_price_positions].copy()
    factor_quantile = factor_quantile.iloc[valid_price_positions].copy()
    can_enter_t1 = can_enter_t1.iloc[valid_price_positions].copy()
    t_grouping_rows_after_price_filter = len(factor)

    # 保留完整交易日 level，明确 period 是交易日偏移且起点就是 T 日。
    analysis_calendar = calendar_dates[
        (calendar_dates >= factor_dates.min()) & (calendar_dates <= factor_dates.max())
    ]
    factor_date_codes = analysis_calendar.get_indexer(
        factor.index.get_level_values("date")
    )
    factor_asset_codes = assets.get_indexer(factor.index.get_level_values("asset"))
    factor_index = pd.MultiIndex(
        levels=[analysis_calendar, assets],
        codes=[factor_date_codes, factor_asset_codes],
        names=["date", "asset"],
        verify_integrity=True,
    )
    factor.index = factor_index
    factor_quantile.index = factor_index
    can_enter_t1.index = factor_index

    # 将价格矩阵按模式平移，但因子索引始终保留信号日 T。
    return_prices = build_return_price_view(prices, str(mode["name"]))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=FutureWarning)
        factor_data = al.utils.compute_forward_returns(
            factor=factor,
            prices=return_prices,
            periods=periods,
            filter_zscore=None,
            cumulative_returns=True,
        )
    if not factor_data.index.equals(factor.index):
        raise AssertionError("Alphalens 收益键与 T 日因子键不一致")
    factor_data["factor"] = factor.to_numpy(np.float64)
    factor_data["factor_quantile"] = factor_quantile.to_numpy(np.int64)
    forward_columns = tuple(f"{period}D" for period in periods)
    missing_columns = set((*forward_columns, "factor", "factor_quantile")).difference(  # 检查输入缺失字段
        factor_data.columns
    )
    if missing_columns:
        raise AssertionError(f"Alphalens 缺少字段：{sorted(missing_columns)}")
    if not np.isfinite(
        factor_data[[*forward_columns, "factor"]].to_numpy(np.float64)
    ).all():
        raise AssertionError("Alphalens 输出含 NaN 或 Inf")

    # IC 和基准始终使用完整 T 日截面；只有口径1的组合与换手应用 T+1 掩码。
    execution_factor_data = select_execution_factor_data(
        factor_data, can_enter_t1, str(mode["name"])
    )

    # 必须保留完整交易日 level：Alphalens 从 level.freq 获取交易日频率。
    # remove_unused_levels() 会在存在缺失因子日时把频率清成 None，继而
    # 让 asfreq(None) 退化为自然日并插入周末 NaN，使 22 日滚动均线全空。
    date_level = factor_data.index.names.index("date")
    factor_calendar_frequency = factor_data.index.levels[date_level].freq
    execution_calendar_frequency = execution_factor_data.index.levels[date_level].freq
    if factor_calendar_frequency is None:
        raise AssertionError("因子日期 level 缺少交易日频率")
    if execution_calendar_frequency != factor_calendar_frequency:
        raise AssertionError("完整因子与可交易视图的交易日频率不一致")
    factor_calendar_frequency_name = type(factor_calendar_frequency).__name__
    portfolio_rows_after_execution_filter = len(execution_factor_data)
    execution_quantile_counts = (  # 按完整索引重排并保留缺项
        execution_factor_data.groupby(
            [pd.Grouper(level="date"), "factor_quantile"], observed=True
        ).size().unstack("factor_quantile")
        .reindex(columns=range(1, quantiles + 1))
    )
    if execution_quantile_counts.isna().any(axis=None):
        raise AssertionError("执行过滤后存在空分位组，无法计算组合收益")

    # 抽样手算，确认 Alphalens 与所选价格起点完全一致。
    audit_positions = np.unique(
        np.linspace(0, len(factor_data) - 1, min(25, len(factor_data)), dtype=np.int64)
    )
    audit_index = factor_data.index[audit_positions]
    audit_keys = audit_index.to_frame(index=False)
    audit_rows = prices.index.get_indexer(audit_keys["date"])
    audit_columns = prices.columns.get_indexer(audit_keys["asset"])
    formula_records = []
    for period, column in zip(periods, forward_columns):
        manual = (
            price_values[
                audit_rows + price_start_offset + period, audit_columns
            ]
            / price_values[audit_rows + price_start_offset, audit_columns]
            - 1
        )
        library = factor_data.loc[audit_index, column].to_numpy(np.float64)
        np.testing.assert_allclose(library, manual, rtol=1e-12, atol=1e-12)  # 在数值容差内核对结果
        formula_records.append(
            {
                "period": column,
                "price_start": price_point_label(price_start_offset),
                "price_end": price_point_label(price_start_offset + period),
                "formula": forward_return_formula(price_start_offset),
                "samples_checked": len(audit_index),
                "max_abs_error": float(np.max(np.abs(library - manual))),
            }
        )

    daily_ic = al.performance.factor_information_coefficient(  # 固定日期与股票排序
        factor_data, group_adjust=False, by_group=False
    ).sort_index()
    ic_nan_rows_by_period = {
        column: int(daily_ic[column].isna().sum()) for column in forward_columns
    }
    if daily_ic.index.freq is None:
        raise AssertionError("日度 IC 索引缺少交易日频率")
    if daily_ic.index.freq != factor_calendar_frequency:
        raise AssertionError("日度 IC 与因子数据的交易日频率不一致")
    ic_rolling_22 = daily_ic.rolling(window=22).mean()
    ic_rolling_22_valid_rows_by_period = {
        column: int(ic_rolling_22[column].notna().sum())
        for column in forward_columns
    }
    if any(count == 0 for count in ic_rolling_22_valid_rows_by_period.values()):
        raise AssertionError("1 month moving avg 没有有效点")

    # Alphalens 0.4.6 的均值/标准差会自动跳过 NaN，但 SciPy 统计量
    # 默认传播 NaN。这里保持原表格结构，只把后四项改为逐列忽略缺失值。
    def plot_information_table_omit_nan(ic_data: Any, return_df: Any=False) -> Any:
        """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。"""
        values = ic_data.to_numpy(np.float64)
        table = pd.DataFrame(index=ic_data.columns)
        table["IC Mean"] = ic_data.mean()
        table["IC Std."] = ic_data.std()
        table["Risk-Adjusted IC"] = table["IC Mean"] / table["IC Std."]
        test = stats.ttest_1samp(  # 检验日度指标均值是否显著
            values, popmean=0.0, axis=0, nan_policy="omit"
        )
        table["t-stat(IC)"] = test.statistic
        table["p-value(IC)"] = test.pvalue
        table["IC Skew"] = stats.skew(values, axis=0, nan_policy="omit")  # 统计因子偏度
        table["IC Kurtosis"] = stats.kurtosis(  # 统计因子峰度
            values, axis=0, nan_policy="omit"
        )
        if return_df:
            return table
        print("Information Analysis")
        al.utils.print_table(table.apply(lambda column: column.round(3)).T)

    ic_records, annual_records = [], []
    for column in forward_columns:
        values = daily_ic[column].dropna()  # 剔除缺失值后计算
        test = stats.ttest_1samp(values.to_numpy(), popmean=0.0)  # 检验日度指标均值是否显著
        ic_records.append(
            {
                "period": column,
                "days": len(values),
                "mean_ic": float(values.mean()),
                "median_ic": float(values.median()),
                "std_ic": float(values.std(ddof=1)),
                "positive_ratio": float(values.gt(0).mean()),
                "t_stat": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )
        for year, annual in values.groupby(values.index.year):
            annual_records.append(
                {
                    "year": int(year), "period": column, "days": len(annual),
                    "mean_ic": float(annual.mean()),
                    "median_ic": float(annual.median()),
                    "std_ic": float(annual.std(ddof=1)),
                    "positive_ratio": float(annual.gt(0).mean()),
                }
            )
    ic_summary = pd.DataFrame(ic_records)
    annual_ic = pd.DataFrame(annual_records)
    information_table_audit = plot_information_table_omit_nan(  # 按完整索引重排并保留缺项
        daily_ic, return_df=True
    ).reindex(forward_columns)
    if not np.isfinite(information_table_audit.to_numpy(np.float64)).all():
        raise AssertionError("Information Analysis 统计量仍含 NaN 或 Inf")
    np.testing.assert_allclose(  # 在数值容差内核对结果
        information_table_audit["t-stat(IC)"].to_numpy(np.float64),
        ic_summary["t_stat"].to_numpy(np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(  # 在数值容差内核对结果
        information_table_audit["p-value(IC)"].to_numpy(np.float64),
        ic_summary["p_value"].to_numpy(np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    mean_by_quantile, _ = al.performance.mean_return_by_quantile(
        execution_factor_data,
        by_date=False,
        by_group=False,
        demeaned=False,
        group_adjust=False,
    )
    mean_by_date, standard_error_by_date = al.performance.mean_return_by_quantile(
        execution_factor_data,
        by_date=True,
        by_group=False,
        demeaned=False,
        group_adjust=False,
    )
    daily_spread, _ = al.performance.compute_mean_returns_spread(
        mean_by_date, upper_quant=quantiles, lower_quant=1,
        std_err=standard_error_by_date,
    )
    quantile_return_summary = mean_by_quantile.reset_index().rename(
        columns={"factor_quantile": "quantile"}
    )
    spread_summary = pd.DataFrame(
        {
            "period": forward_columns,
            "Q5_minus_Q1_return": [float(daily_spread[column].mean()) for column in forward_columns],
            "standard_error": [float(daily_spread[column].sem(ddof=1)) for column in forward_columns],
        }
    )
    daily_quantile_returns = mean_by_date["1D"].unstack("factor_quantile").sort_index()
    equal_weight_market = compute_equal_weight_market_returns(factor_data)
    portfolio_returns = pd.concat(
        {
            "Q5 long-only": daily_quantile_returns[quantiles],
            "Q5-Q1 long-short": (
                daily_quantile_returns[quantiles] - daily_quantile_returns[1]
            ),
            EQUAL_WEIGHT_MARKET_COLUMN: equal_weight_market,
        },
        axis=1,
    )
    if portfolio_returns.isna().any(axis=None):
        raise AssertionError("组合收益与全市场等权基准日期未完全对齐")
    relative_performance = compute_relative_performance(portfolio_returns)
    pyfolio_performance = pd.concat(
        {
            column: pf.timeseries.perf_stats(portfolio_returns[column])
            for column in portfolio_returns
        },
        axis=1,
    )

    package_names = [
        "alphalens-reloaded", "pyfolio-reloaded", "pandas", "numpy", "pyarrow"
    ]
    installed_versions = {
        package: metadata.version(package) for package in package_names
    }
    final_audit = pd.DataFrame(
        {
            "item": [
                "A2 run", "prediction years", "backtest mode",
                "T-day grouping rows before price filter",
                "T-day grouping rows after price filter / IC rows",
                "portfolio rows after execution filter",
                "factor grouping policy", "IC scope", "portfolio scope",
                "execution mask", "benchmark scope", "benchmark weighting",
                "factor dates", "IC calendar frequency",
                "IC calendar NaN rows", "IC rolling-22 valid rows",
                "Information table NaN policy", "price column",
                "price start", "price start offset",
                "forward return formula", "periods", "quantiles",
                "MAX_LOSS", "Universe signature", "dataset signature",
            ],
            "value": [
                run_name, str(prediction_years), str(mode["name"]),
                f"{t_grouping_rows_before_price_filter:,}",
                f"{t_grouping_rows_after_price_filter:,}",
                f"{portfolio_rows_after_execution_filter:,}",
                "T full cross-section before execution filter",
                "T full cross-section with valid forward prices",
                str(mode["portfolio_scope"]), str(mode["execution_mask"]),
                "full T factor pool after required-price filter",
                "daily equal weight; no can_enter_t1",
                f"{factor_data.index.get_level_values('date').nunique():,}",
                factor_calendar_frequency_name, str(ic_nan_rows_by_period),
                str(ic_rolling_22_valid_rows_by_period), "omit", price_column,
                str(mode["price_start"]), str(price_start_offset),
                forward_return_formula(price_start_offset),
                str(periods), str(quantiles), str(max_loss),
                universe_manifest["signature"], dataset_manifest["signature"],
            ],
        }
    )

    atomic_parquet(prediction_audit, evaluation_dir / "prediction_universe_audit.parquet")  # 原子落盘，避免半成品
    atomic_parquet(price_coverage, evaluation_dir / "price_coverage.parquet")  # 原子落盘，避免半成品
    atomic_parquet(pd.DataFrame(formula_records), evaluation_dir / "return_formula_audit.parquet")  # 原子落盘，避免半成品
    atomic_parquet(ic_summary, evaluation_dir / "ic_summary.parquet")  # 原子落盘，避免半成品
    atomic_parquet(annual_ic, evaluation_dir / "annual_ic_summary.parquet")  # 原子落盘，避免半成品
    atomic_parquet(quantile_return_summary, evaluation_dir / "quantile_return_summary.parquet")  # 原子落盘，避免半成品
    atomic_parquet(spread_summary, evaluation_dir / "spread_summary.parquet")  # 原子落盘，避免半成品
    atomic_parquet(portfolio_returns.reset_index(), evaluation_dir / "portfolio_returns.parquet")  # 原子落盘，避免半成品
    atomic_parquet(  # 原子落盘，避免半成品
        pyfolio_performance.rename_axis("metric").reset_index(),
        evaluation_dir / "pyfolio_performance.parquet",
    )
    atomic_parquet(  # 原子落盘，避免半成品
        relative_performance, evaluation_dir / "relative_performance.parquet"
    )
    atomic_parquet(final_audit, evaluation_dir / "final_audit.parquet")  # 原子落盘，避免半成品
    evaluation_manifest = {
        "schema_version": 2,
        "run_name": run_name,
        "prediction_years": list(prediction_years),
        "dataset_signature": dataset_manifest["signature"],
        "universe_signature": universe_manifest["signature"],
        "input_prediction_rows": int(prediction_audit["prediction_rows"].sum()),
        "t1_excluded_rows": int(prediction_audit["t1_excluded_rows"].sum()),
        "factor_rows_before_price_filter": t_grouping_rows_before_price_filter,
        "clean_factor_rows": t_grouping_rows_after_price_filter,
        "t_day_grouping_rows_before_price_filter": t_grouping_rows_before_price_filter,
        "t_day_grouping_rows_after_price_filter": t_grouping_rows_after_price_filter,
        "portfolio_rows_after_execution_filter": portfolio_rows_after_execution_filter,
        "backtest_mode": str(mode["name"]),
        "grouping_policy": "T_full_cross_section_before_execution_filter",
        "ic_scope": "T_full_cross_section_with_valid_forward_prices",
        "portfolio_scope": str(mode["portfolio_scope"]),
        "execution_mask": str(mode["execution_mask"]),
        "factor_dates": int(factor_data.index.get_level_values("date").nunique()),
        "factor_calendar_frequency": factor_calendar_frequency_name,
        "ic_rolling_22_valid_rows_by_period": ic_rolling_22_valid_rows_by_period,
        "price_column": price_column,
        "price_start": str(mode["price_start"]),
        "price_start_offset": price_start_offset,
        "forward_return_formula": forward_return_formula(price_start_offset),
        "periods": list(periods),
        "quantiles": quantiles,
        "max_loss": max_loss,
        "benchmark_column": EQUAL_WEIGHT_MARKET_COLUMN,
        "benchmark_scope": "feature_valid_and_in_universe_t_full_factor_pool_after_required_price_filter",
        "benchmark_weighting": "daily_equal_weight",
        "benchmark_execution_mask": "none",
        "annual_excess_method": "portfolio_geometric_annual_return_minus_benchmark_geometric_annual_return",
        "information_ratio_method": "mean_daily_active_return_div_daily_tracking_error_times_sqrt_252",
        "annual_tracking_error_method": "std_daily_active_return_ddof_1_times_sqrt_252",
        "annualization_days": ANNUALIZATION_DAYS,
        "long_short_gross_exposure": 2.0,
        "information_table_nan_policy": "omit",
        "ic_nan_rows_by_period": ic_nan_rows_by_period,
        "plot_font": "Noto Sans CJK SC",
        "versions": installed_versions,
    }

    display(prediction_audit)
    display(price_coverage.style.format({"coverage": "{:.6%}"}))
    display(pd.DataFrame(formula_records))
    display(ic_summary)
    display(annual_ic)
    display(quantile_return_summary)
    display(spread_summary)
    display(pyfolio_performance)
    display(relative_performance)
    display(final_audit)
    pic_dir = evaluation_dir / "pic"
    pic_dir.mkdir(parents=True, exist_ok=True)

    portfolio_net_values = (1 + portfolio_returns).cumprod()  # 复利累计得到净值
    price_start_title = "T" if price_start_offset == 0 else "T+1"
    cumulative_ax = portfolio_net_values.plot(
        figsize=(12, 5),
        title=f"A2 因子组合与全市场等权累计毛收益（{price_start_title} 日复权收盘起算）",
    )
    cumulative_ax.grid(alpha=0.25)
    cumulative_figure = cumulative_ax.get_figure()
    cumulative_path = evaluation_dir / "portfolio_cumulative_returns.png"
    cumulative_figure.savefig(cumulative_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"累计收益图：{cumulative_path}")
    plt.close(cumulative_figure)

    # 回撤按净值相对历史峰值计算；图例同时标出各组合最大回撤。
    portfolio_drawdowns = portfolio_net_values.div(
        portfolio_net_values.cummax().clip(lower=1.0)
    ).sub(1)
    np.testing.assert_allclose(  # 在数值容差内核对结果
        portfolio_drawdowns.min().to_numpy(np.float64),
        pyfolio_performance.loc["Max drawdown"].to_numpy(np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    drawdown_plot = portfolio_drawdowns.copy()
    drawdown_plot.columns = [
        f"{column}（最大回撤 {portfolio_drawdowns[column].min():.2%}）"
        for column in portfolio_drawdowns
    ]
    drawdown_ax = drawdown_plot.plot(
        figsize=(12, 5), title="A2 因子组合回撤曲线"
    )
    drawdown_ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    drawdown_ax.set_ylabel("回撤")
    drawdown_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    drawdown_ax.grid(alpha=0.25)
    drawdown_figure = drawdown_ax.get_figure()
    drawdown_path = pic_dir / "portfolio_drawdown.png"
    drawdown_figure.savefig(drawdown_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"最大回撤图：{drawdown_path}")
    plt.close(drawdown_figure)

    # Alphalens 在每个 tear sheet 的 show() 之后立即关闭 figure；
    # 因此在 show 时先保存，再继续让 Notebook 正常内嵌显示。
    for old_path in pic_dir.glob("alphalens_*_tear_sheet*.png"):
        old_path.unlink()
    tear_sheet_sections = ("returns", "information", "turnover")
    tear_sheet_files: list[str] = []
    original_show = plt.show
    original_information_table = al.plotting.plot_information_table
    section_number = 0

    def save_tear_sheet_figures(*show_args: Any, **show_kwargs: Any) -> None:
        """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。"""
        nonlocal section_number
        figures = []
        for number in plt.get_fignums():
            figure = plt.figure(number)
            if figure.axes:
                figures.append(figure)
        if not figures:
            raise RuntimeError("Alphalens tear sheet 未生成 figure")
        section = (
            tear_sheet_sections[section_number]
            if section_number < len(tear_sheet_sections)
            else f"section_{section_number + 1}"
        )
        for figure_number, figure in enumerate(figures, start=1):
            suffix = "" if len(figures) == 1 else f"_{figure_number}"
            path = pic_dir / f"alphalens_{section}_tear_sheet{suffix}.png"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            tear_sheet_files.append(str(path.relative_to(evaluation_dir)))
        original_show(*show_args, **show_kwargs)
        section_number += 1

    plt.close("all")
    plt.show = save_tear_sheet_figures
    al.plotting.plot_information_table = plot_information_table_omit_nan
    try:
        # information 始终看完整 T 日截面；returns/turnover 使用模式对应执行视图。
        al.plotting.plot_quantile_statistics_table(factor_data)
        al.tears.create_returns_tear_sheet(
            execution_factor_data,
            long_short=True,
            group_neutral=False,
            by_group=False,
        )
        al.tears.create_information_tear_sheet(
            factor_data,
            group_neutral=False,
            by_group=False,
        )
        al.tears.create_turnover_tear_sheet(execution_factor_data)
    finally:
        plt.show = original_show
        al.plotting.plot_information_table = original_information_table
        plt.close("all")
    if section_number != len(tear_sheet_sections):
        raise AssertionError(
            f"Alphalens tear sheet 数量异常：{section_number}"
        )
    evaluation_manifest["plot_files"] = [
        str(cumulative_path.relative_to(evaluation_dir)),
        str(drawdown_path.relative_to(evaluation_dir)),
        *tear_sheet_files,
    ]
    evaluation_manifest["drawdown_file"] = str(
        drawdown_path.relative_to(evaluation_dir)
    )
    evaluation_manifest["tear_sheet_files"] = tear_sheet_files
    evaluation_manifest["tear_sheet_scopes"] = {
        "returns": str(mode["portfolio_scope"]),
        "information": "T_full_cross_section_with_valid_forward_prices",
        "turnover": str(mode["portfolio_scope"]),
    }
    atomic_json(  # 原子落盘，避免半成品
        evaluation_manifest,
        evaluation_dir / "factor_evaluation_manifest.json",
    )
    print(f"Alphalens tear sheet：{pic_dir}")

    del factor, factor_quantile, can_enter_t1
    del factor_data, execution_factor_data
    del prices, return_prices, price_values, eod
    gc.collect()
    generate_experiment_report(run_name=run_name, data_root=root)
    print(
        f"样本外因子评价完成；模式={mode['name']}，"
        f"价格起点={price_point_label(price_start_offset)}。"
    )
    return final_audit


# ============================ 实验报告 ============================
def _markdown_table(frame: pd.DataFrame) -> str:
    """避免引入额外模板依赖的小型 Markdown 表格。"""

    if frame.empty:
        return "（暂无记录）"
    clean = frame.copy().fillna("")
    headers = [str(column) for column in clean.columns]
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for record in clean.astype(str).itertuples(index=False, name=None):
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in record) + " |")
    return "\n".join(rows)


def generate_experiment_report(
    run_name: str = DEFAULT_A2_RUN_NAME,
    data_root: str | Path | None = None,
) -> Path:
    """根据指定 run 的 manifest 和审计表生成中文实验报告。"""

    run_name = validate_run_name(run_name)
    root = (
        Path(data_root).resolve()
        if data_root is not None
        else get_data_root().resolve()
    )
    if not root.is_dir():
        raise FileNotFoundError(root)
    quality_run = root / "quality" / "runs" / run_name
    universe_manifest = read_json(root / "universe" / "universe_manifest.json")
    label_manifest = read_json(root / "labels_t10" / "labels_manifest.json")
    dataset_manifest = read_json(root / "feature_store" / "dataset_manifest.json")
    rolling_path = root / "feature_store" / "rolling_split_summary.parquet"
    if universe_manifest is None or label_manifest is None or dataset_manifest is None:
        raise FileNotFoundError("Universe、标签或 feature store manifest 尚未生成")
    universe_quality = pd.read_parquet(
        root / "quality" / "universe_quality_by_date.parquet"
    )
    universe_quality["year"] = universe_quality["trade_date"].astype(str).str[:4]
    universe_yearly = universe_quality.groupby("year", as_index=False).agg(  # 按既定口径聚合
        candidate_rows=("candidate_rows", "sum"),
        in_universe_t_rows=("in_universe_t_rows", "sum"),
        prediction_eligible_rows=("prediction_eligible_rows", "sum"),
        label_candidate_rows=("label_candidate_rows", "sum"),
        missing_status_t_rows=("missing_status_t_rows", "sum"),
        suspended_t_rows=("suspended_t_rows", "sum"),
        limit_up_t_rows=("limit_up_t_rows", "sum"),
        limit_down_t_rows=("limit_down_t_rows", "sum"),
    )
    label_quality = pd.read_parquet(root / "quality" / "label_quality_by_date.parquet")
    label_quality["year"] = label_quality["trade_date"].astype(str).str[:4]
    label_yearly = label_quality.groupby("year", as_index=False).agg(  # 按既定口径聚合
        label_candidate_rows=("label_candidate_rows", "sum"),
        label_valid_rows=("label_valid_rows", "sum"),
        missing_or_nonpositive_price_rows=("missing_or_nonpositive_price_rows", "sum"),
    )
    rolling = pd.read_parquet(rolling_path)[
        ["forecast_year", "train_sample_count", "validation_sample_count", "prediction_sample_count"]
    ]
    training_path = quality_run / "vit_training_summary.parquet"
    if training_path.exists():
        training = pd.read_parquet(training_path)
        aggregations = {
            "completed_seeds": ("seed", "nunique"),
            "mean_best_validation_ic": ("best_validation_ic", "mean"),
            "min_best_validation_ic": ("best_validation_ic", "min"),
            "mean_best_epoch": ("best_epoch", "mean"),
        }
        if "best_validation_rank_ic" in training.columns:
            aggregations["mean_best_validation_rank_ic"] = (
                "best_validation_rank_ic", "mean"
            )
        if "best_validation_daily_pearson_ic" in training.columns:
            aggregations["mean_best_validation_daily_pearson_ic"] = (
                "best_validation_daily_pearson_ic", "mean"
            )
        elif "best_validation_daily_ic" in training.columns:
            aggregations["mean_best_validation_daily_pearson_ic"] = (
                "best_validation_daily_ic", "mean"
            )
        if "best_validation_pooled_ic" in training.columns:
            aggregations["mean_best_validation_pooled_ic"] = (
                "best_validation_pooled_ic", "mean"
            )
        training_yearly = training.groupby(  # 按既定口径聚合
            "forecast_year", as_index=False
        ).agg(**aggregations)
    else:
        training_yearly = pd.DataFrame()
    evaluation_dir = quality_run / "factor_evaluation"
    ic_path = evaluation_dir / "ic_summary.parquet"
    spread_path = evaluation_dir / "spread_summary.parquet"
    audit_path = evaluation_dir / "final_audit.parquet"
    pyfolio_path = evaluation_dir / "pyfolio_performance.parquet"
    relative_path = evaluation_dir / "relative_performance.parquet"
    ic_summary = pd.read_parquet(ic_path) if ic_path.exists() else pd.DataFrame()
    spread_summary = pd.read_parquet(spread_path) if spread_path.exists() else pd.DataFrame()
    final_audit = pd.read_parquet(audit_path) if audit_path.exists() else pd.DataFrame()
    pyfolio = (
        pd.read_parquet(pyfolio_path) if pyfolio_path.exists() else pd.DataFrame()
    )
    relative_performance = (
        pd.read_parquet(relative_path) if relative_path.exists() else pd.DataFrame()
    )
    evaluation_manifest = read_json(
        evaluation_dir / "factor_evaluation_manifest.json"
    )
    if evaluation_manifest:
        price_start_offset = int(evaluation_manifest.get("price_start_offset", 0))
        period_description = (
            "1/5/10D 对应 T→T+1/T+5/T+10"
            if price_start_offset == 0
            else "1/5/10D 对应 T+1→T+2/T+6/T+11"
        )
        execution_description = (
            "组合收益和换手在完整 T 日分组后应用 `can_enter_t1`"
            if evaluation_manifest.get("execution_mask") == "can_enter_t1"
            else "组合收益和换手不应用 `can_enter_t1` 未来状态过滤"
        )
        evaluation_policy_lines = [
            f"- 回测模式：`{evaluation_manifest.get('backtest_mode')}`；{execution_description}。",
            f"- 收益公式：`{evaluation_manifest.get('forward_return_formula')}`；{period_description}。",
            "- 全市场等权基准使用完整因子覆盖池和相同收益时点，不应用 `can_enter_t1`。",
            "- Q5 与 Q5−Q1 的超额年化为组合几何年化减基准几何年化；信息比使用日度主动收益并按 252 日年化。",
        ]
    else:
        evaluation_policy_lines = ["- 因子评价尚未按新版回测口径生成。"]
    training_manifest = read_json(quality_run / "a2_training_manifest.json")
    batch_summary = pd.DataFrame(
        training_manifest.get("training_batch_size_by_year", [])
        if training_manifest else []
    )

    lines = [
        f"# ViT Universe + A2 实验报告｜{run_name}",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        "## 实验口径",
        "",
        "- Universe 以 T 日状态定义，仅剔除停牌、涨停、跌停和状态缺失股票；不处理 ST 或上市时间。",
        "- 五分位和 RankIC 始终使用完整 T 日因子截面；训练标签仍使用 T+1 执行掩码。",
        "- 标签为复权收盘价 T+1→T+11 收益，每日 1%/99% Winsorize 后以 `ddof=0` 做截面 Z-score。",
        "- feature store 为 `feature_valid & in_universe_t`；无标签 target 保持 NaN，训练/验证只使用 `label_valid=True`。",
        *evaluation_policy_lines,
        "- Information Analysis 对缺失 IC 日期逐列采用 `nan_policy=omit`，其 t-stat/p-value 与 `ic_summary.parquet` 一致。",
        "- IC 索引保留 SSE `CustomBusinessDay` 频率，并校验 22 个交易日移动均线存在有效点。",
        "",
        "## Universe 年度覆盖",
        "",
        _markdown_table(universe_yearly),
        "",
        "## 标签年度覆盖",
        "",
        _markdown_table(label_yearly),
        "",
        "## 滚动切分",
        "",
        _markdown_table(rolling),
        "",
        "## A2 配置与训练结果",
        "",
    ]
    if training_manifest:
        train_batch_mode = training_manifest.get(
            "train_batch_mode", "fixed_samples"
        )
        inference_batch_size = training_manifest.get(
            "inference_batch_size",
            training_manifest.get("batch_size_per_task"),
        )
        validation_metric = training_manifest.get(
            "validation_metric", "pooled_pearson_ic"
        )
        loss_name = training_manifest.get("loss_name", "pearson_ic")
        loss_parameters = training_manifest.get("loss_parameters", {})
        lines.extend(
            [
                f"- 运行名：`{run_name}`",
                f"- GPU / 每卡并发：{training_manifest['device_ids']} / {training_manifest['tasks_per_gpu']}",
                f"- 年度 × 种子：{training_manifest['years']} × {training_manifest['seeds']}",
                f"- 训练 batch 模式：{train_batch_mode}；batch：{training_manifest['batch_size_per_task']}",
                f"- 损失函数：{loss_name}；参数：{loss_parameters}",
                f"- prediction batch / workers / lr：{inference_batch_size} / {training_manifest['workers_per_task']} / {training_manifest['learning_rate']}",
                f"- reshuffle / 验证选模指标：{training_manifest.get('reshuffle_each_epoch', True)} / {validation_metric}",
                f"- compile / dynamic / fused Adam / CUDA prefetch：{training_manifest['torch_compile']} / {training_manifest.get('compile_dynamic_batch', False)} / {training_manifest['fused_adam']} / {training_manifest['cuda_prefetch']}",
                "",
                "### 训练 batch 年度统计",
                "",
                _markdown_table(batch_summary),
                "",
            ]
        )
    else:
        lines.extend(["训练尚未完成。", ""])
    lines.extend(
        [
            _markdown_table(training_yearly),
            "",
            "## 样本外因子评价",
            "",
            _markdown_table(final_audit),
            "",
            "### RankIC",
            "",
            _markdown_table(ic_summary),
            "",
            "### Q5−Q1 收益差",
            "",
            _markdown_table(spread_summary),
            "",
            "### Pyfolio 绝对绩效",
            "",
            _markdown_table(pyfolio),
            "",
            "### 相对全市场等权绩效",
            "",
            _markdown_table(relative_performance),
            "",
            "## 产物与结论",
            "",
            f"- Universe 签名：`{universe_manifest['signature']}`",
            f"- 标签签名：`{label_manifest['signature']}`",
            f"- feature store 签名：`{dataset_manifest['signature']}`",
            "- 累计收益图：`factor_evaluation/portfolio_cumulative_returns.png`。",
            "- 最大回撤图：`factor_evaluation/pic/portfolio_drawdown.png`。",
            "- Alphalens full tear sheet：`factor_evaluation/pic/`；information 使用完整 T 日截面，returns/turnover 使用所选回测模式的执行样本。",
            "- 市场基准与相对绩效：`factor_evaluation/portfolio_returns.parquet`、`relative_performance.parquet`。",
            "- 原始 Wind 与 Level-2 图片未修改；未生成 delivery 交付包。",
            "- 图片数据截止 2024-12-31，因此 2025 只有三种子模型，没有 2025 年预测因子。",
            "",
        ]
    )
    quality_run.mkdir(parents=True, exist_ok=True)
    report_path = quality_run / "EXPERIMENT_REPORT.md"
    temporary = report_path.with_name(f"{report_path.name}.{os.getpid()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(report_path)
    print(f"实验报告：{report_path}")
    return report_path


# ============================== 评价 CLI ==============================
def add_evaluation_arguments(parser: argparse.ArgumentParser, *, include_backtest_mode: bool = False) -> None:
    """向命令行解析器注册样本外评价参数。"""
    parser.add_argument("--data-root", default=str(VIT_DIR / "data"))
    parser.add_argument("--run-name", default=DEFAULT_A2_RUN_NAME)
    if include_backtest_mode:
        parser.add_argument("--backtest-mode", choices=BACKTEST_MODES, default=DEFAULT_BACKTEST_MODE)


def parse_args() -> argparse.Namespace:
    """解析阶段 08 的回测或报告命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_evaluation_arguments(commands.add_parser("backtest"), include_backtest_mode=True)  # 无梯度计算验证指标
    add_evaluation_arguments(commands.add_parser("report"))  # 无梯度计算验证指标
    return parser.parse_args()


def main() -> None:
    """分派样本外回测或实验报告生成流程。"""
    args = parse_args()
    if args.command == "backtest":
        run_oos_backtest(run_name=args.run_name, data_root=args.data_root, backtest_mode=args.backtest_mode)
    elif args.command == "report":
        generate_experiment_report(run_name=args.run_name, data_root=args.data_root)


if __name__ == "__main__":
    main()
