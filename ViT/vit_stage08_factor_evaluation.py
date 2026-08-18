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


# 设置 `DEFAULT_A2_RUN_NAME` 的固定配置值，供后续流程统一使用。
DEFAULT_A2_RUN_NAME = "date_batch"
# 组织一组有序值，并保存到 `BACKTEST_MODES`。
BACKTEST_MODES = (
    "mode1_t_with_t1_filter",
    "mode2_t_no_filter",
    "mode3_t1_no_filter",
)
# 设置 `DEFAULT_BACKTEST_MODE` 的固定配置值，供后续流程统一使用。
DEFAULT_BACKTEST_MODE = "mode2_t_no_filter"
# 设置 `EQUAL_WEIGHT_MARKET_COLUMN` 的固定配置值，供后续流程统一使用。
EQUAL_WEIGHT_MARKET_COLUMN = "Equal-weight market"
# 组织一组有序值，并保存到 `RELATIVE_PORTFOLIOS`。
RELATIVE_PORTFOLIOS = ("Q5 long-only", "Q5-Q1 long-short")
# 设置 `ANNUALIZATION_DAYS` 的固定配置值，供后续流程统一使用。
ANNUALIZATION_DAYS = 252
# 构造键值映射，并保存到 `BACKTEST_MODE_CONFIGS`，供后续按键查询。
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


# 定义 `validate_run_name` 函数，封装这一阶段可独立调用的处理步骤。
def validate_run_name(value: str) -> str:
    """校验 run 名称，避免路径穿越和非法目录字符。

    输入：
        value: str。
    输出：
        str。
    """
    # 调用 `str(value).strip` 去除字符串首尾空白以统一配置值，并将结果保存到 `run_name`。
    run_name = str(value).strip()
    # 调用 `set` 整理数据结构，并将结果保存到 `allowed`。
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
    # 检查 `not run_name or run_name in {'.', '..'} or any((character not in allowed for character in run_n…`；根据条件结果选择对应处理分支。
    if (
        not run_name
        or run_name in {".", ".."}
        or any(character not in allowed for character in run_name)
    ):
        # 抛出 `ValueError('运行名含非法字符')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("运行名含非法字符")
    # 将 `run_name` 作为当前函数结果返回给调用方。
    return run_name


# 定义 `resolve_backtest_mode` 函数，封装这一阶段可独立调用的处理步骤。
def resolve_backtest_mode(backtest_mode: str) -> dict[str, object]:
    """校验回测模式，并返回配置副本。

    输入：
        backtest_mode: str。
    输出：
        dict[str, object]。
    """

    # 调用 `str(backtest_mode).strip` 去除字符串首尾空白以统一配置值，并将结果保存到 `mode`。
    mode = str(backtest_mode).strip()
    # 检查 `mode not in BACKTEST_MODE_CONFIGS`；根据条件结果选择对应处理分支。
    if mode not in BACKTEST_MODE_CONFIGS:
        # 抛出 `ValueError(f'未知回测模式：{backtest_mode!r}；可选 {BACKTEST_MODES}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"未知回测模式：{backtest_mode!r}；可选 {BACKTEST_MODES}"
        )
    # 将 `{'name': mode, **BACKTEST_MODE_CONFIGS[mode]}` 作为当前函数结果返回给调用方。
    return {"name": mode, **BACKTEST_MODE_CONFIGS[mode]}


# 定义 `price_point_label` 函数，封装这一阶段可独立调用的处理步骤。
def price_point_label(offset: int) -> str:
    """返回回测价格时点的可读标签。

    输入：
        offset: int。
    输出：
        str。
    """
    # 将 `'T' if offset == 0 else f'T+{offset}'` 作为当前函数结果返回给调用方。
    return "T" if offset == 0 else f"T+{offset}"


# 定义 `forward_return_formula` 函数，封装这一阶段可独立调用的处理步骤。
def forward_return_formula(price_start_offset: int) -> str:
    """返回指定持有期对应的远期收益公式说明。

    输入：
        price_start_offset: int。
    输出：
        str。
    """
    # 检查 `price_start_offset == 0`；根据条件结果选择对应处理分支。
    if price_start_offset == 0:
        # 将 `'adj_close(T+k) / adj_close(T) - 1'` 作为当前函数结果返回给调用方。
        return "adj_close(T+k) / adj_close(T) - 1"
    # 将 `'adj_close(T+1+k) / adj_close(T+1) - 1'` 作为当前函数结果返回给调用方。
    return "adj_close(T+1+k) / adj_close(T+1) - 1"


# 定义 `geometric_annual_return` 函数，封装这一阶段可独立调用的处理步骤。
def geometric_annual_return(
    returns: pd.Series | np.ndarray, annualization: int = ANNUALIZATION_DAYS
) -> float:
    """以日收益复利计算几何年化收益。

    输入：
        returns: pd.Series | np.ndarray。
        annualization: int。
    输出：
        float。
    """

    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `values`。
    values = np.asarray(returns, dtype=np.float64)
    # 检查 `len(values) == 0 or not np.isfinite(values).all()`；根据条件结果选择对应处理分支。
    if len(values) == 0 or not np.isfinite(values).all():
        # 抛出 `ValueError('年化收益输入必须是非空有限日收益')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("年化收益输入必须是非空有限日收益")
    # 检查 `(values <= -1).any()`；根据条件结果选择对应处理分支。
    if (values <= -1).any():
        # 抛出 `ValueError('日收益必须大于 -100%')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("日收益必须大于 -100%")
    # 调用 `float` 统一设备或数据类型，并将结果保存到 `growth`。
    growth = float(np.prod(1.0 + values))
    # 检查 `not np.isfinite(growth) or growth <= 0`；根据条件结果选择对应处理分支。
    if not np.isfinite(growth) or growth <= 0:
        # 抛出 `ValueError('累计净值必须为正且有限')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("累计净值必须为正且有限")
    # 将 `growth ** (annualization / len(values)) - 1.0` 作为当前函数结果返回给调用方。
    return growth ** (annualization / len(values)) - 1.0


# 定义 `compute_relative_performance` 函数，封装这一阶段可独立调用的处理步骤。
def compute_relative_performance(
    portfolio_returns: pd.DataFrame,
    benchmark_column: str = EQUAL_WEIGHT_MARKET_COLUMN,
) -> pd.DataFrame:
    """计算 Q5 与 Q5-Q1 相对全市场等权基准的绩效。

    输入：
        portfolio_returns: pd.DataFrame。
        benchmark_column: str。
    输出：
        pd.DataFrame。
    """

    # 组织一组有序值，并保存到 `required`。
    required = (*RELATIVE_PORTFOLIOS, benchmark_column)
    # 调用 `set(required).difference` 计算两个集合的差集以找出缺失字段，并将结果保存到 `missing`。
    missing = set(required).difference(portfolio_returns.columns)
    # 检查 `missing`；根据条件结果选择对应处理分支。
    if missing:
        # 抛出 `KeyError(f'相对绩效缺少收益列：{sorted(missing)}')`，阻止不满足数据契约的流程继续执行。
        raise KeyError(f"相对绩效缺少收益列：{sorted(missing)}")
    # 调用 `portfolio_returns[list(required)].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `values`。
    values = portfolio_returns[list(required)].to_numpy(np.float64)
    # 检查 `len(portfolio_returns) < 2 or not np.isfinite(values).all()`；根据条件结果选择对应处理分支。
    if len(portfolio_returns) < 2 or not np.isfinite(values).all():
        # 抛出 `ValueError('相对绩效至少需要两日有限收益')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("相对绩效至少需要两日有限收益")

    # 调用 `portfolio_returns[benchmark_column].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `benchmark`。
    benchmark = portfolio_returns[benchmark_column].to_numpy(np.float64)
    # 调用 `geometric_annual_return` 由日收益复利计算几何年化收益，并将结果保存到 `benchmark_annual_return`。
    benchmark_annual_return = geometric_annual_return(benchmark)
    # 构造当前步骤所需的列表，并保存到 `records`。
    records = []
    # 遍历 `RELATIVE_PORTFOLIOS`，并把当前元素绑定到 `portfolio` 逐项处理。
    for portfolio in RELATIVE_PORTFOLIOS:
        # 调用 `portfolio_returns[portfolio].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `returns`。
        returns = portfolio_returns[portfolio].to_numpy(np.float64)
        # 计算 `returns - benchmark`，并将数值结果保存到 `active`。
        active = returns - benchmark
        # 调用 `max` 汇总当前数据，并将统计量保存到 `active_scale`。
        active_scale = max(1.0, float(np.max(np.abs(active))))
        # 检查 `float(np.ptp(active)) <= np.finfo(np.float64).eps * active_scale * 16`；根据条件结果选择对应处理分支。
        if float(np.ptp(active)) <= np.finfo(np.float64).eps * active_scale * 16:
            # 抛出 `ValueError(f'{portfolio} 主动收益方差为零，信息比无定义')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{portfolio} 主动收益方差为零，信息比无定义")
        # 调用 `float` 统一设备或数据类型，并将结果保存到 `active_std`。
        active_std = float(active.std(ddof=1))
        # 检查 `not np.isfinite(active_std) or active_std <= 0`；根据条件结果选择对应处理分支。
        if not np.isfinite(active_std) or active_std <= 0:
            # 抛出 `ValueError(f'{portfolio} 主动收益方差为零，信息比无定义')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{portfolio} 主动收益方差为零，信息比无定义")
        # 调用 `geometric_annual_return` 由日收益复利计算几何年化收益，并将结果保存到 `portfolio_annual_return`。
        portfolio_annual_return = geometric_annual_return(returns)
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
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
    # 将 `pd.DataFrame(records)` 作为当前函数结果返回给调用方。
    return pd.DataFrame(records)

# 定义 `build_return_price_view` 函数，封装这一阶段可独立调用的处理步骤。
def build_return_price_view(
    prices: pd.DataFrame, backtest_mode: str = DEFAULT_BACKTEST_MODE
) -> pd.DataFrame:
    """让 Alphalens 在信号日 T 索引上按所选起点计算远期收益。

    输入：
        prices: pd.DataFrame。
        backtest_mode: str。
    输出：
        pd.DataFrame。
    """

    # 调用 `int` 统一设备或数据类型，并将结果保存到 `offset`。
    offset = int(resolve_backtest_mode(backtest_mode)["price_start_offset"])
    # 将 `prices if offset == 0 else prices.shift(-offset)` 作为当前函数结果返回给调用方。
    return prices if offset == 0 else prices.shift(-offset)


# 定义 `select_execution_factor_data` 函数，封装这一阶段可独立调用的处理步骤。
def select_execution_factor_data(
    factor_data: pd.DataFrame,
    can_enter_t1: pd.Series | np.ndarray,
    backtest_mode: str = DEFAULT_BACKTEST_MODE,
) -> pd.DataFrame:
    """仅口径 1 在已固定分位后应用 T+1 可进入掩码。

    输入：
        factor_data: pd.DataFrame。
        can_enter_t1: pd.Series | np.ndarray。
        backtest_mode: str。
    输出：
        pd.DataFrame。
    """

    # 调用 `resolve_backtest_mode` 解析用户配置并转换为内部统一口径，并将结果保存到 `mode`。
    mode = resolve_backtest_mode(backtest_mode)
    # 调用 `np.asarray` 转换数据表示，并将标准化对象保存到 `mask`。
    mask = np.asarray(can_enter_t1, dtype=bool)
    # 检查 `len(mask) != len(factor_data)`；根据条件结果选择对应处理分支。
    if len(mask) != len(factor_data):
        # 抛出 `ValueError('can_enter_t1 与因子数据长度不一致')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("can_enter_t1 与因子数据长度不一致")
    # 检查 `bool(mode['apply_can_enter_t1'])`；根据条件结果选择对应处理分支。
    if bool(mode["apply_can_enter_t1"]):
        # 将 `factor_data.iloc[np.flatnonzero(mask)].copy()` 作为当前函数结果返回给调用方。
        return factor_data.iloc[np.flatnonzero(mask)].copy()
    # 将 `factor_data` 作为当前函数结果返回给调用方。
    return factor_data


# 定义 `compute_equal_weight_market_returns` 函数，封装这一阶段可独立调用的处理步骤。
def compute_equal_weight_market_returns(
    factor_data: pd.DataFrame, return_column: str = "1D"
) -> pd.Series:
    """在完整、价格有效的 T 日因子覆盖池内计算等权市场收益。

    输入：
        factor_data: pd.DataFrame。
        return_column: str。
    输出：
        pd.Series。
    """

    # 检查 `return_column not in factor_data.columns`；根据条件结果选择对应处理分支。
    if return_column not in factor_data.columns:
        # 抛出 `KeyError(f'缺少基准收益列：{return_column}')`，阻止不满足数据契约的流程继续执行。
        raise KeyError(f"缺少基准收益列：{return_column}")
    # 调用 `factor_data[return_column].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `values`。
    values = factor_data[return_column].to_numpy(np.float64)
    # 检查 `not np.isfinite(values).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(values).all():
        # 抛出 `ValueError('全市场等权基准输入含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("全市场等权基准输入含 NaN 或 Inf")
    # 将 `factor_data[return_column].groupby(level='date', sort=True).mean().rename(EQUAL_WEIGHT_MARKET_C…` 作为当前函数结果返回给调用方。
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
    """按指定价格起点与执行过滤口径计算样本外收益。

    输入：
        run_name: str。
        data_root: str | Path | None。
        backtest_mode: str。
    输出：
        pd.DataFrame。
    """

    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `run_name`。
    run_name = validate_run_name(run_name)
    # 调用 `resolve_backtest_mode` 解析用户配置并转换为内部统一口径，并将结果保存到 `mode`。
    mode = resolve_backtest_mode(backtest_mode)
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `price_start_offset`。
    price_start_offset = int(mode["price_start_offset"])
    # 调用 `warnings.filterwarnings` 根据有效性条件筛选当前样本或构造掩码。
    warnings.filterwarnings(
        "ignore", message=r'Module "zipline\.assets" not found.*'
    )
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        import alphalens as al
        import matplotlib.pyplot as plt
        import pyfolio as pf
        from matplotlib.ticker import PercentFormatter
        from scipy import stats
    # 捕获 `except ModuleNotFoundError as error` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except ModuleNotFoundError as error:
        # 抛出 `RuntimeError('请使用 Jupyter 内核 `ViT Factor Eval (Alphalens/Pyfolio)`。')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError(
            "请使用 Jupyter 内核 `ViT Factor Eval (Alphalens/Pyfolio)`。"
        ) from error

    # factor-eval 环境已安装该字体；显式置顶，避免 Matplotlib 默认
    # DejaVu Sans 缺少中文字符而在 Notebook 中产生方框和大量 warning。
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    # 设置 `plt.rcParams['axes.unicode_minus']` 的固定配置值，供后续流程统一使用。
    plt.rcParams["axes.unicode_minus"] = False

    # 设置 `factor_column` 的固定配置值，供后续流程统一使用。
    factor_column = "prediction_ensemble"
    # 组织一组有序值，并保存到 `periods`。
    periods = (1, 5, 10)
    # 设置 `quantiles` 的固定配置值，供后续流程统一使用。
    quantiles = 5
    # 设置 `max_loss` 的固定配置值，供后续流程统一使用。
    max_loss = 0.001
    # 设置 `price_column` 的固定配置值，供后续流程统一使用。
    price_column = "s_dq_adjclose"
    # 按 `data_root is not None` 在两个候选值间选择，并保存到 `root`。
    root = (
        Path(data_root).resolve()
        if data_root is not None
        else get_data_root().resolve()
    )
    # 检查 `not root.is_dir()`；根据条件结果选择对应处理分支。
    if not root.is_dir():
        # 抛出 `FileNotFoundError(root)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(root)
    # 计算 `root / 'predictions' / 'runs' / run_name`，并将数值结果保存到 `prediction_dir`。
    prediction_dir = root / "predictions" / "runs" / run_name
    # 计算 `root / 'universe'`，并将数值结果保存到 `universe_dir`。
    universe_dir = root / "universe"
    # 计算 `root / 'quality' / 'runs' / run_name`，并将数值结果保存到 `quality_dir`。
    quality_dir = root / "quality" / "runs" / run_name
    # 计算 `quality_dir / 'factor_evaluation'`，并将数值结果保存到 `evaluation_dir`。
    evaluation_dir = quality_dir / "factor_evaluation"
    # 调用 `evaluation_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    # 调用 `read_json` 读取源数据，并将结果保存到 `training_manifest`。
    training_manifest = read_json(quality_dir / "a2_training_manifest.json")
    # 调用 `read_json` 读取源数据，并将结果保存到 `universe_manifest`。
    universe_manifest = read_json(universe_dir / "universe_manifest.json")
    # 调用 `read_json` 读取源数据，并将结果保存到 `dataset_manifest`。
    dataset_manifest = read_json(root / "feature_store" / "dataset_manifest.json")
    # 检查 `training_manifest is None or universe_manifest is None or dataset_manifest is None`；根据条件结果选择对应处理分支。
    if training_manifest is None or universe_manifest is None or dataset_manifest is None:
        # 抛出 `FileNotFoundError('训练、Universe 或 feature store manifest 不完整')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("训练、Universe 或 feature store manifest 不完整")
    # 检查 `training_manifest['universe_signature'] != universe_manifest['signature']`；根据条件结果选择对应处理分支。
    if training_manifest["universe_signature"] != universe_manifest["signature"]:
        # 抛出 `AssertionError('训练使用的 Universe 与当前 Universe 签名不一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("训练使用的 Universe 与当前 Universe 签名不一致")
    # 检查 `training_manifest['dataset_signature'] != dataset_manifest['signature']`；根据条件结果选择对应处理分支。
    if training_manifest["dataset_signature"] != dataset_manifest["signature"]:
        # 抛出 `AssertionError('训练使用的 feature store 与当前数据签名不一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("训练使用的 feature store 与当前数据签名不一致")

    # 调用 `tuple` 整理数据结构，并将结果保存到 `prediction_years`。
    prediction_years = tuple(
        year
        for year in training_manifest["years"]
        if (prediction_dir / f"vit_predictions.{year}.parquet").exists()
    )
    # 检查 `not prediction_years`；根据条件结果选择对应处理分支。
    if not prediction_years:
        # 抛出 `FileNotFoundError('没有可评价的年度预测')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("没有可评价的年度预测")
    # 构造当前步骤所需的列表，并保存到 `prediction_parts`。
    prediction_parts: list[pd.DataFrame] = []
    # 构造当前步骤所需的列表，并保存到 `audit_records`。
    audit_records: list[dict[str, object]] = []
    # 遍历 `prediction_years`，并把当前元素绑定到 `year` 逐项处理。
    for year in prediction_years:
        # 计算 `prediction_dir / f'vit_predictions.{year}.parquet'`，并将数值结果保存到 `prediction_path`。
        prediction_path = prediction_dir / f"vit_predictions.{year}.parquet"
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `prediction`。
        prediction = pd.read_parquet(
            prediction_path,
            columns=["trade_date", "instrument_id", factor_column],
        )
        # 调用 `prediction['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `prediction['trade_date']`。
        prediction["trade_date"] = prediction["trade_date"].astype(str)
        # 调用 `prediction['instrument_id'].astype` 统一设备或数据类型，并将结果保存到 `prediction['instrument_id']`。
        prediction["instrument_id"] = prediction["instrument_id"].astype(str)
        # 检查 `prediction.duplicated(['trade_date', 'instrument_id']).any()`；根据条件结果选择对应处理分支。
        if prediction.duplicated(["trade_date", "instrument_id"]).any():
            # 抛出 `ValueError(f'{year} prediction 业务键重复')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{year} prediction 业务键重复")
        # 检查 `not prediction['trade_date'].str.startswith(str(year)).all()`；根据条件结果选择对应处理分支。
        if not prediction["trade_date"].str.startswith(str(year)).all():
            # 抛出 `ValueError(f'{year} prediction 含跨年度日期')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{year} prediction 含跨年度日期")
        # 检查 `not np.isfinite(prediction[factor_column].to_numpy(np.float64)).all()`；根据条件结果选择对应处理分支。
        if not np.isfinite(prediction[factor_column].to_numpy(np.float64)).all():
            # 抛出 `ValueError(f'{year} 因子含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{year} 因子含 NaN 或 Inf")

        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `universe`。
        universe = pd.read_parquet(
            universe_dir / f"universe.{year}.parquet",
            columns=[
                "trade_date", "instrument_id", "feature_valid",
                "in_universe_t", "can_enter_t1",
            ],
        )
        # 调用 `copy` 复制当前对象以避免后续原地修改污染源数据，并将结果保存到 `expected`。
        expected = universe.loc[
            universe["feature_valid"] & universe["in_universe_t"],
            ["trade_date", "instrument_id", "can_enter_t1"],
        ].copy()
        # 调用 `expected['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `expected['trade_date']`。
        expected["trade_date"] = expected["trade_date"].astype(str)
        # 调用 `expected['instrument_id'].astype` 统一设备或数据类型，并将结果保存到 `expected['instrument_id']`。
        expected["instrument_id"] = expected["instrument_id"].astype(str)
        # 调用 `prediction.merge` 按键合并数据，并将对齐后的结果保存到 `joined`。
        joined = prediction.merge(
            expected,
            on=["trade_date", "instrument_id"],
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        # 检查 `not joined['_merge'].eq('both').all()`；根据条件结果选择对应处理分支。
        if not joined["_merge"].eq("both").all():
            # 抛出 `AssertionError(f'{year} prediction 与 T 日 Universe 键不一致')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} prediction 与 T 日 Universe 键不一致")
        # 调用 `joined.drop` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `joined`。
        joined = joined.drop(columns="_merge")
        # 检查 `joined['can_enter_t1'].isna().any()`；根据条件结果选择对应处理分支。
        if joined["can_enter_t1"].isna().any():
            # 抛出 `AssertionError(f'{year} can_enter_t1 含缺失值')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} can_enter_t1 含缺失值")
        # 调用 `joined['can_enter_t1'].astype` 统一设备或数据类型，并将结果保存到 `joined['can_enter_t1']`。
        joined["can_enter_t1"] = joined["can_enter_t1"].astype(bool)
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `can_enter_t1_rows`。
        can_enter_t1_rows = int(joined["can_enter_t1"].sum())
        # 把本轮结果追加到 `audit_records`，累积后续需要汇总的记录。
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
        # 把本轮结果追加到 `prediction_parts`，累积后续需要汇总的记录。
        prediction_parts.append(joined)

    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `prediction_audit`。
    prediction_audit = pd.DataFrame(audit_records)
    # 调用 `pd.concat` 拼接各批结果，并将合并值保存到 `panel`。
    panel = pd.concat(prediction_parts, ignore_index=True)
    # 删除 `prediction_parts, prediction, universe, expected, joined`，释放不再使用的引用。
    del prediction_parts, prediction, universe, expected, joined
    # 检查 `panel.duplicated(['trade_date', 'instrument_id']).any()`；根据条件结果选择对应处理分支。
    if panel.duplicated(["trade_date", "instrument_id"]).any():
        # 抛出 `ValueError('跨年度因子业务键重复')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("跨年度因子业务键重复")
    # 调用 `len` 汇总当前数据，并将统计量保存到 `t_grouping_rows_before_price_filter`。
    t_grouping_rows_before_price_filter = len(panel)
    # 调用 `pd.to_datetime` 把日期字段转换为 Pandas 时间类型，并将结果保存到 `panel['date']`。
    panel["date"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d")
    # 调用 `sort_index` 按索引键排序以保证时间和股票顺序稳定，并将结果保存到 `factor_panel`。
    factor_panel = (
        panel.set_index(["date", "instrument_id"])[
            [factor_column, "can_enter_t1"]
        ]
        .sort_index()
    )
    # 调用 `factor_panel.index.set_names` 设置多级索引各层名称，并将结果保存到 `factor_panel.index`。
    factor_panel.index = factor_panel.index.set_names(["date", "asset"])
    # 调用 `factor_panel[factor_column].rename` 把完整临时产物替换为正式输出，并将结果保存到 `factor`。
    factor = factor_panel[factor_column].rename("factor")
    # 调用 `factor_panel['can_enter_t1'].astype` 统一设备或数据类型，并将结果保存到 `can_enter_t1`。
    can_enter_t1 = factor_panel["can_enter_t1"].astype(bool)
    # 删除 `panel, factor_panel`，释放不再使用的引用。
    del panel, factor_panel

    # 分位组只能使用 T 日已经可知的因子截面。先对完整 T 日预测池分组，
    # 后续价格缺失与 T+1 可进入状态都只能删行，不能重新改变组别边界。
    groups = factor.groupby(level="date", sort=False)
    # 调用 `groups.rank` 根据分数计算截面排序或排序索引，并将结果保存到 `ordinal_rank`。
    ordinal_rank = groups.rank(method="first", ascending=True)
    # 调用 `groups.transform` 把分组统计结果广播回每一条原始记录，并将结果保存到 `group_size`。
    group_size = groups.transform("size")
    # 调用 `pd.Series` 转换数据表示，并将标准化对象保存到 `factor_quantile`。
    factor_quantile = pd.Series(
        np.floor(
            (ordinal_rank.to_numpy(np.float64) - 1) * quantiles
            / group_size.to_numpy(np.float64)
        ).astype(np.int64)
        + 1,
        index=factor.index,
        name="factor_quantile",
    )
    # 调用 `unstack` 把分组索引展开为列以形成日期×分组矩阵，并将结果保存到 `t_grouping_quantile_counts`。
    t_grouping_quantile_counts = (
        factor_quantile.groupby(
            [pd.Grouper(level="date"), factor_quantile], observed=True
        ).size().unstack("factor_quantile")
    )
    # 检查 `t_grouping_quantile_counts.isna().any(axis=None) or (t_grouping_quantile_counts.max(axis=1) - t…`；根据条件结果选择对应处理分支。
    if t_grouping_quantile_counts.isna().any(axis=None) or (
        t_grouping_quantile_counts.max(axis=1)
        - t_grouping_quantile_counts.min(axis=1)
    ).gt(1).any():
        # 抛出 `AssertionError('T 日完整截面五分位分组不均衡')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("T 日完整截面五分位分组不均衡")

    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `calendar`。
    calendar = pd.read_parquet(
        CALENDAR_PATH,
        columns=["trade_days", "s_info_exchmarket"],
        filters=[("s_info_exchmarket", "==", "SSE")],
    )
    # 调用 `pd.DatetimeIndex` 构造有序日期索引以完成交易日对齐，并将结果保存到 `calendar_dates`。
    calendar_dates = pd.DatetimeIndex(
        pd.to_datetime(normalize_dates(calendar["trade_days"]), format="%Y%m%d")
        .sort_values().unique(),
        name="date",
    )
    # 调用 `pd.DatetimeIndex` 构造有序日期索引以完成交易日对齐，并将结果保存到 `factor_dates`。
    factor_dates = pd.DatetimeIndex(
        factor.index.get_level_values("date").unique().sort_values()
    )
    # 调用 `pd.Series` 转换数据表示，并将标准化对象保存到 `positions`。
    positions = pd.Series(np.arange(len(calendar_dates)), index=calendar_dates)
    # 调用 `factor_dates.map` 按映射关系转换当前序列中的每个元素，并将结果保存到 `factor_positions`。
    factor_positions = factor_dates.map(positions)
    # 检查 `factor_positions.isna().any()`；根据条件结果选择对应处理分支。
    if factor_positions.isna().any():
        # 抛出 `AssertionError('因子日期无法映射到交易日历')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("因子日期无法映射到交易日历")
    # 调用 `factor_dates.min` 汇总当前数据，并将统计量保存到 `first_price_date`。
    first_price_date = factor_dates.min()
    # 计算 `int(factor_positions[-1]) + price_start_offset + max(periods)`，并将数值结果保存到 `final_price_position`。
    final_price_position = (
        int(factor_positions[-1]) + price_start_offset + max(periods)
    )
    # 检查 `final_price_position >= len(calendar_dates)`；根据条件结果选择对应处理分支。
    if final_price_position >= len(calendar_dates):
        # 抛出 `ValueError('交易日历不足以覆盖最大收益期限')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("交易日历不足以覆盖最大收益期限")
    # 读取 `calendar_dates[final_price_position]` 的当前值，并赋给 `last_price_date`。
    last_price_date = calendar_dates[final_price_position]
    # 读取 `calendar_dates[calendar_dates.get_loc(first_price_date):final_price_position + 1]` 的当前值，并赋给 `price_calendar`。
    price_calendar = calendar_dates[
        calendar_dates.get_loc(first_price_date) : final_price_position + 1
    ]
    # 调用 `pd.Index` 构造带名称的一维 Pandas 索引，并将结果保存到 `assets`。
    assets = pd.Index(
        sorted(factor.index.get_level_values("asset").unique()), name="asset"
    )
    # 调用 `pd.Series(assets).str.extract` 按正则捕获组拆分字符串字段，并将结果保存到 `wind_codes`。
    wind_codes = pd.Series(assets).str.extract(
        r"^(?P<exchange>SH|SZ)\.(?P<number>\d{6})$"
    )
    # 检查 `wind_codes.isna().any(axis=None)`；根据条件结果选择对应处理分支。
    if wind_codes.isna().any(axis=None):
        # 抛出 `ValueError('因子股票代码格式异常')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("因子股票代码格式异常")
    # 调用 `set` 整理数据结构，并将结果保存到 `needed_wind_codes`。
    needed_wind_codes = set(wind_codes["number"] + "." + wind_codes["exchange"])
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `eod`。
    eod = pd.read_parquet(
        EOD_PATH,
        columns=["s_info_windcode", "trade_dt", price_column],
        filters=[
            ("trade_dt", ">=", first_price_date.strftime("%Y%m%d")),
            ("trade_dt", "<=", last_price_date.strftime("%Y%m%d")),
        ],
    )
    # 调用 `normalize_dates` 按当前截面统计量完成缩尾或标准化处理，并将结果保存到 `eod['trade_dt']`。
    eod["trade_dt"] = normalize_dates(eod["trade_dt"])
    # 调用 `eod['s_info_windcode'].astype` 统一设备或数据类型，并将结果保存到 `eod['s_info_windcode']`。
    eod["s_info_windcode"] = eod["s_info_windcode"].astype(str)
    # 复制 `eod.loc[eod['s_info_windcode'].isin(needed_wind_codes)].copy`，避免后续更新污染原对象，并保存到 `eod`。
    eod = eod.loc[eod["s_info_windcode"].isin(needed_wind_codes)].copy()
    # 调用 `eod['s_info_windcode'].str.extract` 按正则捕获组拆分字符串字段，并将结果保存到 `parts`。
    parts = eod["s_info_windcode"].str.extract(
        r"^(?P<number>\d{6})\.(?P<exchange>SH|SZ)$"
    )
    # 计算 `parts['exchange'] + '.' + parts['number']`，并将数值结果保存到 `eod['asset']`。
    eod["asset"] = parts["exchange"] + "." + parts["number"]
    # 调用 `pd.to_datetime` 把日期字段转换为 Pandas 时间类型，并将结果保存到 `eod['date']`。
    eod["date"] = pd.to_datetime(eod["trade_dt"], format="%Y%m%d")
    # 检查 `eod.duplicated(['date', 'asset']).any()`；根据条件结果选择对应处理分支。
    if eod.duplicated(["date", "asset"]).any():
        # 抛出 `ValueError('Wind 复权收盘价业务键重复')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("Wind 复权收盘价业务键重复")
    # 计算 `np.isfinite(eod[price_column].to_numpy(np.float64)) & eod[price_column].gt(0).to_numpy()`，并将数值结果保存到 `valid_price`。
    valid_price = (
        np.isfinite(eod[price_column].to_numpy(np.float64))
        & eod[price_column].gt(0).to_numpy()
    )
    # 读取 `np.nan` 的当前值，并赋给 `eod.loc[~valid_price, price_column]`。
    eod.loc[~valid_price, price_column] = np.nan
    # 调用 `astype` 把数组转换为指定数据类型，并将结果保存到 `prices`。
    prices = (
        eod.pivot(index="date", columns="asset", values=price_column)
        .reindex(index=price_calendar, columns=assets)
        .astype(np.float64)
    )
    # 组织一组有序值，并保存到 `(prices.index.name, prices.columns.name)`。
    prices.index.name, prices.columns.name = "date", "asset"

    # 调用 `factor.index.to_frame` 把索引展开为普通 DataFrame 列，并将结果保存到 `factor_keys`。
    factor_keys = factor.index.to_frame(index=False)
    # 调用 `prices.index.get_indexer` 把目标标签映射为索引位置，并将结果保存到 `start_rows`。
    start_rows = prices.index.get_indexer(factor_keys["date"])
    # 调用 `prices.columns.get_indexer` 把目标标签映射为索引位置，并将结果保存到 `asset_columns`。
    asset_columns = prices.columns.get_indexer(factor_keys["asset"])
    # 检查 `(start_rows < 0).any() or (asset_columns < 0).any()`；根据条件结果选择对应处理分支。
    if (start_rows < 0).any() or (asset_columns < 0).any():
        # 抛出 `AssertionError('因子键无法映射到 T 日价格矩阵')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("因子键无法映射到 T 日价格矩阵")
    # 调用 `prices.to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `price_values`。
    price_values = prices.to_numpy(copy=False)
    # 调用 `np.ones` 按目标形状初始化数值容器 `all_prices`。
    all_prices = np.ones(len(factor), dtype=bool)
    # 构造当前步骤所需的列表，并保存到 `coverage_records`。
    coverage_records = []
    # 组织一组有序值，并保存到 `required_offsets`。
    required_offsets = (
        price_start_offset,
        *(price_start_offset + period for period in periods),
    )
    # 遍历 `required_offsets`，并把当前元素绑定到 `offset` 逐项处理。
    for offset in required_offsets:
        # 读取 `price_values[start_rows + offset, asset_columns]` 的当前值，并赋给 `selected`。
        selected = price_values[start_rows + offset, asset_columns]
        # 计算 `np.isfinite(selected) & (selected > 0)`，并将数值结果保存到 `available`。
        available = np.isfinite(selected) & (selected > 0)
        # 用 `available` 对 `all_prices` 做 `BitAnd` 累积更新。
        all_prices &= available
        # 把本轮结果追加到 `coverage_records`，累积后续需要汇总的记录。
        coverage_records.append(
            {
                "price_point": price_point_label(offset),
                "available_rows": int(available.sum()),
                "missing_rows": int((~available).sum()),
                "coverage": float(available.mean()),
            }
        )
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `price_coverage`。
    price_coverage = pd.DataFrame(coverage_records)
    # 检查 `1 - float(all_prices.mean()) > max_loss`；根据条件结果选择对应处理分支。
    if 1 - float(all_prices.mean()) > max_loss:
        # 抛出 `AssertionError('复权收盘价缺失比例超过 MAX_LOSS=0.001')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("复权收盘价缺失比例超过 MAX_LOSS=0.001")
    # 调用 `np.flatnonzero` 返回非零元素的一维位置，并将结果保存到 `valid_price_positions`。
    valid_price_positions = np.flatnonzero(all_prices)
    # 复制 `factor.iloc[valid_price_positions].copy`，避免后续更新污染原对象，并保存到 `factor`。
    factor = factor.iloc[valid_price_positions].copy()
    # 复制 `factor_quantile.iloc[valid_price_positions].copy`，避免后续更新污染原对象，并保存到 `factor_quantile`。
    factor_quantile = factor_quantile.iloc[valid_price_positions].copy()
    # 复制 `can_enter_t1.iloc[valid_price_positions].copy`，避免后续更新污染原对象，并保存到 `can_enter_t1`。
    can_enter_t1 = can_enter_t1.iloc[valid_price_positions].copy()
    # 调用 `len` 汇总当前数据，并将统计量保存到 `t_grouping_rows_after_price_filter`。
    t_grouping_rows_after_price_filter = len(factor)

    # 保留完整交易日 level，明确 period 是交易日偏移且起点就是 T 日。
    analysis_calendar = calendar_dates[
        (calendar_dates >= factor_dates.min()) & (calendar_dates <= factor_dates.max())
    ]
    # 调用 `analysis_calendar.get_indexer` 把目标标签映射为索引位置，并将结果保存到 `factor_date_codes`。
    factor_date_codes = analysis_calendar.get_indexer(
        factor.index.get_level_values("date")
    )
    # 调用 `assets.get_indexer` 把目标标签映射为索引位置，并将结果保存到 `factor_asset_codes`。
    factor_asset_codes = assets.get_indexer(factor.index.get_level_values("asset"))
    # 调用 `pd.MultiIndex` 由日期和股票编码构造完整多级索引，并将结果保存到 `factor_index`。
    factor_index = pd.MultiIndex(
        levels=[analysis_calendar, assets],
        codes=[factor_date_codes, factor_asset_codes],
        names=["date", "asset"],
        verify_integrity=True,
    )
    # 读取 `factor_index` 的当前值，并赋给 `factor.index`。
    factor.index = factor_index
    # 读取 `factor_index` 的当前值，并赋给 `factor_quantile.index`。
    factor_quantile.index = factor_index
    # 读取 `factor_index` 的当前值，并赋给 `can_enter_t1.index`。
    can_enter_t1.index = factor_index

    # 将价格矩阵按模式平移，但因子索引始终保留信号日 T。
    return_prices = build_return_price_view(prices, str(mode["name"]))
    # 在 `warnings.catch_warnings()` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with warnings.catch_warnings():
        # 调用 `warnings.filterwarnings` 根据有效性条件筛选当前样本或构造掩码。
        warnings.filterwarnings("ignore", category=FutureWarning)
        # 调用 `al.utils.compute_forward_returns` 按函数定义的数值口径计算当前统计量，并将结果保存到 `factor_data`。
        factor_data = al.utils.compute_forward_returns(
            factor=factor,
            prices=return_prices,
            periods=periods,
            filter_zscore=None,
            cumulative_returns=True,
        )
    # 检查 `not factor_data.index.equals(factor.index)`；根据条件结果选择对应处理分支。
    if not factor_data.index.equals(factor.index):
        # 抛出 `AssertionError('Alphalens 收益键与 T 日因子键不一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("Alphalens 收益键与 T 日因子键不一致")
    # 调用 `factor.to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `factor_data['factor']`。
    factor_data["factor"] = factor.to_numpy(np.float64)
    # 调用 `factor_quantile.to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `factor_data['factor_quantile']`。
    factor_data["factor_quantile"] = factor_quantile.to_numpy(np.int64)
    # 调用 `tuple` 整理数据结构，并将结果保存到 `forward_columns`。
    forward_columns = tuple(f"{period}D" for period in periods)
    # 调用 `set((*forward_columns, 'factor', 'factor_quantile')).difference` 计算两个集合的差集以找出缺失字段，并将结果保存到 `missing_columns`。
    missing_columns = set((*forward_columns, "factor", "factor_quantile")).difference(
        factor_data.columns
    )
    # 检查 `missing_columns`；根据条件结果选择对应处理分支。
    if missing_columns:
        # 抛出 `AssertionError(f'Alphalens 缺少字段：{sorted(missing_columns)}')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(f"Alphalens 缺少字段：{sorted(missing_columns)}")
    # 检查 `not np.isfinite(factor_data[[*forward_columns, 'factor']].to_numpy(np.float64)).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(
        factor_data[[*forward_columns, "factor"]].to_numpy(np.float64)
    ).all():
        # 抛出 `AssertionError('Alphalens 输出含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("Alphalens 输出含 NaN 或 Inf")

    # IC 和基准始终使用完整 T 日截面；只有口径1的组合与换手应用 T+1 掩码。
    execution_factor_data = select_execution_factor_data(
        factor_data, can_enter_t1, str(mode["name"])
    )

    # 必须保留完整交易日 level：Alphalens 从 level.freq 获取交易日频率。
    # remove_unused_levels() 会在存在缺失因子日时把频率清成 None，继而
    # 让 asfreq(None) 退化为自然日并插入周末 NaN，使 22 日滚动均线全空。
    date_level = factor_data.index.names.index("date")
    # 读取 `factor_data.index.levels[date_level].freq` 的当前值，并赋给 `factor_calendar_frequency`。
    factor_calendar_frequency = factor_data.index.levels[date_level].freq
    # 读取 `execution_factor_data.index.levels[date_level].freq` 的当前值，并赋给 `execution_calendar_frequency`。
    execution_calendar_frequency = execution_factor_data.index.levels[date_level].freq
    # 检查 `factor_calendar_frequency is None`；根据条件结果选择对应处理分支。
    if factor_calendar_frequency is None:
        # 抛出 `AssertionError('因子日期 level 缺少交易日频率')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("因子日期 level 缺少交易日频率")
    # 检查 `execution_calendar_frequency != factor_calendar_frequency`；根据条件结果选择对应处理分支。
    if execution_calendar_frequency != factor_calendar_frequency:
        # 抛出 `AssertionError('完整因子与可交易视图的交易日频率不一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("完整因子与可交易视图的交易日频率不一致")
    # 读取 `type(factor_calendar_frequency).__name__` 的当前值，并赋给 `factor_calendar_frequency_name`。
    factor_calendar_frequency_name = type(factor_calendar_frequency).__name__
    # 调用 `len` 汇总当前数据，并将统计量保存到 `portfolio_rows_after_execution_filter`。
    portfolio_rows_after_execution_filter = len(execution_factor_data)
    # 调用 `reindex` 按完整目标索引重排并显式暴露缺项，并将结果保存到 `execution_quantile_counts`。
    execution_quantile_counts = (
        execution_factor_data.groupby(
            [pd.Grouper(level="date"), "factor_quantile"], observed=True
        ).size().unstack("factor_quantile")
        .reindex(columns=range(1, quantiles + 1))
    )
    # 检查 `execution_quantile_counts.isna().any(axis=None)`；根据条件结果选择对应处理分支。
    if execution_quantile_counts.isna().any(axis=None):
        # 抛出 `AssertionError('执行过滤后存在空分位组，无法计算组合收益')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("执行过滤后存在空分位组，无法计算组合收益")

    # 抽样手算，确认 Alphalens 与所选价格起点完全一致。
    audit_positions = np.unique(
        np.linspace(0, len(factor_data) - 1, min(25, len(factor_data)), dtype=np.int64)
    )
    # 读取 `factor_data.index[audit_positions]` 的当前值，并赋给 `audit_index`。
    audit_index = factor_data.index[audit_positions]
    # 调用 `audit_index.to_frame` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `audit_keys`。
    audit_keys = audit_index.to_frame(index=False)
    # 调用 `prices.index.get_indexer` 把目标标签映射为索引位置，并将结果保存到 `audit_rows`。
    audit_rows = prices.index.get_indexer(audit_keys["date"])
    # 调用 `prices.columns.get_indexer` 把目标标签映射为索引位置，并将结果保存到 `audit_columns`。
    audit_columns = prices.columns.get_indexer(audit_keys["asset"])
    # 构造当前步骤所需的列表，并保存到 `formula_records`。
    formula_records = []
    # 遍历 `zip(periods, forward_columns)`，并把当前元素绑定到 `(period, column)` 逐项处理。
    for period, column in zip(periods, forward_columns):
        # 计算 `price_values[audit_rows + price_start_offset + period, audit_columns] / price_values[audit_rows…`，并将数值结果保存到 `manual`。
        manual = (
            price_values[
                audit_rows + price_start_offset + period, audit_columns
            ]
            / price_values[audit_rows + price_start_offset, audit_columns]
            - 1
        )
        # 调用 `factor_data.loc[audit_index, column].to_numpy` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `library`。
        library = factor_data.loc[audit_index, column].to_numpy(np.float64)
        # 调用 `np.testing.assert_allclose` 比较两组数值是否在给定误差范围内一致。
        np.testing.assert_allclose(library, manual, rtol=1e-12, atol=1e-12)
        # 把本轮结果追加到 `formula_records`，累积后续需要汇总的记录。
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

    # 调用 `sort_index` 按索引键排序以保证时间和股票顺序稳定，并将结果保存到 `daily_ic`。
    daily_ic = al.performance.factor_information_coefficient(
        factor_data, group_adjust=False, by_group=False
    ).sort_index()
    # 构造键值映射，并保存到 `ic_nan_rows_by_period`，供后续按键查询。
    ic_nan_rows_by_period = {
        column: int(daily_ic[column].isna().sum()) for column in forward_columns
    }
    # 检查 `daily_ic.index.freq is None`；根据条件结果选择对应处理分支。
    if daily_ic.index.freq is None:
        # 抛出 `AssertionError('日度 IC 索引缺少交易日频率')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("日度 IC 索引缺少交易日频率")
    # 检查 `daily_ic.index.freq != factor_calendar_frequency`；根据条件结果选择对应处理分支。
    if daily_ic.index.freq != factor_calendar_frequency:
        # 抛出 `AssertionError('日度 IC 与因子数据的交易日频率不一致')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("日度 IC 与因子数据的交易日频率不一致")
    # 调用 `daily_ic.rolling(window=22).mean` 汇总当前数据，并将统计量保存到 `ic_rolling_22`。
    ic_rolling_22 = daily_ic.rolling(window=22).mean()
    # 构造键值映射，并保存到 `ic_rolling_22_valid_rows_by_period`，供后续按键查询。
    ic_rolling_22_valid_rows_by_period = {
        column: int(ic_rolling_22[column].notna().sum())
        for column in forward_columns
    }
    # 检查 `any((count == 0 for count in ic_rolling_22_valid_rows_by_period.values()))`；根据条件结果选择对应处理分支。
    if any(count == 0 for count in ic_rolling_22_valid_rows_by_period.values()):
        # 抛出 `AssertionError('1 month moving avg 没有有效点')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("1 month moving avg 没有有效点")

    # Alphalens 0.4.6 的均值/标准差会自动跳过 NaN，但 SciPy 统计量
    # 默认传播 NaN。这里保持原表格结构，只把后四项改为逐列忽略缺失值。
    def plot_information_table_omit_nan(ic_data: Any, return_df: Any=False) -> Any:
        """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。

        输入：
            ic_data: Any。
            return_df: Any。
        输出：
            Any。
        """
        # 调用 `ic_data.to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `values`。
        values = ic_data.to_numpy(np.float64)
        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `table`。
        table = pd.DataFrame(index=ic_data.columns)
        # 调用 `ic_data.mean` 汇总当前数据，并将统计量保存到 `table['IC Mean']`。
        table["IC Mean"] = ic_data.mean()
        # 调用 `ic_data.std` 汇总当前数据，并将统计量保存到 `table['IC Std.']`。
        table["IC Std."] = ic_data.std()
        # 计算 `table['IC Mean'] / table['IC Std.']`，并将数值结果保存到 `table['Risk-Adjusted IC']`。
        table["Risk-Adjusted IC"] = table["IC Mean"] / table["IC Std."]
        # 调用 `stats.ttest_1samp` 对日度指标执行均值为零的单样本 t 检验，并将结果保存到 `test`。
        test = stats.ttest_1samp(
            values, popmean=0.0, axis=0, nan_policy="omit"
        )
        # 读取 `test.statistic` 的当前值，并赋给 `table['t-stat(IC)']`。
        table["t-stat(IC)"] = test.statistic
        # 读取 `test.pvalue` 的当前值，并赋给 `table['p-value(IC)']`。
        table["p-value(IC)"] = test.pvalue
        # 调用 `stats.skew` 计算样本分布偏度，并将结果保存到 `table['IC Skew']`。
        table["IC Skew"] = stats.skew(values, axis=0, nan_policy="omit")
        # 调用 `stats.kurtosis` 计算样本分布峰度，并将结果保存到 `table['IC Kurtosis']`。
        table["IC Kurtosis"] = stats.kurtosis(
            values, axis=0, nan_policy="omit"
        )
        # 检查 `return_df`；根据条件结果选择对应处理分支。
        if return_df:
            # 将 `table` 作为当前函数结果返回给调用方。
            return table
        # 调用 `print` 记录当前流程状态或诊断信息。
        print("Information Analysis")
        # 调用 `al.utils.print_table` 执行 `print_table` 对应的明确变换，并产生后续步骤使用的结果。
        al.utils.print_table(table.apply(lambda column: column.round(3)).T)

    # 组织一组有序值，并保存到 `(ic_records, annual_records)`。
    ic_records, annual_records = [], []
    # 遍历 `forward_columns`，并把当前元素绑定到 `column` 逐项处理。
    for column in forward_columns:
        # 调用 `daily_ic[column].dropna` 剔除缺失观测后再计算统计量，并将结果保存到 `values`。
        values = daily_ic[column].dropna()
        # 调用 `stats.ttest_1samp` 对日度指标执行均值为零的单样本 t 检验，并将结果保存到 `test`。
        test = stats.ttest_1samp(values.to_numpy(), popmean=0.0)
        # 把本轮结果追加到 `ic_records`，累积后续需要汇总的记录。
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
        # 遍历 `values.groupby(values.index.year)`，并把当前元素绑定到 `(year, annual)` 逐项处理。
        for year, annual in values.groupby(values.index.year):
            # 把本轮结果追加到 `annual_records`，累积后续需要汇总的记录。
            annual_records.append(
                {
                    "year": int(year), "period": column, "days": len(annual),
                    "mean_ic": float(annual.mean()),
                    "median_ic": float(annual.median()),
                    "std_ic": float(annual.std(ddof=1)),
                    "positive_ratio": float(annual.gt(0).mean()),
                }
            )
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `ic_summary`。
    ic_summary = pd.DataFrame(ic_records)
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `annual_ic`。
    annual_ic = pd.DataFrame(annual_records)
    # 调用 `reindex` 按完整目标索引重排并显式暴露缺项，并将结果保存到 `information_table_audit`。
    information_table_audit = plot_information_table_omit_nan(
        daily_ic, return_df=True
    ).reindex(forward_columns)
    # 检查 `not np.isfinite(information_table_audit.to_numpy(np.float64)).all()`；根据条件结果选择对应处理分支。
    if not np.isfinite(information_table_audit.to_numpy(np.float64)).all():
        # 抛出 `AssertionError('Information Analysis 统计量仍含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("Information Analysis 统计量仍含 NaN 或 Inf")
    # 调用 `np.testing.assert_allclose` 比较两组数值是否在给定误差范围内一致。
    np.testing.assert_allclose(
        information_table_audit["t-stat(IC)"].to_numpy(np.float64),
        ic_summary["t_stat"].to_numpy(np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    # 调用 `np.testing.assert_allclose` 比较两组数值是否在给定误差范围内一致。
    np.testing.assert_allclose(
        information_table_audit["p-value(IC)"].to_numpy(np.float64),
        ic_summary["p_value"].to_numpy(np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    # 调用 `al.performance.mean_return_by_quantile` 执行 `mean_return_by_quantile` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `(mean_by_quantile, _)`。
    mean_by_quantile, _ = al.performance.mean_return_by_quantile(
        execution_factor_data,
        by_date=False,
        by_group=False,
        demeaned=False,
        group_adjust=False,
    )
    # 调用 `al.performance.mean_return_by_quantile` 执行 `mean_return_by_quantile` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `(mean_by_date, standard_error_by_date)`。
    mean_by_date, standard_error_by_date = al.performance.mean_return_by_quantile(
        execution_factor_data,
        by_date=True,
        by_group=False,
        demeaned=False,
        group_adjust=False,
    )
    # 调用 `al.performance.compute_mean_returns_spread` 按函数定义的数值口径计算当前统计量，并将结果保存到 `(daily_spread, _)`。
    daily_spread, _ = al.performance.compute_mean_returns_spread(
        mean_by_date, upper_quant=quantiles, lower_quant=1,
        std_err=standard_error_by_date,
    )
    # 调用 `mean_by_quantile.reset_index().rename` 把完整临时产物替换为正式输出，并将结果保存到 `quantile_return_summary`。
    quantile_return_summary = mean_by_quantile.reset_index().rename(
        columns={"factor_quantile": "quantile"}
    )
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `spread_summary`。
    spread_summary = pd.DataFrame(
        {
            "period": forward_columns,
            "Q5_minus_Q1_return": [float(daily_spread[column].mean()) for column in forward_columns],
            "standard_error": [float(daily_spread[column].sem(ddof=1)) for column in forward_columns],
        }
    )
    # 调用 `mean_by_date['1D'].unstack('factor_quantile').sort_index` 根据分数计算截面排序或排序索引，并将结果保存到 `daily_quantile_returns`。
    daily_quantile_returns = mean_by_date["1D"].unstack("factor_quantile").sort_index()
    # 调用 `compute_equal_weight_market_returns` 按函数定义的数值口径计算当前统计量，并将结果保存到 `equal_weight_market`。
    equal_weight_market = compute_equal_weight_market_returns(factor_data)
    # 调用 `pd.concat` 拼接各批结果，并将合并值保存到 `portfolio_returns`。
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
    # 检查 `portfolio_returns.isna().any(axis=None)`；根据条件结果选择对应处理分支。
    if portfolio_returns.isna().any(axis=None):
        # 抛出 `AssertionError('组合收益与全市场等权基准日期未完全对齐')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("组合收益与全市场等权基准日期未完全对齐")
    # 调用 `compute_relative_performance` 按函数定义的数值口径计算当前统计量，并将结果保存到 `relative_performance`。
    relative_performance = compute_relative_performance(portfolio_returns)
    # 调用 `pd.concat` 拼接各批结果，并将合并值保存到 `pyfolio_performance`。
    pyfolio_performance = pd.concat(
        {
            column: pf.timeseries.perf_stats(portfolio_returns[column])
            for column in portfolio_returns
        },
        axis=1,
    )

    # 构造当前步骤所需的列表，并保存到 `package_names`。
    package_names = [
        "alphalens-reloaded", "pyfolio-reloaded", "pandas", "numpy", "pyarrow"
    ]
    # 构造键值映射，并保存到 `installed_versions`，供后续按键查询。
    installed_versions = {
        package: metadata.version(package) for package in package_names
    }
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `final_audit`。
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

    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(prediction_audit, evaluation_dir / "prediction_universe_audit.parquet")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(price_coverage, evaluation_dir / "price_coverage.parquet")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(pd.DataFrame(formula_records), evaluation_dir / "return_formula_audit.parquet")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(ic_summary, evaluation_dir / "ic_summary.parquet")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(annual_ic, evaluation_dir / "annual_ic_summary.parquet")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(quantile_return_summary, evaluation_dir / "quantile_return_summary.parquet")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(spread_summary, evaluation_dir / "spread_summary.parquet")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(portfolio_returns.reset_index(), evaluation_dir / "portfolio_returns.parquet")
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(
        pyfolio_performance.rename_axis("metric").reset_index(),
        evaluation_dir / "pyfolio_performance.parquet",
    )
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(
        relative_performance, evaluation_dir / "relative_performance.parquet"
    )
    # 调用 `atomic_parquet` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_parquet(final_audit, evaluation_dir / "final_audit.parquet")
    # 构造键值映射，并保存到 `evaluation_manifest`，供后续按键查询。
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

    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(prediction_audit)
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(price_coverage.style.format({"coverage": "{:.6%}"}))
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(pd.DataFrame(formula_records))
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(ic_summary)
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(annual_ic)
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(quantile_return_summary)
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(spread_summary)
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(pyfolio_performance)
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(relative_performance)
    # 调用 `display` 在 Notebook 输出区展示当前表格或图形。
    display(final_audit)
    # 计算 `evaluation_dir / 'pic'`，并将数值结果保存到 `pic_dir`。
    pic_dir = evaluation_dir / "pic"
    # 调用 `pic_dir.mkdir` 创建输出目录，确保后续文件可以安全写入。
    pic_dir.mkdir(parents=True, exist_ok=True)

    # 调用 `(1 + portfolio_returns).cumprod` 对日收益逐日连乘得到累计净值，并将结果保存到 `portfolio_net_values`。
    portfolio_net_values = (1 + portfolio_returns).cumprod()
    # 按 `price_start_offset == 0` 在两个候选值间选择，并保存到 `price_start_title`。
    price_start_title = "T" if price_start_offset == 0 else "T+1"
    # 调用 `portfolio_net_values.plot` 绘制当前时间序列或截面统计图，并将结果保存到 `cumulative_ax`。
    cumulative_ax = portfolio_net_values.plot(
        figsize=(12, 5),
        title=f"A2 因子组合与全市场等权累计毛收益（{price_start_title} 日复权收盘起算）",
    )
    # 调用 `cumulative_ax.grid` 为坐标轴添加辅助网格。
    cumulative_ax.grid(alpha=0.25)
    # 调用 `cumulative_ax.get_figure` 取得坐标轴所属图形对象，并将结果保存到 `cumulative_figure`。
    cumulative_figure = cumulative_ax.get_figure()
    # 计算 `evaluation_dir / 'portfolio_cumulative_returns.png'`，并将数值结果保存到 `cumulative_path`。
    cumulative_path = evaluation_dir / "portfolio_cumulative_returns.png"
    # 调用 `cumulative_figure.savefig` 把当前图形写入目标图片文件。
    cumulative_figure.savefig(cumulative_path, dpi=150, bbox_inches="tight")
    # 调用 `plt.show` 在当前交互环境展示已生成图形。
    plt.show()
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"累计收益图：{cumulative_path}")
    # 调用 `plt.close` 释放当前步骤占用的资源。
    plt.close(cumulative_figure)

    # 回撤按净值相对历史峰值计算；图例同时标出各组合最大回撤。
    portfolio_drawdowns = portfolio_net_values.div(
        portfolio_net_values.cummax().clip(lower=1.0)
    ).sub(1)
    # 调用 `np.testing.assert_allclose` 比较两组数值是否在给定误差范围内一致。
    np.testing.assert_allclose(
        portfolio_drawdowns.min().to_numpy(np.float64),
        pyfolio_performance.loc["Max drawdown"].to_numpy(np.float64),
        rtol=1e-12,
        atol=1e-12,
    )
    # 复制 `portfolio_drawdowns.copy`，避免后续更新污染原对象，并保存到 `drawdown_plot`。
    drawdown_plot = portfolio_drawdowns.copy()
    # 构造当前步骤所需的列表，并保存到 `drawdown_plot.columns`。
    drawdown_plot.columns = [
        f"{column}（最大回撤 {portfolio_drawdowns[column].min():.2%}）"
        for column in portfolio_drawdowns
    ]
    # 调用 `drawdown_plot.plot` 绘制当前时间序列或截面统计图，并将结果保存到 `drawdown_ax`。
    drawdown_ax = drawdown_plot.plot(
        figsize=(12, 5), title="A2 因子组合回撤曲线"
    )
    # 调用 `drawdown_ax.axhline` 绘制零基准水平线。
    drawdown_ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    # 调用 `drawdown_ax.set_ylabel` 设置图形纵轴含义。
    drawdown_ax.set_ylabel("回撤")
    # 调用 `drawdown_ax.yaxis.set_major_formatter` 设置坐标轴主刻度的百分比显示格式。
    drawdown_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    # 调用 `drawdown_ax.grid` 为坐标轴添加辅助网格。
    drawdown_ax.grid(alpha=0.25)
    # 调用 `drawdown_ax.get_figure` 取得坐标轴所属图形对象，并将结果保存到 `drawdown_figure`。
    drawdown_figure = drawdown_ax.get_figure()
    # 计算 `pic_dir / 'portfolio_drawdown.png'`，并将数值结果保存到 `drawdown_path`。
    drawdown_path = pic_dir / "portfolio_drawdown.png"
    # 调用 `drawdown_figure.savefig` 把当前图形写入目标图片文件。
    drawdown_figure.savefig(drawdown_path, dpi=150, bbox_inches="tight")
    # 调用 `plt.show` 在当前交互环境展示已生成图形。
    plt.show()
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"最大回撤图：{drawdown_path}")
    # 调用 `plt.close` 释放当前步骤占用的资源。
    plt.close(drawdown_figure)

    # Alphalens 在每个 tear sheet 的 show() 之后立即关闭 figure；
    # 因此在 show 时先保存，再继续让 Notebook 正常内嵌显示。
    for old_path in pic_dir.glob("alphalens_*_tear_sheet*.png"):
        # 调用 `old_path.unlink` 解析当前 run 对应的数据与输出路径。
        old_path.unlink()
    # 组织一组有序值，并保存到 `tear_sheet_sections`。
    tear_sheet_sections = ("returns", "information", "turnover")
    # 构造当前步骤所需的列表，并保存到 `tear_sheet_files`。
    tear_sheet_files: list[str] = []
    # 读取 `plt.show` 的当前值，并赋给 `original_show`。
    original_show = plt.show
    # 读取 `al.plotting.plot_information_table` 的当前值，并赋给 `original_information_table`。
    original_information_table = al.plotting.plot_information_table
    # 设置 `section_number` 的固定配置值，供后续流程统一使用。
    section_number = 0

    # 定义 `save_tear_sheet_figures` 函数，封装这一阶段可独立调用的处理步骤。
    def save_tear_sheet_figures(*show_args: Any, **show_kwargs: Any) -> None:
        """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。

        输入：
            show_args: Any。
            show_kwargs: Any。
        输出：
            None。
        """
        # 声明 `section_number` 复用外层函数状态，供闭包更新。
        nonlocal section_number
        # 构造当前步骤所需的列表，并保存到 `figures`。
        figures = []
        # 遍历 `plt.get_fignums()`，并把当前元素绑定到 `number` 逐项处理。
        for number in plt.get_fignums():
            # 调用 `plt.figure` 激活指定编号的 Matplotlib 图形，并将结果保存到 `figure`。
            figure = plt.figure(number)
            # 检查 `figure.axes`；根据条件结果选择对应处理分支。
            if figure.axes:
                # 把本轮结果追加到 `figures`，累积后续需要汇总的记录。
                figures.append(figure)
        # 检查 `not figures`；根据条件结果选择对应处理分支。
        if not figures:
            # 抛出 `RuntimeError('Alphalens tear sheet 未生成 figure')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError("Alphalens tear sheet 未生成 figure")
        # 按 `section_number < len(tear_sheet_sections)` 在两个候选值间选择，并保存到 `section`。
        section = (
            tear_sheet_sections[section_number]
            if section_number < len(tear_sheet_sections)
            else f"section_{section_number + 1}"
        )
        # 遍历 `enumerate(figures, start=1)`，并把当前元素绑定到 `(figure_number, figure)` 逐项处理。
        for figure_number, figure in enumerate(figures, start=1):
            # 按 `len(figures) == 1` 在两个候选值间选择，并保存到 `suffix`。
            suffix = "" if len(figures) == 1 else f"_{figure_number}"
            # 计算 `pic_dir / f'alphalens_{section}_tear_sheet{suffix}.png'`，并将数值结果保存到 `path`。
            path = pic_dir / f"alphalens_{section}_tear_sheet{suffix}.png"
            # 调用 `figure.savefig` 把当前图形按指定分辨率写入图片文件。
            figure.savefig(path, dpi=150, bbox_inches="tight")
            # 把本轮结果追加到 `tear_sheet_files`，累积后续需要汇总的记录。
            tear_sheet_files.append(str(path.relative_to(evaluation_dir)))
        # 调用 `original_show` 执行 `original_show` 对应的明确变换，并产生后续步骤使用的结果。
        original_show(*show_args, **show_kwargs)
        # 用 `1` 对 `section_number` 做 `Add` 累积更新。
        section_number += 1

    # 调用 `plt.close` 释放当前步骤占用的资源。
    plt.close("all")
    # 读取 `save_tear_sheet_figures` 的当前值，并赋给 `plt.show`。
    plt.show = save_tear_sheet_figures
    # 读取 `plot_information_table_omit_nan` 的当前值，并赋给 `al.plotting.plot_information_table`。
    al.plotting.plot_information_table = plot_information_table_omit_nan
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        # information 始终看完整 T 日截面；returns/turnover 使用模式对应执行视图。
        al.plotting.plot_quantile_statistics_table(factor_data)
        # 调用 `al.tears.create_returns_tear_sheet` 根据当前配置和输入构造后续流程所需对象。
        al.tears.create_returns_tear_sheet(
            execution_factor_data,
            long_short=True,
            group_neutral=False,
            by_group=False,
        )
        # 调用 `al.tears.create_information_tear_sheet` 根据当前配置和输入构造后续流程所需对象。
        al.tears.create_information_tear_sheet(
            factor_data,
            group_neutral=False,
            by_group=False,
        )
        # 调用 `al.tears.create_turnover_tear_sheet` 根据当前配置和输入构造后续流程所需对象。
        al.tears.create_turnover_tear_sheet(execution_factor_data)
    # 无论前序步骤是否抛出异常，都执行本分支完成必要的收尾。
    finally:
        # 读取 `original_show` 的当前值，并赋给 `plt.show`。
        plt.show = original_show
        # 读取 `original_information_table` 的当前值，并赋给 `al.plotting.plot_information_table`。
        al.plotting.plot_information_table = original_information_table
        # 调用 `plt.close` 释放当前步骤占用的资源。
        plt.close("all")
    # 检查 `section_number != len(tear_sheet_sections)`；根据条件结果选择对应处理分支。
    if section_number != len(tear_sheet_sections):
        # 抛出 `AssertionError(f'Alphalens tear sheet 数量异常：{section_number}')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(
            f"Alphalens tear sheet 数量异常：{section_number}"
        )
    # 构造当前步骤所需的列表，并保存到 `evaluation_manifest['plot_files']`。
    evaluation_manifest["plot_files"] = [
        str(cumulative_path.relative_to(evaluation_dir)),
        str(drawdown_path.relative_to(evaluation_dir)),
        *tear_sheet_files,
    ]
    # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `evaluation_manifest['drawdown_file']`。
    evaluation_manifest["drawdown_file"] = str(
        drawdown_path.relative_to(evaluation_dir)
    )
    # 读取 `tear_sheet_files` 的当前值，并赋给 `evaluation_manifest['tear_sheet_files']`。
    evaluation_manifest["tear_sheet_files"] = tear_sheet_files
    # 构造键值映射，并保存到 `evaluation_manifest['tear_sheet_scopes']`，供后续按键查询。
    evaluation_manifest["tear_sheet_scopes"] = {
        "returns": str(mode["portfolio_scope"]),
        "information": "T_full_cross_section_with_valid_forward_prices",
        "turnover": str(mode["portfolio_scope"]),
    }
    # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_json(
        evaluation_manifest,
        evaluation_dir / "factor_evaluation_manifest.json",
    )
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"Alphalens tear sheet：{pic_dir}")

    # 删除 `factor, factor_quantile, can_enter_t1`，释放不再使用的引用。
    del factor, factor_quantile, can_enter_t1
    # 删除 `factor_data, execution_factor_data`，释放不再使用的引用。
    del factor_data, execution_factor_data
    # 删除 `prices, return_prices, price_values, eod`，释放不再使用的引用。
    del prices, return_prices, price_values, eod
    # 调用 `gc.collect` 触发 Python 垃圾回收以释放失去引用的对象。
    gc.collect()
    # 调用 `generate_experiment_report` 根据当前结果生成可审计的报告或展示产物。
    generate_experiment_report(run_name=run_name, data_root=root)
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(
        f"样本外因子评价完成；模式={mode['name']}，"
        f"价格起点={price_point_label(price_start_offset)}。"
    )
    # 将 `final_audit` 作为当前函数结果返回给调用方。
    return final_audit


# ============================ 实验报告 ============================
def _markdown_table(frame: pd.DataFrame) -> str:
    """避免引入额外模板依赖的小型 Markdown 表格。

    输入：
        frame: pd.DataFrame。
    输出：
        str。
    """

    # 检查 `frame.empty`；根据条件结果选择对应处理分支。
    if frame.empty:
        # 将 `'（暂无记录）'` 作为当前函数结果返回给调用方。
        return "（暂无记录）"
    # 调用 `frame.copy().fillna` 用约定值填充展示表中的缺失项，并将结果保存到 `clean`。
    clean = frame.copy().fillna("")
    # 构造当前步骤所需的列表，并保存到 `headers`。
    headers = [str(column) for column in clean.columns]
    # 构造当前步骤所需的列表，并保存到 `rows`。
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    # 遍历 `clean.astype(str).itertuples(index=False, name=None)`，并把当前元素绑定到 `record` 逐项处理。
    for record in clean.astype(str).itertuples(index=False, name=None):
        # 把本轮结果追加到 `rows`，累积后续需要汇总的记录。
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in record) + " |")
    # 将 `'\n'.join(rows)` 作为当前函数结果返回给调用方。
    return "\n".join(rows)


# 定义 `generate_experiment_report` 函数，封装这一阶段可独立调用的处理步骤。
def generate_experiment_report(
    run_name: str = DEFAULT_A2_RUN_NAME,
    data_root: str | Path | None = None,
) -> Path:
    """根据指定 run 的 manifest 和审计表生成中文实验报告。

    输入：
        run_name: str。
        data_root: str | Path | None。
    输出：
        Path。
    """

    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `run_name`。
    run_name = validate_run_name(run_name)
    # 按 `data_root is not None` 在两个候选值间选择，并保存到 `root`。
    root = (
        Path(data_root).resolve()
        if data_root is not None
        else get_data_root().resolve()
    )
    # 检查 `not root.is_dir()`；根据条件结果选择对应处理分支。
    if not root.is_dir():
        # 抛出 `FileNotFoundError(root)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(root)
    # 计算 `root / 'quality' / 'runs' / run_name`，并将数值结果保存到 `quality_run`。
    quality_run = root / "quality" / "runs" / run_name
    # 调用 `read_json` 读取源数据，并将结果保存到 `universe_manifest`。
    universe_manifest = read_json(root / "universe" / "universe_manifest.json")
    # 调用 `read_json` 读取源数据，并将结果保存到 `label_manifest`。
    label_manifest = read_json(root / "labels_t10" / "labels_manifest.json")
    # 调用 `read_json` 读取源数据，并将结果保存到 `dataset_manifest`。
    dataset_manifest = read_json(root / "feature_store" / "dataset_manifest.json")
    # 计算 `root / 'feature_store' / 'rolling_split_summary.parquet'`，并将数值结果保存到 `rolling_path`。
    rolling_path = root / "feature_store" / "rolling_split_summary.parquet"
    # 检查 `universe_manifest is None or label_manifest is None or dataset_manifest is None`；根据条件结果选择对应处理分支。
    if universe_manifest is None or label_manifest is None or dataset_manifest is None:
        # 抛出 `FileNotFoundError('Universe、标签或 feature store manifest 尚未生成')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("Universe、标签或 feature store manifest 尚未生成")
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `universe_quality`。
    universe_quality = pd.read_parquet(
        root / "quality" / "universe_quality_by_date.parquet"
    )
    # 读取 `universe_quality['trade_date'].astype(str).str[:4]` 的当前值，并赋给 `universe_quality['year']`。
    universe_quality["year"] = universe_quality["trade_date"].astype(str).str[:4]
    # 调用 `universe_quality.groupby('year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `universe_yearly`。
    universe_yearly = universe_quality.groupby("year", as_index=False).agg(
        candidate_rows=("candidate_rows", "sum"),
        in_universe_t_rows=("in_universe_t_rows", "sum"),
        prediction_eligible_rows=("prediction_eligible_rows", "sum"),
        label_candidate_rows=("label_candidate_rows", "sum"),
        missing_status_t_rows=("missing_status_t_rows", "sum"),
        suspended_t_rows=("suspended_t_rows", "sum"),
        limit_up_t_rows=("limit_up_t_rows", "sum"),
        limit_down_t_rows=("limit_down_t_rows", "sum"),
    )
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `label_quality`。
    label_quality = pd.read_parquet(root / "quality" / "label_quality_by_date.parquet")
    # 读取 `label_quality['trade_date'].astype(str).str[:4]` 的当前值，并赋给 `label_quality['year']`。
    label_quality["year"] = label_quality["trade_date"].astype(str).str[:4]
    # 调用 `label_quality.groupby('year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `label_yearly`。
    label_yearly = label_quality.groupby("year", as_index=False).agg(
        label_candidate_rows=("label_candidate_rows", "sum"),
        label_valid_rows=("label_valid_rows", "sum"),
        missing_or_nonpositive_price_rows=("missing_or_nonpositive_price_rows", "sum"),
    )
    # 读取 `pd.read_parquet(rolling_path)[['forecast_year', 'train_sample_count', 'validation_sample_count'…` 的当前值，并赋给 `rolling`。
    rolling = pd.read_parquet(rolling_path)[
        ["forecast_year", "train_sample_count", "validation_sample_count", "prediction_sample_count"]
    ]
    # 计算 `quality_run / 'vit_training_summary.parquet'`，并将数值结果保存到 `training_path`。
    training_path = quality_run / "vit_training_summary.parquet"
    # 检查 `training_path.exists()`；根据条件结果选择对应处理分支。
    if training_path.exists():
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `training`。
        training = pd.read_parquet(training_path)
        # 构造键值映射，并保存到 `aggregations`，供后续按键查询。
        aggregations = {
            "completed_seeds": ("seed", "nunique"),
            "mean_best_validation_ic": ("best_validation_ic", "mean"),
            "min_best_validation_ic": ("best_validation_ic", "min"),
            "mean_best_epoch": ("best_epoch", "mean"),
        }
        # 检查 `'best_validation_rank_ic' in training.columns`；根据条件结果选择对应处理分支。
        if "best_validation_rank_ic" in training.columns:
            # 组织一组有序值，并保存到 `aggregations['mean_best_validation_rank_ic']`。
            aggregations["mean_best_validation_rank_ic"] = (
                "best_validation_rank_ic", "mean"
            )
        # 检查 `'best_validation_daily_pearson_ic' in training.columns`；根据条件结果选择对应处理分支。
        if "best_validation_daily_pearson_ic" in training.columns:
            # 组织一组有序值，并保存到 `aggregations['mean_best_validation_daily_pearson_ic']`。
            aggregations["mean_best_validation_daily_pearson_ic"] = (
                "best_validation_daily_pearson_ic", "mean"
            )
        # 检查 `'best_validation_daily_ic' in training.columns`；根据条件结果选择对应处理分支。
        elif "best_validation_daily_ic" in training.columns:
            # 组织一组有序值，并保存到 `aggregations['mean_best_validation_daily_pearson_ic']`。
            aggregations["mean_best_validation_daily_pearson_ic"] = (
                "best_validation_daily_ic", "mean"
            )
        # 检查 `'best_validation_pooled_ic' in training.columns`；根据条件结果选择对应处理分支。
        if "best_validation_pooled_ic" in training.columns:
            # 组织一组有序值，并保存到 `aggregations['mean_best_validation_pooled_ic']`。
            aggregations["mean_best_validation_pooled_ic"] = (
                "best_validation_pooled_ic", "mean"
            )
        # 调用 `training.groupby('forecast_year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `training_yearly`。
        training_yearly = training.groupby(
            "forecast_year", as_index=False
        ).agg(**aggregations)
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `training_yearly`。
        training_yearly = pd.DataFrame()
    # 计算 `quality_run / 'factor_evaluation'`，并将数值结果保存到 `evaluation_dir`。
    evaluation_dir = quality_run / "factor_evaluation"
    # 计算 `evaluation_dir / 'ic_summary.parquet'`，并将数值结果保存到 `ic_path`。
    ic_path = evaluation_dir / "ic_summary.parquet"
    # 计算 `evaluation_dir / 'spread_summary.parquet'`，并将数值结果保存到 `spread_path`。
    spread_path = evaluation_dir / "spread_summary.parquet"
    # 计算 `evaluation_dir / 'final_audit.parquet'`，并将数值结果保存到 `audit_path`。
    audit_path = evaluation_dir / "final_audit.parquet"
    # 计算 `evaluation_dir / 'pyfolio_performance.parquet'`，并将数值结果保存到 `pyfolio_path`。
    pyfolio_path = evaluation_dir / "pyfolio_performance.parquet"
    # 计算 `evaluation_dir / 'relative_performance.parquet'`，并将数值结果保存到 `relative_path`。
    relative_path = evaluation_dir / "relative_performance.parquet"
    # 按 `ic_path.exists()` 在两个候选值间选择，并保存到 `ic_summary`。
    ic_summary = pd.read_parquet(ic_path) if ic_path.exists() else pd.DataFrame()
    # 按 `spread_path.exists()` 在两个候选值间选择，并保存到 `spread_summary`。
    spread_summary = pd.read_parquet(spread_path) if spread_path.exists() else pd.DataFrame()
    # 按 `audit_path.exists()` 在两个候选值间选择，并保存到 `final_audit`。
    final_audit = pd.read_parquet(audit_path) if audit_path.exists() else pd.DataFrame()
    # 按 `pyfolio_path.exists()` 在两个候选值间选择，并保存到 `pyfolio`。
    pyfolio = (
        pd.read_parquet(pyfolio_path) if pyfolio_path.exists() else pd.DataFrame()
    )
    # 按 `relative_path.exists()` 在两个候选值间选择，并保存到 `relative_performance`。
    relative_performance = (
        pd.read_parquet(relative_path) if relative_path.exists() else pd.DataFrame()
    )
    # 调用 `read_json` 读取源数据，并将结果保存到 `evaluation_manifest`。
    evaluation_manifest = read_json(
        evaluation_dir / "factor_evaluation_manifest.json"
    )
    # 检查 `evaluation_manifest`；根据条件结果选择对应处理分支。
    if evaluation_manifest:
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `price_start_offset`。
        price_start_offset = int(evaluation_manifest.get("price_start_offset", 0))
        # 按 `price_start_offset == 0` 在两个候选值间选择，并保存到 `period_description`。
        period_description = (
            "1/5/10D 对应 T→T+1/T+5/T+10"
            if price_start_offset == 0
            else "1/5/10D 对应 T+1→T+2/T+6/T+11"
        )
        # 按 `evaluation_manifest.get('execution_mask') == 'can_enter_t1'` 在两个候选值间选择，并保存到 `execution_description`。
        execution_description = (
            "组合收益和换手在完整 T 日分组后应用 `can_enter_t1`"
            if evaluation_manifest.get("execution_mask") == "can_enter_t1"
            else "组合收益和换手不应用 `can_enter_t1` 未来状态过滤"
        )
        # 构造当前步骤所需的列表，并保存到 `evaluation_policy_lines`。
        evaluation_policy_lines = [
            f"- 回测模式：`{evaluation_manifest.get('backtest_mode')}`；{execution_description}。",
            f"- 收益公式：`{evaluation_manifest.get('forward_return_formula')}`；{period_description}。",
            "- 全市场等权基准使用完整因子覆盖池和相同收益时点，不应用 `can_enter_t1`。",
            "- Q5 与 Q5−Q1 的超额年化为组合几何年化减基准几何年化；信息比使用日度主动收益并按 252 日年化。",
        ]
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 构造当前步骤所需的列表，并保存到 `evaluation_policy_lines`。
        evaluation_policy_lines = ["- 因子评价尚未按新版回测口径生成。"]
    # 调用 `read_json` 读取源数据，并将结果保存到 `training_manifest`。
    training_manifest = read_json(quality_run / "a2_training_manifest.json")
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `batch_summary`。
    batch_summary = pd.DataFrame(
        training_manifest.get("training_batch_size_by_year", [])
        if training_manifest else []
    )

    # 构造当前步骤所需的列表，并保存到 `lines`。
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
    # 检查 `training_manifest`；根据条件结果选择对应处理分支。
    if training_manifest:
        # 调用 `training_manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `train_batch_mode`。
        train_batch_mode = training_manifest.get(
            "train_batch_mode", "fixed_samples"
        )
        # 调用 `training_manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `inference_batch_size`。
        inference_batch_size = training_manifest.get(
            "inference_batch_size",
            training_manifest.get("batch_size_per_task"),
        )
        # 调用 `training_manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `validation_metric`。
        validation_metric = training_manifest.get(
            "validation_metric", "pooled_pearson_ic"
        )
        # 调用 `training_manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `loss_name`。
        loss_name = training_manifest.get("loss_name", "pearson_ic")
        # 调用 `training_manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `loss_parameters`。
        loss_parameters = training_manifest.get("loss_parameters", {})
        # 把一组结果展开追加到 `lines`，保持统一的一维记录序列。
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
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 把一组结果展开追加到 `lines`，保持统一的一维记录序列。
        lines.extend(["训练尚未完成。", ""])
    # 把一组结果展开追加到 `lines`，保持统一的一维记录序列。
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
    # 调用 `quality_run.mkdir` 创建输出目录，确保后续文件可以安全写入。
    quality_run.mkdir(parents=True, exist_ok=True)
    # 计算 `quality_run / 'EXPERIMENT_REPORT.md'`，并将数值结果保存到 `report_path`。
    report_path = quality_run / "EXPERIMENT_REPORT.md"
    # 调用 `report_path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = report_path.with_name(f"{report_path.name}.{os.getpid()}.tmp")
    # 调用 `temporary.write_text` 将当前结果持久化到目标文件。
    temporary.write_text("\n".join(lines), encoding="utf-8")
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(report_path)
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"实验报告：{report_path}")
    # 将 `report_path` 作为当前函数结果返回给调用方。
    return report_path


# ============================== 评价 CLI ==============================
def add_evaluation_arguments(parser: argparse.ArgumentParser, *, include_backtest_mode: bool = False) -> None:
    """向命令行解析器注册样本外评价参数。

    输入：
        parser: argparse.ArgumentParser。
        include_backtest_mode: bool。
    输出：
        None。
    """
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--data-root", default=str(VIT_DIR / "data"))
    # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
    parser.add_argument("--run-name", default=DEFAULT_A2_RUN_NAME)
    # 检查 `include_backtest_mode`；根据条件结果选择对应处理分支。
    if include_backtest_mode:
        # 调用 `parser.add_argument` 向命令行解析器注册参数、类型、默认值和帮助信息。
        parser.add_argument("--backtest-mode", choices=BACKTEST_MODES, default=DEFAULT_BACKTEST_MODE)


# 定义 `parse_args` 函数，封装这一阶段可独立调用的处理步骤。
def parse_args() -> argparse.Namespace:
    """解析阶段 08 的回测或报告命令。

    输入：
        无显式输入参数。
    输出：
        argparse.Namespace。
    """
    # 调用 `argparse.ArgumentParser` 创建命令行参数解析器，并将结果保存到 `parser`。
    parser = argparse.ArgumentParser(description=__doc__)
    # 调用 `parser.add_subparsers` 创建子命令分发器以区分不同运行模式，并将结果保存到 `commands`。
    commands = parser.add_subparsers(dest="command", required=True)
    # 调用 `add_evaluation_arguments` 在不更新参数的条件下计算验证或样本外评价指标。
    add_evaluation_arguments(commands.add_parser("backtest"), include_backtest_mode=True)
    # 调用 `add_evaluation_arguments` 在不更新参数的条件下计算验证或样本外评价指标。
    add_evaluation_arguments(commands.add_parser("report"))
    # 将 `parser.parse_args()` 作为当前函数结果返回给调用方。
    return parser.parse_args()


# 定义命令行入口，解析参数并按用户选择调度当前阶段的主流程。
def main() -> None:
    """分派样本外回测或实验报告生成流程。

    输入：
        无显式输入参数。
    输出：
        None。
    """
    # 调用 `parse_args` 解析命令行文本并生成结构化参数，并将结果保存到 `args`。
    args = parse_args()
    # 检查 `args.command == 'backtest'`；根据条件结果选择对应处理分支。
    if args.command == "backtest":
        # 调用 `run_oos_backtest` 按当前配置执行对应的完整子流程。
        run_oos_backtest(run_name=args.run_name, data_root=args.data_root, backtest_mode=args.backtest_mode)
    # 检查 `args.command == 'report'`；根据条件结果选择对应处理分支。
    elif args.command == "report":
        # 调用 `generate_experiment_report` 根据当前结果生成可审计的报告或展示产物。
        generate_experiment_report(run_name=args.run_name, data_root=args.data_root)


# 检查 `__name__ == '__main__'`；根据条件结果选择对应处理分支。
if __name__ == "__main__":
    # 调用 `main` 启动当前模块的命令行主流程。
    main()
