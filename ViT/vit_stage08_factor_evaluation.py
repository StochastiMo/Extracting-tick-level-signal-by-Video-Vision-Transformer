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


DEFAULT_A2_RUN_NAME = "date_batch"  # 设置默认a2执行名称。
BACKTEST_MODES = (  # 设置backtestmodes。
    "mode1_t_with_t1_filter",
    "mode2_t_no_filter",
    "mode3_t1_no_filter",
)
DEFAULT_BACKTEST_MODE = "mode2_t_no_filter"  # 设置默认backtest模式。
EQUAL_WEIGHT_MARKET_COLUMN = "Equal-weight market"  # 设置equal权重market列。
RELATIVE_PORTFOLIOS = ("Q5 long-only", "Q5-Q1 long-short")  # 设置relativeportfolios。
ANNUALIZATION_DAYS = 252  # 设置annualizationdays。
BACKTEST_MODE_CONFIGS = {  # 设置backtest模式configs。
    "mode1_t_with_t1_filter": {  # 记录mode1twitht1filter。
        "price_start_offset": 0,  # 记录价格起点偏移。
        "apply_can_enter_t1": True,  # 记录applycanentert1。
        "price_start": "T_no_shift",  # 把 price_start 映射为"T_no_shift"。
        "portfolio_scope": "preassigned_quantile_then_can_enter_t1",  # 把 portfolio_scope 映射为"preassigned_quantile_then_can_enter_t1"。
        "execution_mask": "can_enter_t1",  # 把 execution_mask 映射为"can_enter_t1"。
    },
    "mode2_t_no_filter": {  # 记录mode2tnofilter。
        "price_start_offset": 0,  # 记录价格起点偏移。
        "apply_can_enter_t1": False,  # 记录applycanentert1。
        "price_start": "T_no_shift",  # 把 price_start 映射为"T_no_shift"。
        "portfolio_scope": "full_t_cross_section_no_t1_filter",  # 把 portfolio_scope 映射为"full_t_cross_section_no_t1_filter"。
        "execution_mask": "none",  # 把 execution_mask 映射为"none"。
    },
    "mode3_t1_no_filter": {  # 记录mode3t1nofilter。
        "price_start_offset": 1,  # 记录价格起点偏移。
        "apply_can_enter_t1": False,  # 记录applycanentert1。
        "price_start": "T_plus_1",  # 把 price_start 映射为"T_plus_1"。
        "portfolio_scope": "full_t_cross_section_no_t1_filter",  # 把 portfolio_scope 映射为"full_t_cross_section_no_t1_filter"。
        "execution_mask": "none",  # 把 execution_mask 映射为"none"。
    },
}


def validate_run_name(value: str) -> str:  # 校验 run 名称，避免路径穿越和非法目录字符。
    """校验 run 名称，避免路径穿越和非法目录字符。"""
    run_name = str(value).strip()  # 去除首尾空白并规范化运行名称。
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")  # 初始化allowed容器。
    if (  # 开始组合多项有效性或配置条件。
        not run_name
        or run_name in {".", ".."}
        or any(character not in allowed for character in run_name)
    ):
        raise ValueError("运行名含非法字符")  # 输入数据不满足业务或数值约束时立即终止。
    return run_name  # 返回运行名称。


def resolve_backtest_mode(backtest_mode: str) -> dict[str, object]:  # 校验回测模式，并返回配置副本。
    """校验回测模式，并返回配置副本。"""

    mode = str(backtest_mode).strip()  # 去除首尾空白并规范化运行模式。
    if mode not in BACKTEST_MODE_CONFIGS:  # 检查当前值是否属于允许集合。
        raise ValueError(  # 输入数据不满足业务或数值约束时立即终止。
            f"未知回测模式：{backtest_mode!r}；可选 {BACKTEST_MODES}"
        )
    return {"name": mode, **BACKTEST_MODE_CONFIGS[mode]}  # 返回校验回测模式，并返回配置副本。


def price_point_label(offset: int) -> str:  # 返回回测价格时点的可读标签。
    """返回回测价格时点的可读标签。"""
    return "T" if offset == 0 else f"T+{offset}"  # 返回回测价格时点的可读标签。


def forward_return_formula(price_start_offset: int) -> str:  # 返回指定持有期对应的远期收益公式说明。
    """返回指定持有期对应的远期收益公式说明。"""
    if price_start_offset == 0:  # 按当前模式、标识或状态选择处理分支。
        return "adj_close(T+k) / adj_close(T) - 1"  # 返回指定持有期对应的远期收益公式说明。
    return "adj_close(T+1+k) / adj_close(T+1) - 1"  # 返回指定持有期对应的远期收益公式说明。


def geometric_annual_return(  # 以日收益复利计算几何年化收益。
    returns: pd.Series | np.ndarray, annualization: int = ANNUALIZATION_DAYS  # 返回以日收益复利计算几何年化收益。
) -> float:  # 声明当前函数的返回类型。
    """以日收益复利计算几何年化收益。"""

    values = np.asarray(returns, dtype=np.float64)  # 转换并统一值的数据表示。
    if len(values) == 0 or not np.isfinite(values).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("年化收益输入必须是非空有限日收益")  # 输入数据不满足业务或数值约束时立即终止。
    if (values <= -1).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError("日收益必须大于 -100%")  # 输入数据不满足业务或数值约束时立即终止。
    growth = float(np.prod(1.0 + values))  # 计算growth。
    if not np.isfinite(growth) or growth <= 0:  # 检查参与计算的数值是否全部有限。
        raise ValueError("累计净值必须为正且有限")  # 输入数据不满足业务或数值约束时立即终止。
    return growth ** (annualization / len(values)) - 1.0  # 返回以日收益复利计算几何年化收益。


def compute_relative_performance(  # 计算 Q5 与 Q5-Q1 相对全市场等权基准的绩效。
    portfolio_returns: pd.DataFrame,  # 接收portfolio收益。
    benchmark_column: str = EQUAL_WEIGHT_MARKET_COLUMN,  # 接收benchmark列。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """计算 Q5 与 Q5-Q1 相对全市场等权基准的绩效。"""

    required = (*RELATIVE_PORTFOLIOS, benchmark_column)  # 计算必需字段。
    missing = set(required).difference(portfolio_returns.columns)  # 检查输入缺失字段。
    if missing:  # 根据当前数据状态选择处理分支。
        raise KeyError(f"相对绩效缺少收益列：{sorted(missing)}")  # 缺少必需字段或配置键时立即终止。
    values = portfolio_returns[list(required)].to_numpy(np.float64)  # 转换并统一值的数据表示。
    if len(portfolio_returns) < 2 or not np.isfinite(values).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("相对绩效至少需要两日有限收益")  # 输入数据不满足业务或数值约束时立即终止。

    benchmark = portfolio_returns[benchmark_column].to_numpy(np.float64)  # 转换并统一benchmark的数据表示。
    benchmark_annual_return = geometric_annual_return(benchmark)  # 由日收益复利年化。
    records = []  # 初始化records容器。
    for portfolio in RELATIVE_PORTFOLIOS:  # 遍历relativeportfolios，逐项处理portfolio。
        returns = portfolio_returns[portfolio].to_numpy(np.float64)  # 返回上述计算结果。
        active = returns - benchmark  # 计算active。
        active_scale = max(1.0, float(np.max(np.abs(active))))  # 计算activescale。
        if float(np.ptp(active)) <= np.finfo(np.float64).eps * active_scale * 16:  # 检查当前数值是否越过业务或数值阈值。
            raise ValueError(f"{portfolio} 主动收益方差为零，信息比无定义")  # 输入数据不满足业务或数值约束时立即终止。
        active_std = float(active.std(ddof=1))  # 计算active标准差。
        if not np.isfinite(active_std) or active_std <= 0:  # 检查参与计算的数值是否全部有限。
            raise ValueError(f"{portfolio} 主动收益方差为零，信息比无定义")  # 输入数据不满足业务或数值约束时立即终止。
        portfolio_annual_return = geometric_annual_return(returns)  # 由日收益复利年化。
        records.append(  # 把当前结果追加到汇总列表。
            {
                "portfolio": portfolio,  # 记录portfolio。
                "benchmark": benchmark_column,  # 记录benchmark。
                "observations": len(active),  # 记录observations。
                "portfolio_annual_return": portfolio_annual_return,  # 记录portfolio年度收益。
                "benchmark_annual_return": benchmark_annual_return,  # 记录benchmark年度收益。
                "annual_excess_return": (  # 记录年度excess收益。
                    portfolio_annual_return - benchmark_annual_return
                ),
                "annual_tracking_error": (  # 记录年度tracking误差。
                    active_std * np.sqrt(ANNUALIZATION_DAYS)
                ),
                "information_ratio": (  # 记录information比例。
                    float(active.mean()) / active_std
                    * np.sqrt(ANNUALIZATION_DAYS)
                ),
            }
        )
    return pd.DataFrame(records)  # 返回上述计算结果。

def build_return_price_view(  # 让 Alphalens 在信号日 T 索引上按所选起点计算远期收益。
    prices: pd.DataFrame, backtest_mode: str = DEFAULT_BACKTEST_MODE  # 计算复权价矩阵。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """让 Alphalens 在信号日 T 索引上按所选起点计算远期收益。"""

    offset = int(resolve_backtest_mode(backtest_mode)["price_start_offset"])  # 解析offset对应的运行配置。
    return prices if offset == 0 else prices.shift(-offset)  # 返回让 Alphalens 在信号日 T 索引上按所选起点计算远期收益。


def select_execution_factor_data(  # 仅口径 1 在已固定分位后应用 T+1 可进入掩码。
    factor_data: pd.DataFrame,  # 接收因子数据。
    can_enter_t1: pd.Series | np.ndarray,  # 接收canentert1。
    backtest_mode: str = DEFAULT_BACKTEST_MODE,  # 接收backtest模式。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """仅口径 1 在已固定分位后应用 T+1 可进入掩码。"""

    mode = resolve_backtest_mode(backtest_mode)  # 解析运行模式对应的运行配置。
    mask = np.asarray(can_enter_t1, dtype=bool)  # 转换并统一掩码的数据表示。
    if len(mask) != len(factor_data):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("can_enter_t1 与因子数据长度不一致")  # 输入数据不满足业务或数值约束时立即终止。
    if bool(mode["apply_can_enter_t1"]):  # 根据当前数据状态选择处理分支。
        return factor_data.iloc[np.flatnonzero(mask)].copy()  # 返回仅口径 1 在已固定分位后应用 T+1 可进入掩码。
    return factor_data  # 返回因子数据。


def compute_equal_weight_market_returns(  # 在完整、价格有效的 T 日因子覆盖池内计算等权市场收益。
    factor_data: pd.DataFrame, return_column: str = "1D"  # 计算因子数据。
) -> pd.Series:  # 声明当前函数的返回类型。
    """在完整、价格有效的 T 日因子覆盖池内计算等权市场收益。"""

    if return_column not in factor_data.columns:  # 检查当前值是否属于允许集合。
        raise KeyError(f"缺少基准收益列：{return_column}")  # 缺少必需字段或配置键时立即终止。
    values = factor_data[return_column].to_numpy(np.float64)  # 转换并统一值的数据表示。
    if not np.isfinite(values).all():  # 检查参与计算的数值是否全部有限。
        raise ValueError("全市场等权基准输入含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。
    return (  # 返回在完整、价格有效的 T 日因子覆盖池内计算等权市场收益。
        factor_data[return_column]  # 汇入因子数据。
        .groupby(level="date", sort=True)
        .mean()
        .rename(EQUAL_WEIGHT_MARKET_COLUMN)
    )


# ====================== 04｜Alphalens 样本外评价 ======================
def run_oos_backtest(  # 按指定价格起点与执行过滤口径计算样本外收益。
    run_name: str = DEFAULT_A2_RUN_NAME,  # 接收执行名称。
    data_root: str | Path | None = None,  # 接收数据根目录。
    backtest_mode: str = DEFAULT_BACKTEST_MODE,  # 接收backtest模式。
) -> pd.DataFrame:  # 声明当前函数的返回类型。
    """按指定价格起点与执行过滤口径计算样本外收益。"""

    run_name = validate_run_name(run_name)  # 校验并规范化运行名称。
    mode = resolve_backtest_mode(backtest_mode)  # 解析运行模式对应的运行配置。
    price_start_offset = int(mode["price_start_offset"])  # 从上游结果中读取price起点offset。
    warnings.filterwarnings(
        "ignore", message=r'Module "zipline\.assets" not found.*'
    )
    try:  # 执行可能失败的外部读取或计算步骤。
        import alphalens as al
        import matplotlib.pyplot as plt
        import pyfolio as pf
        from matplotlib.ticker import PercentFormatter
        from scipy import stats
    except ModuleNotFoundError as error:  # 捕获预期异常并执行对应恢复逻辑。
        raise RuntimeError(  # 当前运行环境或任务状态不支持继续执行。
            "请使用 Jupyter 内核 `ViT Factor Eval (Alphalens/Pyfolio)`。"
        ) from error

  # factor-eval 环境已安装该字体；显式置顶，避免 Matplotlib 默认。
  # DejaVu Sans 缺少中文字符而在 Notebook 中产生方框和大量 warning。
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    factor_column = "prediction_ensemble"  # 计算因子列。
    periods = (1, 5, 10)  # 计算periods。
    quantiles = 5  # 计算quantiles。
    max_loss = 0.001  # 计算max损失。
    price_column = "s_dq_adjclose"  # 计算price列。
    root = (  # 计算根目录。
        Path(data_root).resolve()
        if data_root is not None  # 检查必需数据、配置或校验结果是否缺失。
        else get_data_root().resolve()
    )
    if not root.is_dir():  # 检查所需文件或目录是否存在。
        raise FileNotFoundError(root)  # 目标文件不存在时立即终止。
    prediction_dir = root / "predictions" / "runs" / run_name  # 构造预测目录。
    universe_dir = root / "universe"  # 构造股票池目录。
    quality_dir = root / "quality" / "runs" / run_name  # 构造质量目录。
    evaluation_dir = quality_dir / "factor_evaluation"  # 构造evaluation目录。
    evaluation_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    training_manifest = read_json(quality_dir / "a2_training_manifest.json")  # 读取训练清单。
    universe_manifest = read_json(universe_dir / "universe_manifest.json")  # 读取股票池清单。
    dataset_manifest = read_json(root / "feature_store" / "dataset_manifest.json")  # 读取dataset清单。
    if training_manifest is None or universe_manifest is None or dataset_manifest is None:  # 检查所需对象是否尚未初始化。
        raise FileNotFoundError("训练、Universe 或 feature store manifest 不完整")  # 目标文件不存在时立即终止。
    if training_manifest["universe_signature"] != universe_manifest["signature"]:  # 按当前模式、标识或状态选择处理分支。
        raise AssertionError("训练使用的 Universe 与当前 Universe 签名不一致")  # 关键数据契约被破坏时立即终止。
    if training_manifest["dataset_signature"] != dataset_manifest["signature"]:  # 按当前模式、标识或状态选择处理分支。
        raise AssertionError("训练使用的 feature store 与当前数据签名不一致")  # 关键数据契约被破坏时立即终止。

    prediction_years = tuple(  # 初始化预测年份容器。
        year  # 汇入年份。
        for year in training_manifest["years"]  # 遍历训练清单，逐项处理年份。
        if (prediction_dir / f"vit_predictions.{year}.parquet").exists()  # 检查目标文件或目录是否满足存在性要求。
    )
    if not prediction_years:  # 检查必需数据、配置或校验结果是否缺失。
        raise FileNotFoundError("没有可评价的年度预测")  # 目标文件不存在时立即终止。
    prediction_parts: list[pd.DataFrame] = []  # 接收预测parts。
    audit_records: list[dict[str, object]] = []  # 初始化auditrecords容器。
    for year in prediction_years:  # 遍历预测年份，逐项处理年份。
        prediction_path = prediction_dir / f"vit_predictions.{year}.parquet"  # 构造预测路径。
        prediction = pd.read_parquet(  # 读取预测。
            prediction_path,  # 汇入预测路径。
            columns=["trade_date", "instrument_id", factor_column],  # 限定本次读取或输出使用的字段。
        )
        prediction["trade_date"] = prediction["trade_date"].astype(str)  # 转换预测的数据类型或存储位置。
        prediction["instrument_id"] = prediction["instrument_id"].astype(str)  # 转换预测的数据类型或存储位置。
        if prediction.duplicated(["trade_date", "instrument_id"]).any():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError(f"{year} prediction 业务键重复")  # 输入数据不满足业务或数值约束时立即终止。
        if not prediction["trade_date"].str.startswith(str(year)).all():  # 检查当前掩码或数据是否满足有效性要求。
            raise ValueError(f"{year} prediction 含跨年度日期")  # 输入数据不满足业务或数值约束时立即终止。
        if not np.isfinite(prediction[factor_column].to_numpy(np.float64)).all():  # 检查参与计算的数值是否全部有限。
            raise ValueError(f"{year} 因子含 NaN 或 Inf")  # 输入数据不满足业务或数值约束时立即终止。

        universe = pd.read_parquet(  # 读取股票池。
            universe_dir / f"universe.{year}.parquet",
            columns=[  # 初始化列容器。
                "trade_date", "instrument_id", "feature_valid",
                "in_universe_t", "can_enter_t1",
            ],
        )
        expected = universe.loc[  # 计算预期结果。
            universe["feature_valid"] & universe["in_universe_t"],
            ["trade_date", "instrument_id", "can_enter_t1"],
        ].copy()
        expected["trade_date"] = expected["trade_date"].astype(str)  # 转换预期的数据类型或存储位置。
        expected["instrument_id"] = expected["instrument_id"].astype(str)  # 转换预期的数据类型或存储位置。
        joined = prediction.merge(  # 按业务键对齐，避免依赖原始行序。
            expected,  # 汇入预期。
            on=["trade_date", "instrument_id"],  # 指定数据连接使用的业务键。
            how="outer",  # 指定数据连接方式。
            validate="one_to_one",  # 校验数据连接的基数关系。
            indicator=True,  # 保留连接来源标记以审计匹配率。
        )
        if not joined["_merge"].eq("both").all():  # 检查当前掩码或数据是否满足有效性要求。
            raise AssertionError(f"{year} prediction 与 T 日 Universe 键不一致")  # 关键数据契约被破坏时立即终止。
        joined = joined.drop(columns="_merge")  # 按股票与交易日业务键对齐。
        if joined["can_enter_t1"].isna().any():  # 检查当前掩码或数据是否满足有效性要求。
            raise AssertionError(f"{year} can_enter_t1 含缺失值")  # 关键数据契约被破坏时立即终止。
        joined["can_enter_t1"] = joined["can_enter_t1"].astype(bool)  # 转换joined的数据类型或存储位置。
        can_enter_t1_rows = int(joined["can_enter_t1"].sum())  # 计算canentert1行数。
        audit_records.append(  # 把当前结果追加到汇总列表。
            {
                "year": year,  # 记录年份。
                "prediction_rows": len(joined),  # 记录预测行数。
                "can_enter_t1_rows": can_enter_t1_rows,  # 记录canentert1行数。
                "t1_excluded_rows": len(joined) - can_enter_t1_rows,  # 记录t1excluded行数。
                "dates": joined["trade_date"].nunique(),  # 记录交易日。
                "first_date": joined["trade_date"].min(),  # 记录首个交易日。
                "last_date": joined["trade_date"].max(),  # 记录最后一个交易日。
            }
        )
        prediction_parts.append(joined)  # 把当前结果追加到汇总列表。

    prediction_audit = pd.DataFrame(audit_records)  # 把累计记录整理为预测audit。
    panel = pd.concat(prediction_parts, ignore_index=True)  # 对齐并合并panel。
    del prediction_parts, prediction, universe, expected, joined  # 释放不再使用的中间引用。
    if panel.duplicated(["trade_date", "instrument_id"]).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError("跨年度因子业务键重复")  # 输入数据不满足业务或数值约束时立即终止。
    t_grouping_rows_before_price_filter = len(panel)  # 计算tgrouping行数beforepricefilter。
    panel["date"] = pd.to_datetime(panel["trade_date"], format="%Y%m%d")  # 解析并统一panel。
    factor_panel = (  # 固定日期与股票排序。
        panel.set_index(["date", "instrument_id"])[
            [factor_column, "can_enter_t1"]
        ]
        .sort_index()
    )
    factor_panel.index = factor_panel.index.set_names(["date", "asset"])
    factor = factor_panel[factor_column].rename("factor")  # 提取并重命名因子序列。
    can_enter_t1 = factor_panel["can_enter_t1"].astype(bool)  # 转换canentert1的数据类型或存储位置。
    del panel, factor_panel  # 释放不再使用的中间引用。

  # 分位组只能使用 T 日已经可知的因子截面。先对完整 T 日预测池分组，。
  # 后续价格缺失与 T+1 可进入状态都只能删行，不能重新改变组别边界。
    groups = factor.groupby(level="date", sort=False)  # 按业务键分组得到分组。
    ordinal_rank = groups.rank(method="first", ascending=True)  # 计算ordinal排名。
    group_size = groups.transform("size")  # 计算group大小。
    factor_quantile = pd.Series(  # 构造带业务索引的因子quantile。
        np.floor(
            (ordinal_rank.to_numpy(np.float64) - 1) * quantiles
            / group_size.to_numpy(np.float64)
        ).astype(np.int64)
        + 1,
        index=factor.index,  # 控制是否把 DataFrame 索引写入结果。
        name="factor_quantile",  # 设置结果对象名称。
    )
    t_grouping_quantile_counts = (  # 计算tgroupingquantile数量。
        factor_quantile.groupby(
            [pd.Grouper(level="date"), factor_quantile], observed=True
        ).size().unstack("factor_quantile")
    )
    if t_grouping_quantile_counts.isna().any(axis=None) or (  # 根据当前数据状态选择处理分支。
        t_grouping_quantile_counts.max(axis=1)
        - t_grouping_quantile_counts.min(axis=1)
    ).gt(1).any():
        raise AssertionError("T 日完整截面五分位分组不均衡")  # 关键数据契约被破坏时立即终止。

    calendar = pd.read_parquet(  # 读取交易日历。
        CALENDAR_PATH,  # 汇入交易日历路径。
        columns=["trade_days", "s_info_exchmarket"],  # 限定本次读取或输出使用的字段。
        filters=[("s_info_exchmarket", "==", "SSE")],  # 下推数据筛选条件，减少无关数据读取。
    )
    calendar_dates = pd.DatetimeIndex(  # 构造有序交易日索引。
        pd.to_datetime(normalize_dates(calendar["trade_days"]), format="%Y%m%d")
        .sort_values().unique(),
        name="date",  # 设置结果对象名称。
    )
    factor_dates = pd.DatetimeIndex(  # 构造有序交易日索引。
        factor.index.get_level_values("date").unique().sort_values()
    )
    positions = pd.Series(np.arange(len(calendar_dates)), index=calendar_dates)  # 构造带业务索引的位置索引。
    factor_positions = factor_dates.map(positions)  # 计算因子位置。
    if factor_positions.isna().any():  # 检查当前掩码或数据是否满足有效性要求。
        raise AssertionError("因子日期无法映射到交易日历")  # 关键数据契约被破坏时立即终止。
    first_price_date = factor_dates.min()  # 计算firstprice交易日。
    final_price_position = (  # 计算finalprice位置。
        int(factor_positions[-1]) + price_start_offset + max(periods)
    )
    if final_price_position >= len(calendar_dates):  # 检查当前样本数量是否满足计算要求。
        raise ValueError("交易日历不足以覆盖最大收益期限")  # 输入数据不满足业务或数值约束时立即终止。
    last_price_date = calendar_dates[final_price_position]  # 计算lastprice交易日。
    price_calendar = calendar_dates[  # 计算price交易日历。
        calendar_dates.get_loc(first_price_date) : final_price_position + 1
    ]
    assets = pd.Index(  # 计算assets。
        sorted(factor.index.get_level_values("asset").unique()), name="asset"
    )
    wind_codes = pd.Series(assets).str.extract(  # 拆分证券代码字段。
        r"^(?P<exchange>SH|SZ)\.(?P<number>\d{6})$"
    )
    if wind_codes.isna().any(axis=None):  # 根据当前数据状态选择处理分支。
        raise ValueError("因子股票代码格式异常")  # 输入数据不满足业务或数值约束时立即终止。
    needed_wind_codes = set(wind_codes["number"] + "." + wind_codes["exchange"])  # 初始化neededwind代码容器。
    eod = pd.read_parquet(  # 读取日行情。
        EOD_PATH,  # 汇入日行情路径。
        columns=["s_info_windcode", "trade_dt", price_column],  # 限定本次读取或输出使用的字段。
        filters=[  # 初始化filters容器。
            ("trade_dt", ">=", first_price_date.strftime("%Y%m%d")),
            ("trade_dt", "<=", last_price_date.strftime("%Y%m%d")),
        ],
    )
    eod["trade_dt"] = normalize_dates(eod["trade_dt"])  # 仅使用当日截面统计量。
    eod["s_info_windcode"] = eod["s_info_windcode"].astype(str)  # 转换日行情的数据类型或存储位置。
    eod = eod.loc[eod["s_info_windcode"].isin(needed_wind_codes)].copy()  # 转换日行情的数据类型或存储位置。
    parts = eod["s_info_windcode"].str.extract(  # 拆分证券代码字段。
        r"^(?P<number>\d{6})\.(?P<exchange>SH|SZ)$"
    )
    eod["asset"] = parts["exchange"] + "." + parts["number"]  # 从上游结果中读取日行情。
    eod["date"] = pd.to_datetime(eod["trade_dt"], format="%Y%m%d")  # 解析并统一日行情。
    if eod.duplicated(["date", "asset"]).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise ValueError("Wind 复权收盘价业务键重复")  # 输入数据不满足业务或数值约束时立即终止。
    valid_price = (  # 构造有效price，限定后续参与计算的样本。
        np.isfinite(eod[price_column].to_numpy(np.float64))
        & eod[price_column].gt(0).to_numpy()
    )
    eod.loc[~valid_price, price_column] = np.nan
    prices = (  # 计算复权价矩阵。
        eod.pivot(index="date", columns="asset", values=price_column)
        .reindex(index=price_calendar, columns=assets)
        .astype(np.float64)
    )
    prices.index.name, prices.columns.name = "date", "asset"

    factor_keys = factor.index.to_frame(index=False)  # 计算因子键。
    start_rows = prices.index.get_indexer(factor_keys["date"])  # 从上游结果中读取起点行数。
    asset_columns = prices.columns.get_indexer(factor_keys["asset"])  # 从上游结果中读取asset列。
    if (start_rows < 0).any() or (asset_columns < 0).any():  # 检查当前掩码或数据是否满足有效性要求。
        raise AssertionError("因子键无法映射到 T 日价格矩阵")  # 关键数据契约被破坏时立即终止。
    price_values = prices.to_numpy(copy=False)  # 转换并统一price值的数据表示。
    all_prices = np.ones(len(factor), dtype=bool)  # 初始化allprices。
    coverage_records = []  # 初始化coveragerecords容器。
    required_offsets = (  # 计算必需offsets。
        price_start_offset,  # 汇入price起点offset。
        *(price_start_offset + period for period in periods),
    )
    for offset in required_offsets:  # 遍历必需offsets，逐项处理offset。
        selected = price_values[start_rows + offset, asset_columns]  # 构造筛选结果，限定后续参与计算的样本。
        available = np.isfinite(selected) & (selected > 0)  # 计算available判定条件。
        all_prices &= available
        coverage_records.append(  # 把当前结果追加到汇总列表。
            {
                "price_point": price_point_label(offset),  # 记录价格point。
                "available_rows": int(available.sum()),  # 记录available行数。
                "missing_rows": int((~available).sum()),  # 记录缺失值行数。
                "coverage": float(available.mean()),  # 记录覆盖率。
            }
        )
    price_coverage = pd.DataFrame(coverage_records)  # 把累计记录整理为pricecoverage。
    if 1 - float(all_prices.mean()) > max_loss:  # 检查当前数值是否越过业务或数值阈值。
        raise AssertionError("复权收盘价缺失比例超过 MAX_LOSS=0.001")  # 关键数据契约被破坏时立即终止。
    valid_price_positions = np.flatnonzero(all_prices)  # 构造有效price位置，限定后续参与计算的样本。
    factor = factor.iloc[valid_price_positions].copy()  # 转换因子的数据类型或存储位置。
    factor_quantile = factor_quantile.iloc[valid_price_positions].copy()  # 转换因子quantile的数据类型或存储位置。
    can_enter_t1 = can_enter_t1.iloc[valid_price_positions].copy()  # 转换canentert1的数据类型或存储位置。
    t_grouping_rows_after_price_filter = len(factor)  # 计算tgrouping行数afterpricefilter。

  # 保留完整交易日 level，明确 period 是交易日偏移且起点就是 T 日。
    analysis_calendar = calendar_dates[  # 计算analysis交易日历。
        (calendar_dates >= factor_dates.min()) & (calendar_dates <= factor_dates.max())
    ]
    factor_date_codes = analysis_calendar.get_indexer(  # 计算因子交易日代码。
        factor.index.get_level_values("date")
    )
    factor_asset_codes = assets.get_indexer(factor.index.get_level_values("asset"))  # 计算因子asset代码。
    factor_index = pd.MultiIndex(  # 计算因子索引。
        levels=[analysis_calendar, assets],  # 配置levels。
        codes=[factor_date_codes, factor_asset_codes],  # 配置代码。
        names=["date", "asset"],  # 配置名称。
        verify_integrity=True,  # 配置verifyintegrity。
    )
    factor.index = factor_index
    factor_quantile.index = factor_index
    can_enter_t1.index = factor_index

  # 将价格矩阵按模式平移，但因子索引始终保留信号日 T。
    return_prices = build_return_price_view(prices, str(mode["name"]))  # 返回按当前规则组织后的结果。
    with warnings.catch_warnings():  # 进入受控上下文并在退出时自动释放资源。
        warnings.filterwarnings("ignore", category=FutureWarning)
        factor_data = al.utils.compute_forward_returns(  # 计算因子数据。
            factor=factor,  # 配置因子。
            prices=return_prices,  # 配置prices。
            periods=periods,  # 配置periods。
            filter_zscore=None,  # 配置filterzscore。
            cumulative_returns=True,  # 配置cumulative收益。
        )
    if not factor_data.index.equals(factor.index):  # 检查必需数据、配置或校验结果是否缺失。
        raise AssertionError("Alphalens 收益键与 T 日因子键不一致")  # 关键数据契约被破坏时立即终止。
    factor_data["factor"] = factor.to_numpy(np.float64)  # 转换并统一因子数据的数据表示。
    factor_data["factor_quantile"] = factor_quantile.to_numpy(np.int64)  # 转换并统一因子数据的数据表示。
    forward_columns = tuple(f"{period}D" for period in periods)  # 初始化forward列容器。
    missing_columns = set((*forward_columns, "factor", "factor_quantile")).difference(  # 检查输入缺失字段。
        factor_data.columns  # 汇入因子数据列。
    )
    if missing_columns:  # 根据当前数据状态选择处理分支。
        raise AssertionError(f"Alphalens 缺少字段：{sorted(missing_columns)}")  # 关键数据契约被破坏时立即终止。
    if not np.isfinite(  # 检查参与计算的数值是否全部有限。
        factor_data[[*forward_columns, "factor"]].to_numpy(np.float64)
    ).all():
        raise AssertionError("Alphalens 输出含 NaN 或 Inf")  # 关键数据契约被破坏时立即终止。

  # IC 和基准始终使用完整 T 日截面；只有口径1的组合与换手应用 T+1 掩码。
    execution_factor_data = select_execution_factor_data(  # 计算execution因子数据。
        factor_data, can_enter_t1, str(mode["name"])
    )

  # 必须保留完整交易日 level：Alphalens 从 level.freq 获取交易日频率。
  # remove_unused_levels() 会在存在缺失因子日时把频率清成 None，继而。
  # 让 asfreq(None) 退化为自然日并插入周末 NaN，使 22 日滚动均线全空。
    date_level = factor_data.index.names.index("date")  # 计算交易日level。
    factor_calendar_frequency = factor_data.index.levels[date_level].freq  # 计算因子交易日历frequency。
    execution_calendar_frequency = execution_factor_data.index.levels[date_level].freq  # 计算execution交易日历frequency。
    if factor_calendar_frequency is None:  # 检查所需对象是否尚未初始化。
        raise AssertionError("因子日期 level 缺少交易日频率")  # 关键数据契约被破坏时立即终止。
    if execution_calendar_frequency != factor_calendar_frequency:  # 按当前模式、标识或状态选择处理分支。
        raise AssertionError("完整因子与可交易视图的交易日频率不一致")  # 关键数据契约被破坏时立即终止。
    factor_calendar_frequency_name = type(factor_calendar_frequency).__name__  # 计算因子交易日历frequency名称。
    portfolio_rows_after_execution_filter = len(execution_factor_data)  # 计算portfolio行数afterexecutionfilter。
    execution_quantile_counts = (  # 按完整索引重排并保留缺项。
        execution_factor_data.groupby(
            [pd.Grouper(level="date"), "factor_quantile"], observed=True
        ).size().unstack("factor_quantile")
        .reindex(columns=range(1, quantiles + 1))
    )
    if execution_quantile_counts.isna().any(axis=None):  # 根据当前数据状态选择处理分支。
        raise AssertionError("执行过滤后存在空分位组，无法计算组合收益")  # 关键数据契约被破坏时立即终止。

  # 抽样手算，确认 Alphalens 与所选价格起点完全一致。
    audit_positions = np.unique(  # 计算audit位置。
        np.linspace(0, len(factor_data) - 1, min(25, len(factor_data)), dtype=np.int64)
    )
    audit_index = factor_data.index[audit_positions]  # 计算audit索引。
    audit_keys = audit_index.to_frame(index=False)  # 计算audit键。
    audit_rows = prices.index.get_indexer(audit_keys["date"])  # 从上游结果中读取audit行数。
    audit_columns = prices.columns.get_indexer(audit_keys["asset"])  # 从上游结果中读取audit列。
    formula_records = []  # 初始化formularecords容器。
    for period, column in zip(periods, forward_columns):  # 遍历zipperiodsforward列，逐项处理period列。
        manual = (  # 计算manual。
            price_values[  # 汇入price值。
                audit_rows + price_start_offset + period, audit_columns
            ]
            / price_values[audit_rows + price_start_offset, audit_columns]
            - 1
        )
        library = factor_data.loc[audit_index, column].to_numpy(np.float64)  # 转换并统一library的数据表示。
        np.testing.assert_allclose(library, manual, rtol=1e-12, atol=1e-12)  # 在数值容差内核对结果。
        formula_records.append(  # 把当前结果追加到汇总列表。
            {
                "period": column,  # 记录period。
                "price_start": price_point_label(price_start_offset),  # 记录价格起点。
                "price_end": price_point_label(price_start_offset + period),  # 记录价格end。
                "formula": forward_return_formula(price_start_offset),  # 记录formula。
                "samples_checked": len(audit_index),  # 记录样本checked。
                "max_abs_error": float(np.max(np.abs(library - manual))),  # 记录maxabs误差。
            }
        )

    daily_ic = al.performance.factor_information_coefficient(  # 固定日期与股票排序。
        factor_data, group_adjust=False, by_group=False  # 计算当前中间结果。
    ).sort_index()
    ic_nan_rows_by_period = {  # 初始化icnan行数byperiod容器。
        column: int(daily_ic[column].isna().sum()) for column in forward_columns  # 接收列。
    }
    if daily_ic.index.freq is None:  # 检查所需对象是否尚未初始化。
        raise AssertionError("日度 IC 索引缺少交易日频率")  # 关键数据契约被破坏时立即终止。
    if daily_ic.index.freq != factor_calendar_frequency:  # 按当前模式、标识或状态选择处理分支。
        raise AssertionError("日度 IC 与因子数据的交易日频率不一致")  # 关键数据契约被破坏时立即终止。
    ic_rolling_22 = daily_ic.rolling(window=22).mean()  # 计算icrolling22。
    ic_rolling_22_valid_rows_by_period = {  # 构造icrolling22有效行数byperiod，限定后续参与计算的样本。
        column: int(ic_rolling_22[column].notna().sum())  # 接收列。
        for column in forward_columns  # 遍历forward列，逐项处理列。
    }
    if any(count == 0 for count in ic_rolling_22_valid_rows_by_period.values()):  # 按当前模式、标识或状态选择处理分支。
        raise AssertionError("1 month moving avg 没有有效点")  # 关键数据契约被破坏时立即终止。

  # Alphalens 0.4.6 的均值/标准差会自动跳过 NaN，但 SciPy 统计量。
  # 默认传播 NaN。这里保持原表格结构，只把后四项改为逐列忽略缺失值。
    def plot_information_table_omit_nan(ic_data: Any, return_df: Any=False) -> Any:  # 生成并保存该评价环节对应的图表，缺失指标按既定口径处理。
        """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。"""
        values = ic_data.to_numpy(np.float64)  # 转换并统一值的数据表示。
        table = pd.DataFrame(index=ic_data.columns)  # 把累计记录整理为统计表。
        table["IC Mean"] = ic_data.mean()  # 计算表。
        table["IC Std."] = ic_data.std()  # 计算表。
        table["Risk-Adjusted IC"] = table["IC Mean"] / table["IC Std."]  # 从上游结果中读取统计表。
        test = stats.ttest_1samp(  # 检验日度指标均值是否显著。
            values, popmean=0.0, axis=0, nan_policy="omit"  # 计算当前中间结果。
        )
        table["t-stat(IC)"] = test.statistic  # 计算统计表。
        table["p-value(IC)"] = test.pvalue  # 计算统计表。
        table["IC Skew"] = stats.skew(values, axis=0, nan_policy="omit")  # 统计因子偏度。
        table["IC Kurtosis"] = stats.kurtosis(  # 统计因子峰度。
            values, axis=0, nan_policy="omit"  # 计算当前中间结果。
        )
        if return_df:  # 根据当前数据状态选择处理分支。
            return table  # 返回统计表。
        print("Information Analysis")  # 输出当前任务的进度或诊断信息。
        al.utils.print_table(table.apply(lambda column: column.round(3)).T)

    ic_records, annual_records = [], []  # 初始化icrecords容器。
    for column in forward_columns:  # 遍历forward列，逐项处理列。
        values = daily_ic[column].dropna()  # 剔除缺失值后计算。
        test = stats.ttest_1samp(values.to_numpy(), popmean=0.0)  # 检验日度指标均值是否显著。
        ic_records.append(  # 把当前结果追加到汇总列表。
            {
                "period": column,  # 记录period。
                "days": len(values),  # 记录days。
                "mean_ic": float(values.mean()),  # 记录均值ic。
                "median_ic": float(values.median()),  # 记录medianic。
                "std_ic": float(values.std(ddof=1)),  # 记录标准差ic。
                "positive_ratio": float(values.gt(0).mean()),  # 记录positive比例。
                "t_stat": float(test.statistic),  # 记录tstat。
                "p_value": float(test.pvalue),  # 记录p值。
            }
        )
        for year, annual in values.groupby(values.index.year):  # 遍历值groupby值索引年份，逐项处理年份annual。
            annual_records.append(  # 把当前结果追加到汇总列表。
                {
                    "year": int(year), "period": column, "days": len(annual),  # 记录年份。
                    "mean_ic": float(annual.mean()),  # 记录均值ic。
                    "median_ic": float(annual.median()),  # 记录medianic。
                    "std_ic": float(annual.std(ddof=1)),  # 记录标准差ic。
                    "positive_ratio": float(annual.gt(0).mean()),  # 记录positive比例。
                }
            )
    ic_summary = pd.DataFrame(ic_records)  # 把累计记录整理为ic汇总。
    annual_ic = pd.DataFrame(annual_records)  # 把累计记录整理为annualic。
    information_table_audit = plot_information_table_omit_nan(  # 按完整索引重排并保留缺项。
        daily_ic, return_df=True  # 计算当前中间结果。
    ).reindex(forward_columns)
    if not np.isfinite(information_table_audit.to_numpy(np.float64)).all():  # 检查参与计算的数值是否全部有限。
        raise AssertionError("Information Analysis 统计量仍含 NaN 或 Inf")  # 关键数据契约被破坏时立即终止。
    np.testing.assert_allclose(  # 在数值容差内核对结果。
        information_table_audit["t-stat(IC)"].to_numpy(np.float64),
        ic_summary["t_stat"].to_numpy(np.float64),
        rtol=1e-12,  # 配置rtol。
        atol=1e-12,  # 配置atol。
    )
    np.testing.assert_allclose(  # 在数值容差内核对结果。
        information_table_audit["p-value(IC)"].to_numpy(np.float64),
        ic_summary["p_value"].to_numpy(np.float64),
        rtol=1e-12,  # 配置rtol。
        atol=1e-12,  # 配置atol。
    )
    mean_by_quantile, _ = al.performance.mean_return_by_quantile(  # 计算均值byquantile。
        execution_factor_data,  # 汇入execution因子数据。
        by_date=False,  # 配置by交易日。
        by_group=False,  # 配置bygroup。
        demeaned=False,  # 配置demeaned。
        group_adjust=False,  # 配置groupadjust。
    )
    mean_by_date, standard_error_by_date = al.performance.mean_return_by_quantile(  # 计算均值by交易日。
        execution_factor_data,  # 汇入execution因子数据。
        by_date=True,  # 配置by交易日。
        by_group=False,  # 配置bygroup。
        demeaned=False,  # 配置demeaned。
        group_adjust=False,  # 配置groupadjust。
    )
    daily_spread, _ = al.performance.compute_mean_returns_spread(  # 计算当前中间结果。
        mean_by_date, upper_quant=quantiles, lower_quant=1,  # 计算均值by交易日。
        std_err=standard_error_by_date,  # 配置标准差err。
    )
    quantile_return_summary = mean_by_quantile.reset_index().rename(  # 提取并重命名quantile收益汇总。
        columns={"factor_quantile": "quantile"}  # 初始化列容器。
    )
    spread_summary = pd.DataFrame(  # 把累计记录整理为spread汇总。
        {
            "period": forward_columns,  # 记录period。
            "Q5_minus_Q1_return": [float(daily_spread[column].mean()) for column in forward_columns],  # 记录q5minusq1收益。
            "standard_error": [float(daily_spread[column].sem(ddof=1)) for column in forward_columns],  # 记录standard误差。
        }
    )
    daily_quantile_returns = mean_by_date["1D"].unstack("factor_quantile").sort_index()  # 计算dailyquantile收益以固定样本顺序。
    equal_weight_market = compute_equal_weight_market_returns(factor_data)  # 计算equal权重market。
    portfolio_returns = pd.concat(  # 对齐并合并portfolio收益。
        {
            "Q5 long-only": daily_quantile_returns[quantiles],  # 记录q5 long-only。
            "Q5-Q1 long-short": (  # 记录q5-q1 long-short。
                daily_quantile_returns[quantiles] - daily_quantile_returns[1]
            ),
            EQUAL_WEIGHT_MARKET_COLUMN: equal_weight_market,  # 接收equal权重market列。
        },
        axis=1,  # 指定数组运算轴。
    )
    if portfolio_returns.isna().any(axis=None):  # 根据当前数据状态选择处理分支。
        raise AssertionError("组合收益与全市场等权基准日期未完全对齐")  # 关键数据契约被破坏时立即终止。
    relative_performance = compute_relative_performance(portfolio_returns)  # 计算relativeperformance。
    pyfolio_performance = pd.concat(  # 对齐并合并pyfolioperformance。
        {
            column: pf.timeseries.perf_stats(portfolio_returns[column])  # 接收列。
            for column in portfolio_returns  # 遍历portfolio收益，逐项处理列。
        },
        axis=1,  # 指定数组运算轴。
    )

    package_names = [  # 初始化package名称容器。
        "alphalens-reloaded", "pyfolio-reloaded", "pandas", "numpy", "pyarrow"
    ]
    installed_versions = {  # 初始化installedversions容器。
        package: metadata.version(package) for package in package_names  # 接收package。
    }
    final_audit = pd.DataFrame(  # 把累计记录整理为finalaudit。
        {
            "item": [  # 记录item。
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
            "value": [  # 记录值。
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

    atomic_parquet(prediction_audit, evaluation_dir / "prediction_universe_audit.parquet")  # 原子落盘，避免半成品。
    atomic_parquet(price_coverage, evaluation_dir / "price_coverage.parquet")  # 原子落盘，避免半成品。
    atomic_parquet(pd.DataFrame(formula_records), evaluation_dir / "return_formula_audit.parquet")  # 原子落盘，避免半成品。
    atomic_parquet(ic_summary, evaluation_dir / "ic_summary.parquet")  # 原子落盘，避免半成品。
    atomic_parquet(annual_ic, evaluation_dir / "annual_ic_summary.parquet")  # 原子落盘，避免半成品。
    atomic_parquet(quantile_return_summary, evaluation_dir / "quantile_return_summary.parquet")  # 原子落盘，避免半成品。
    atomic_parquet(spread_summary, evaluation_dir / "spread_summary.parquet")  # 原子落盘，避免半成品。
    atomic_parquet(portfolio_returns.reset_index(), evaluation_dir / "portfolio_returns.parquet")  # 原子落盘，避免半成品。
    atomic_parquet(  # 原子落盘，避免半成品。
        pyfolio_performance.rename_axis("metric").reset_index(),
        evaluation_dir / "pyfolio_performance.parquet",
    )
    atomic_parquet(  # 原子落盘，避免半成品。
        relative_performance, evaluation_dir / "relative_performance.parquet"
    )
    atomic_parquet(final_audit, evaluation_dir / "final_audit.parquet")  # 原子落盘，避免半成品。
    evaluation_manifest = {  # 初始化evaluation清单容器。
        "schema_version": 2,  # 记录schema版本。
        "run_name": run_name,  # 记录运行名称。
        "prediction_years": list(prediction_years),  # 记录预测年份。
        "dataset_signature": dataset_manifest["signature"],  # 记录dataset签名，用于来源一致性校验。
        "universe_signature": universe_manifest["signature"],  # 记录股票池签名，用于来源一致性校验。
        "input_prediction_rows": int(prediction_audit["prediction_rows"].sum()),  # 记录输入预测行数。
        "t1_excluded_rows": int(prediction_audit["t1_excluded_rows"].sum()),  # 记录t1excluded行数。
        "factor_rows_before_price_filter": t_grouping_rows_before_price_filter,  # 记录因子行数before价格filter。
        "clean_factor_rows": t_grouping_rows_after_price_filter,  # 记录clean因子行数。
        "t_day_grouping_rows_before_price_filter": t_grouping_rows_before_price_filter,  # 记录tday分组行数before价格filter。
        "t_day_grouping_rows_after_price_filter": t_grouping_rows_after_price_filter,  # 记录tday分组行数after价格filter。
        "portfolio_rows_after_execution_filter": portfolio_rows_after_execution_filter,  # 记录portfolio行数after执行filter。
        "backtest_mode": str(mode["name"]),  # 记录backtest模式。
        "grouping_policy": "T_full_cross_section_before_execution_filter",  # 把 grouping_policy 映射为"T_full_cross_section_before_execution_filter"。
        "ic_scope": "T_full_cross_section_with_valid_forward_prices",  # 把 ic_scope 映射为"T_full_cross_section_with_valid_forward_prices"。
        "portfolio_scope": str(mode["portfolio_scope"]),  # 记录portfolioscope。
        "execution_mask": str(mode["execution_mask"]),  # 记录执行掩码。
        "factor_dates": int(factor_data.index.get_level_values("date").nunique()),  # 记录因子交易日。
        "factor_calendar_frequency": factor_calendar_frequency_name,  # 记录因子交易日历frequency。
        "ic_rolling_22_valid_rows_by_period": ic_rolling_22_valid_rows_by_period,  # 记录icrolling22有效行数byperiod。
        "price_column": price_column,  # 记录价格字段。
        "price_start": str(mode["price_start"]),  # 记录价格起点。
        "price_start_offset": price_start_offset,  # 记录价格起点偏移。
        "forward_return_formula": forward_return_formula(price_start_offset),  # 记录forward收益formula。
        "periods": list(periods),  # 记录periods。
        "quantiles": quantiles,  # 记录quantiles。
        "max_loss": max_loss,  # 记录max损失。
        "benchmark_column": EQUAL_WEIGHT_MARKET_COLUMN,  # 记录benchmark字段。
        "benchmark_scope": "feature_valid_and_in_universe_t_full_factor_pool_after_required_price_filter",  # 把 benchmark_scope 映射为"feature_valid_and_in_universe_t_full_factor_pool_after_required_price_filter"。
        "benchmark_weighting": "daily_equal_weight",  # 把 benchmark_weighting 映射为"daily_equal_weight"。
        "benchmark_execution_mask": "none",  # 把 benchmark_execution_mask 映射为"none"。
        "annual_excess_method": "portfolio_geometric_annual_return_minus_benchmark_geometric_annual_return",  # 把 annual_excess_method 映射为"portfolio_geometric_annual_return_minus_benchmark_geometric_annual_return"。
        "information_ratio_method": "mean_daily_active_return_div_daily_tracking_error_times_sqrt_252",  # 把 information_ratio_method 映射为"mean_daily_active_return_div_daily_tracking_error_times_sqrt_252"。
        "annual_tracking_error_method": "std_daily_active_return_ddof_1_times_sqrt_252",  # 把 annual_tracking_error_method 映射为"std_daily_active_return_ddof_1_times_sqrt_252"。
        "annualization_days": ANNUALIZATION_DAYS,  # 记录annualizationdays。
        "long_short_gross_exposure": 2.0,  # 记录longshortgross暴露。
        "information_table_nan_policy": "omit",  # 把 information_table_nan_policy 映射为"omit"。
        "ic_nan_rows_by_period": ic_nan_rows_by_period,  # 记录icnan行数byperiod。
        "plot_font": "Noto Sans CJK SC",  # 把 plot_font 映射为"Noto Sans CJK SC"。
        "versions": installed_versions,  # 记录versions。
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
    pic_dir = evaluation_dir / "pic"  # 构造pic目录。
    pic_dir.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。

    portfolio_net_values = (1 + portfolio_returns).cumprod()  # 复利累计得到净值。
    price_start_title = "T" if price_start_offset == 0 else "T+1"  # 配置price起点title。
    cumulative_ax = portfolio_net_values.plot(  # 计算cumulativeax。
        figsize=(12, 5),  # 配置figsize。
        title=f"A2 因子组合与全市场等权累计毛收益（{price_start_title} 日复权收盘起算）",  # 配置title。
    )
    cumulative_ax.grid(alpha=0.25)
    cumulative_figure = cumulative_ax.get_figure()  # 计算cumulativefigure。
    cumulative_path = evaluation_dir / "portfolio_cumulative_returns.png"  # 构造cumulative路径。
    cumulative_figure.savefig(cumulative_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"累计收益图：{cumulative_path}")  # 输出当前任务的进度或诊断信息。
    plt.close(cumulative_figure)

  # 回撤按净值相对历史峰值计算；图例同时标出各组合最大回撤。
    portfolio_drawdowns = portfolio_net_values.div(  # 计算portfoliodrawdowns。
        portfolio_net_values.cummax().clip(lower=1.0)
    ).sub(1)
    np.testing.assert_allclose(  # 在数值容差内核对结果。
        portfolio_drawdowns.min().to_numpy(np.float64),
        pyfolio_performance.loc["Max drawdown"].to_numpy(np.float64),
        rtol=1e-12,  # 配置rtol。
        atol=1e-12,  # 配置atol。
    )
    drawdown_plot = portfolio_drawdowns.copy()  # 转换drawdownplot的数据类型或存储位置。
    drawdown_plot.columns = [
        f"{column}（最大回撤 {portfolio_drawdowns[column].min():.2%}）"
        for column in portfolio_drawdowns  # 遍历portfoliodrawdowns，逐项处理列。
    ]
    drawdown_ax = drawdown_plot.plot(  # 计算drawdownax。
        figsize=(12, 5), title="A2 因子组合回撤曲线"  # 计算figsize。
    )
    drawdown_ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
    drawdown_ax.set_ylabel("回撤")
    drawdown_ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    drawdown_ax.grid(alpha=0.25)
    drawdown_figure = drawdown_ax.get_figure()  # 计算drawdownfigure。
    drawdown_path = pic_dir / "portfolio_drawdown.png"  # 构造drawdown路径。
    drawdown_figure.savefig(drawdown_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"最大回撤图：{drawdown_path}")  # 输出当前任务的进度或诊断信息。
    plt.close(drawdown_figure)

  # Alphalens 在每个 tear sheet 的 show() 之后立即关闭 figure；。
  # 因此在 show 时先保存，再继续让 Notebook 正常内嵌显示。
    for old_path in pic_dir.glob("alphalens_*_tear_sheet*.png"):  # 遍历pic目录globalphalenstearsheetpng，逐项处理old路径。
        old_path.unlink()
    tear_sheet_sections = ("returns", "information", "turnover")  # 计算tearsheetsections。
    tear_sheet_files: list[str] = []  # 接收tearsheetfiles。
    original_show = plt.show  # 计算原始show。
    original_information_table = al.plotting.plot_information_table  # 计算原始information表。
    section_number = 0  # 计算sectionnumber。

    def save_tear_sheet_figures(*show_args: Any, **show_kwargs: Any) -> None:  # 生成并保存该评价环节对应的图表，缺失指标按既定口径处理。
        """生成并保存该评价环节对应的图表，缺失指标按既定口径处理。"""
        nonlocal section_number  # 声明后续赋值使用外层作用域变量。
        figures = []  # 初始化figures容器。
        for number in plt.get_fignums():  # 遍历pltgetfignums，逐项处理number。
            figure = plt.figure(number)  # 计算figure。
            if figure.axes:  # 根据当前数据状态选择处理分支。
                figures.append(figure)  # 把当前结果追加到汇总列表。
        if not figures:  # 检查必需数据、配置或校验结果是否缺失。
            raise RuntimeError("Alphalens tear sheet 未生成 figure")  # 当前运行环境或任务状态不支持继续执行。
        section = (  # 计算section。
            tear_sheet_sections[section_number]  # 汇入tearsheetsections。
            if section_number < len(tear_sheet_sections)  # 检查当前样本数量是否满足计算要求。
            else f"section_{section_number + 1}"
        )
        for figure_number, figure in enumerate(figures, start=1):  # 遍历enumeratefigures起点1，逐项处理figurenumberfigure。
            suffix = "" if len(figures) == 1 else f"_{figure_number}"  # 配置suffix。
            path = pic_dir / f"alphalens_{section}_tear_sheet{suffix}.png"  # 拼接得到路径路径。
            figure.savefig(path, dpi=150, bbox_inches="tight")
            tear_sheet_files.append(str(path.relative_to(evaluation_dir)))  # 把当前结果追加到汇总列表。
        original_show(*show_args, **show_kwargs)
        section_number += 1

    plt.close("all")
    plt.show = save_tear_sheet_figures
    al.plotting.plot_information_table = plot_information_table_omit_nan
    try:  # 执行可能失败的外部读取或计算步骤。
  # information 始终看完整 T 日截面；returns/turnover 使用模式对应执行视图。
        al.plotting.plot_quantile_statistics_table(factor_data)
        al.tears.create_returns_tear_sheet(
            execution_factor_data,  # 汇入execution因子数据。
            long_short=True,  # 配置longshort。
            group_neutral=False,  # 配置groupneutral。
            by_group=False,  # 配置bygroup。
        )
        al.tears.create_information_tear_sheet(
            factor_data,  # 汇入因子数据。
            group_neutral=False,  # 配置groupneutral。
            by_group=False,  # 配置bygroup。
        )
        al.tears.create_turnover_tear_sheet(execution_factor_data)
    finally:  # 无论前序步骤是否成功都执行资源清理。
        plt.show = original_show
        al.plotting.plot_information_table = original_information_table
        plt.close("all")
    if section_number != len(tear_sheet_sections):  # 检查当前样本数量是否满足计算要求。
        raise AssertionError(  # 关键数据契约被破坏时立即终止。
            f"Alphalens tear sheet 数量异常：{section_number}"
        )
    evaluation_manifest["plot_files"] = [  # 初始化evaluation清单容器。
        str(cumulative_path.relative_to(evaluation_dir)),
        str(drawdown_path.relative_to(evaluation_dir)),
        *tear_sheet_files,
    ]
    evaluation_manifest["drawdown_file"] = str(  # 计算evaluation清单。
        drawdown_path.relative_to(evaluation_dir)
    )
    evaluation_manifest["tear_sheet_files"] = tear_sheet_files  # 计算evaluation清单。
    evaluation_manifest["tear_sheet_scopes"] = {  # 初始化evaluation清单容器。
        "returns": str(mode["portfolio_scope"]),  # 记录收益。
        "information": "T_full_cross_section_with_valid_forward_prices",  # 把 information 映射为"T_full_cross_section_with_valid_forward_prices"。
        "turnover": str(mode["portfolio_scope"]),  # 记录turnover。
    }
    atomic_json(  # 原子落盘，避免半成品。
        evaluation_manifest,  # 汇入evaluation清单。
        evaluation_dir / "factor_evaluation_manifest.json",
    )
    print(f"Alphalens tear sheet：{pic_dir}")  # 输出当前任务的进度或诊断信息。

    del factor, factor_quantile, can_enter_t1  # 释放不再使用的中间引用。
    del factor_data, execution_factor_data  # 释放不再使用的中间引用。
    del prices, return_prices, price_values, eod  # 释放不再使用的中间引用。
    gc.collect()
    generate_experiment_report(run_name=run_name, data_root=root)
    print(  # 输出当前任务的进度或诊断信息。
        f"样本外因子评价完成；模式={mode['name']}，"
        f"价格起点={price_point_label(price_start_offset)}。"
    )
    return final_audit  # 返回finalaudit。


# ============================ 实验报告 ============================
def _markdown_table(frame: pd.DataFrame) -> str:  # 避免引入额外模板依赖的小型 Markdown 表格。
    """避免引入额外模板依赖的小型 Markdown 表格。"""

    if frame.empty:  # 检查结果表是否为空。
        return "（暂无记录）"  # 返回避免引入额外模板依赖的小型 Markdown 表格。
    clean = frame.copy().fillna("")  # 转换clean的数据类型或存储位置。
    headers = [str(column) for column in clean.columns]  # 初始化headers容器。
    rows = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]  # 对齐并合并行数。
    for record in clean.astype(str).itertuples(index=False, name=None):  # 遍历cleanastypestritertuples索引false名称none，逐项处理record。
        rows.append("| " + " | ".join(value.replace("|", "\\|") for value in record) + " |")  # 把当前结果追加到汇总列表。
    return "\n".join(rows)  # 返回避免引入额外模板依赖的小型 Markdown 表格。


def generate_experiment_report(  # 根据指定 run 的 manifest 和审计表生成中文实验报告。
    run_name: str = DEFAULT_A2_RUN_NAME,  # 接收执行名称。
    data_root: str | Path | None = None,  # 接收数据根目录。
) -> Path:  # 声明当前函数的返回类型。
    """根据指定 run 的 manifest 和审计表生成中文实验报告。"""

    run_name = validate_run_name(run_name)
    root = (
        Path(data_root).resolve()
        if data_root is not None  # 检查必需数据、配置或校验结果是否缺失。
        else get_data_root().resolve()
    )
    if not root.is_dir():  # 检查所需文件或目录是否存在。
        raise FileNotFoundError(root)  # 目标文件不存在时立即终止。
    quality_run = root / "quality" / "runs" / run_name
    universe_manifest = read_json(root / "universe" / "universe_manifest.json")  # 读取股票池清单。
    label_manifest = read_json(root / "labels_t10" / "labels_manifest.json")  # 读取标签清单。
    dataset_manifest = read_json(root / "feature_store" / "dataset_manifest.json")  # 读取dataset清单。
    rolling_path = root / "feature_store" / "rolling_split_summary.parquet"  # 构造rolling路径。
    if universe_manifest is None or label_manifest is None or dataset_manifest is None:  # 检查所需对象是否尚未初始化。
        raise FileNotFoundError("Universe、标签或 feature store manifest 尚未生成")  # 目标文件不存在时立即终止。
    universe_quality = pd.read_parquet(  # 读取股票池质量。
        root / "quality" / "universe_quality_by_date.parquet"
    )
    universe_quality["year"] = universe_quality["trade_date"].astype(str).str[:4]  # 转换股票池质量的数据类型或存储位置。
    universe_yearly = universe_quality.groupby("year", as_index=False).agg(  # 按既定口径聚合。
        candidate_rows=("candidate_rows", "sum"),  # 配置candidate行数。
        in_universe_t_rows=("in_universe_t_rows", "sum"),  # 配置in股票池t行数。
        prediction_eligible_rows=("prediction_eligible_rows", "sum"),  # 配置预测eligible行数。
        label_candidate_rows=("label_candidate_rows", "sum"),  # 配置标签candidate行数。
        missing_status_t_rows=("missing_status_t_rows", "sum"),  # 配置missingstatust行数。
        suspended_t_rows=("suspended_t_rows", "sum"),  # 配置suspendedt行数。
        limit_up_t_rows=("limit_up_t_rows", "sum"),  # 配置limitupt行数。
        limit_down_t_rows=("limit_down_t_rows", "sum"),  # 配置limitdownt行数。
    )
    label_quality = pd.read_parquet(root / "quality" / "label_quality_by_date.parquet")  # 读取标签质量。
    label_quality["year"] = label_quality["trade_date"].astype(str).str[:4]  # 转换标签质量的数据类型或存储位置。
    label_yearly = label_quality.groupby("year", as_index=False).agg(  # 按既定口径聚合。
        label_candidate_rows=("label_candidate_rows", "sum"),  # 配置标签candidate行数。
        label_valid_rows=("label_valid_rows", "sum"),  # 配置标签有效行数。
        missing_or_nonpositive_price_rows=("missing_or_nonpositive_price_rows", "sum"),  # 配置missingornonpositiveprice行数。
    )
    rolling = pd.read_parquet(rolling_path)[  # 读取rolling。
        ["forecast_year", "train_sample_count", "validation_sample_count", "prediction_sample_count"]
    ]
    training_path = quality_run / "vit_training_summary.parquet"  # 构造训练路径。
    if training_path.exists():  # 检查目标文件或目录是否满足存在性要求。
        training = pd.read_parquet(training_path)  # 读取训练。
        aggregations = {  # 初始化aggregations容器。
            "completed_seeds": ("seed", "nunique"),  # 记录completed随机种子。
            "mean_best_validation_ic": ("best_validation_ic", "mean"),  # 记录均值最优验证ic。
            "min_best_validation_ic": ("best_validation_ic", "min"),  # 记录min最优验证ic。
            "mean_best_epoch": ("best_epoch", "mean"),  # 记录均值最优轮次。
        }
        if "best_validation_rank_ic" in training.columns:  # 检查当前值是否属于允许集合。
            aggregations["mean_best_validation_rank_ic"] = (  # 计算aggregations。
                "best_validation_rank_ic", "mean"
            )
        if "best_validation_daily_pearson_ic" in training.columns:  # 检查当前值是否属于允许集合。
            aggregations["mean_best_validation_daily_pearson_ic"] = (  # 计算aggregations。
                "best_validation_daily_pearson_ic", "mean"
            )
        elif "best_validation_daily_ic" in training.columns:  # 检查上一条件不成立时的备选分支。
            aggregations["mean_best_validation_daily_pearson_ic"] = (  # 计算aggregations。
                "best_validation_daily_ic", "mean"
            )
        if "best_validation_pooled_ic" in training.columns:  # 检查当前值是否属于允许集合。
            aggregations["mean_best_validation_pooled_ic"] = (  # 计算aggregations。
                "best_validation_pooled_ic", "mean"
            )
        training_yearly = training.groupby(  # 按既定口径聚合。
            "forecast_year", as_index=False
        ).agg(**aggregations)
    else:  # 处理上述条件均不成立的情况。
        training_yearly = pd.DataFrame()
    evaluation_dir = quality_run / "factor_evaluation"  # 构造evaluation目录。
    ic_path = evaluation_dir / "ic_summary.parquet"  # 构造ic路径。
    spread_path = evaluation_dir / "spread_summary.parquet"  # 构造spread路径。
    audit_path = evaluation_dir / "final_audit.parquet"  # 构造audit路径。
    pyfolio_path = evaluation_dir / "pyfolio_performance.parquet"  # 构造pyfolio路径。
    relative_path = evaluation_dir / "relative_performance.parquet"  # 构造relative路径。
    ic_summary = pd.read_parquet(ic_path) if ic_path.exists() else pd.DataFrame()  # 读取ic汇总。
    spread_summary = pd.read_parquet(spread_path) if spread_path.exists() else pd.DataFrame()  # 读取spread汇总。
    final_audit = pd.read_parquet(audit_path) if audit_path.exists() else pd.DataFrame()  # 读取finalaudit。
    pyfolio = (
        pd.read_parquet(pyfolio_path) if pyfolio_path.exists() else pd.DataFrame()
    )
    relative_performance = (
        pd.read_parquet(relative_path) if relative_path.exists() else pd.DataFrame()
    )
    evaluation_manifest = read_json(  # 读取evaluation清单。
        evaluation_dir / "factor_evaluation_manifest.json"
    )
    if evaluation_manifest:  # 根据当前数据状态选择处理分支。
        price_start_offset = int(evaluation_manifest.get("price_start_offset", 0))
        period_description = (
            "1/5/10D 对应 T→T+1/T+5/T+10"
            if price_start_offset == 0  # 按当前模式、标识或状态选择处理分支。
            else "1/5/10D 对应 T+1→T+2/T+6/T+11"
        )
        execution_description = (
            "组合收益和换手在完整 T 日分组后应用 `can_enter_t1`"
            if evaluation_manifest.get("execution_mask") == "can_enter_t1"  # 按当前模式、标识或状态选择处理分支。
            else "组合收益和换手不应用 `can_enter_t1` 未来状态过滤"
        )
        evaluation_policy_lines = [  # 初始化evaluationpolicylines容器。
            f"- 回测模式：`{evaluation_manifest.get('backtest_mode')}`；{execution_description}。",
            f"- 收益公式：`{evaluation_manifest.get('forward_return_formula')}`；{period_description}。",
            "- 全市场等权基准使用完整因子覆盖池和相同收益时点，不应用 `can_enter_t1`。",
            "- Q5 与 Q5−Q1 的超额年化为组合几何年化减基准几何年化；信息比使用日度主动收益并按 252 日年化。",
        ]
    else:  # 处理上述条件均不成立的情况。
        evaluation_policy_lines = ["- 因子评价尚未按新版回测口径生成。"]  # 初始化evaluationpolicylines容器。
    training_manifest = read_json(quality_run / "a2_training_manifest.json")  # 读取训练清单。
    batch_summary = pd.DataFrame(
        training_manifest.get("training_batch_size_by_year", [])
        if training_manifest else []  # 根据当前数据状态选择处理分支。
    )

    lines = [  # 初始化lines容器。
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
    if training_manifest:  # 根据当前数据状态选择处理分支。
        train_batch_mode = training_manifest.get(
            "train_batch_mode", "fixed_samples"
        )
        inference_batch_size = training_manifest.get(
            "inference_batch_size",
            training_manifest.get("batch_size_per_task"),
        )
        validation_metric = training_manifest.get(  # 计算验证指标。
            "validation_metric", "pooled_pearson_ic"
        )
        loss_name = training_manifest.get("loss_name", "pearson_ic")  # 计算损失名称。
        loss_parameters = training_manifest.get("loss_parameters", {})  # 计算损失parameters。
        lines.extend(  # 把一组结果并入当前列表。
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
    else:  # 处理上述条件均不成立的情况。
        lines.extend(["训练尚未完成。", ""])  # 把一组结果并入当前列表。
    lines.extend(  # 把一组结果并入当前列表。
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
    quality_run.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在。
    report_path = quality_run / "EXPERIMENT_REPORT.md"  # 构造报告路径。
    temporary = report_path.with_name(f"{report_path.name}.{os.getpid()}.tmp")  # 构造临时文件。
    temporary.write_text("\n".join(lines), encoding="utf-8")  # 把文本结果写入临时或目标文件。
    temporary.replace(report_path)  # 用临时文件原子替换正式结果。
    print(f"实验报告：{report_path}")  # 输出当前任务的进度或诊断信息。
    return report_path  # 返回报告路径。


# ============================== 评价 CLI ==============================
def add_evaluation_arguments(parser: argparse.ArgumentParser, *, include_backtest_mode: bool = False) -> None:  # 向命令行解析器注册样本外评价参数。
    """向命令行解析器注册样本外评价参数。"""
    parser.add_argument("--data-root", default=str(VIT_DIR / "data"))  # 注册命令行参数及其默认值。
    parser.add_argument("--run-name", default=DEFAULT_A2_RUN_NAME)  # 注册命令行参数及其默认值。
    if include_backtest_mode:  # 根据当前数据状态选择处理分支。
        parser.add_argument("--backtest-mode", choices=BACKTEST_MODES, default=DEFAULT_BACKTEST_MODE)  # 注册命令行参数及其默认值。


def parse_args() -> argparse.Namespace:  # 解析阶段 08 的回测或报告命令。
    """解析阶段 08 的回测或报告命令。"""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    add_evaluation_arguments(commands.add_parser("backtest"), include_backtest_mode=True)  # 无梯度计算验证指标。
    add_evaluation_arguments(commands.add_parser("report"))  # 无梯度计算验证指标。
    return parser.parse_args()  # 返回解析阶段 08 的回测或报告命令。


def main() -> None:  # 分派样本外回测或实验报告生成流程。
    """分派样本外回测或实验报告生成流程。"""
    args = parse_args()
    if args.command == "backtest":  # 按命令行子命令选择执行路径。
        run_oos_backtest(run_name=args.run_name, data_root=args.data_root, backtest_mode=args.backtest_mode)
    elif args.command == "report":  # 选择 report 子命令的处理路径。
        generate_experiment_report(run_name=args.run_name, data_root=args.data_root)


if __name__ == "__main__":  # 按当前模式、标识或状态选择处理分支。
    main()
