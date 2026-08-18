"""ViViT 样本外因子评价；只读复用 ViT 评价引擎且不依赖 PyTorch。"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

# 读取 `Path(__file__).resolve().parent` 的当前值，并赋给 `VIVIT_DIR`。
VIVIT_DIR = Path(__file__).resolve().parent
# 读取 `VIVIT_DIR.parent` 的当前值，并赋给 `PROJECT_ROOT`。
PROJECT_ROOT = VIVIT_DIR.parent
# 计算 `PROJECT_ROOT / 'ViT'`，并将数值结果保存到 `VIT_DIR`。
VIT_DIR = PROJECT_ROOT / "ViT"
# 计算 `VIVIT_DIR / 'data'`，并将数值结果保存到 `DEFAULT_VIVIT_DATA_ROOT`。
DEFAULT_VIVIT_DATA_ROOT = VIVIT_DIR / "data"
# 计算 `VIT_DIR / 'data'`，并将数值结果保存到 `DEFAULT_REFERENCE_DATA_ROOT`。
DEFAULT_REFERENCE_DATA_ROOT = VIT_DIR / "data"
# 设置 `DEFAULT_RUN_NAME` 的固定配置值，供后续流程统一使用。
DEFAULT_RUN_NAME = "sigmoid_daybatch_rankic"
# 设置 `PREDICTION_PATTERN` 的固定配置值，供后续流程统一使用。
PREDICTION_PATTERN = "vivit_predictions.{year}.parquet"
# 设置 `FACTOR_COLUMN` 的固定配置值，供后续流程统一使用。
FACTOR_COLUMN = "prediction_ensemble"
# 组织一组有序值，并保存到 `PREDICTION_COLUMNS`。
PREDICTION_COLUMNS = (
    "prediction_ensemble",
    "prediction_ensemble_unprojected",
    "prediction_ensemble_projected",
    "prediction_ensemble_raw",
    "prediction_ensemble_alpha",
    "prediction_ensemble_style",
    "prediction_ensemble_total_return",
)
# 构造键值映射，并保存到 `EVALUATION_DIRECTORY_NAMES`，供后续按键查询。
EVALUATION_DIRECTORY_NAMES = {
    "prediction_ensemble": "factor_evaluation",
    "prediction_ensemble_unprojected": "factor_evaluation_unprojected",
    "prediction_ensemble_projected": "factor_evaluation_projected",
    "prediction_ensemble_raw": "factor_evaluation_raw",
    "prediction_ensemble_alpha": "factor_evaluation_alpha",
    "prediction_ensemble_style": "factor_evaluation_style",
    "prediction_ensemble_total_return": "factor_evaluation_total_return",
}
# 构造键值映射，并保存到 `EXPERIMENT_REPORT_NAMES`，供后续按键查询。
EXPERIMENT_REPORT_NAMES = {
    "prediction_ensemble": "EXPERIMENT_REPORT.md",
    "prediction_ensemble_unprojected": "EXPERIMENT_REPORT_unprojected.md",
    "prediction_ensemble_projected": "EXPERIMENT_REPORT_projected.md",
    "prediction_ensemble_raw": "EXPERIMENT_REPORT_raw.md",
    "prediction_ensemble_alpha": "EXPERIMENT_REPORT_alpha.md",
    "prediction_ensemble_style": "EXPERIMENT_REPORT_style.md",
    "prediction_ensemble_total_return": "EXPERIMENT_REPORT_total_return.md",
}
# 组织一组有序值，并保存到 `PERIODS`。
PERIODS = (1, 5, 10)
# 设置 `QUANTILES` 的固定配置值，供后续流程统一使用。
QUANTILES = 5
# 设置 `MAX_LOSS` 的固定配置值，供后续流程统一使用。
MAX_LOSS = 0.001
# 组织一组有序值，并保存到 `BACKTEST_MODES`。
BACKTEST_MODES = (
    "mode1_t_with_t1_filter",
    "mode2_t_no_filter",
    "mode3_t1_no_filter",
)
# 设置 `DEFAULT_BACKTEST_MODE` 的固定配置值，供后续流程统一使用。
DEFAULT_BACKTEST_MODE = "mode2_t_no_filter"
# 设置 `GROUPING_POLICY` 的固定配置值，供后续流程统一使用。
GROUPING_POLICY = "T_full_cross_section_before_execution_filter"
# 设置 `EQUAL_WEIGHT_MARKET_COLUMN` 的固定配置值，供后续流程统一使用。
EQUAL_WEIGHT_MARKET_COLUMN = "Equal-weight market"
# 设置 `BENCHMARK_SCOPE` 的固定配置值，供后续流程统一使用。
BENCHMARK_SCOPE = (
    "feature_valid_and_in_universe_t_full_factor_pool_after_required_price_filter"
)
# 构造键值映射，并保存到 `BACKTEST_MODE_CONFIGS`，供后续按键查询。
BACKTEST_MODE_CONFIGS = {
    "mode1_t_with_t1_filter": {
        "price_start_offset": 0,
        "price_start": "T_no_shift",
        "portfolio_scope": "preassigned_quantile_then_can_enter_t1",
        "execution_mask": "can_enter_t1",
    },
    "mode2_t_no_filter": {
        "price_start_offset": 0,
        "price_start": "T_no_shift",
        "portfolio_scope": "full_t_cross_section_no_t1_filter",
        "execution_mask": "none",
    },
    "mode3_t1_no_filter": {
        "price_start_offset": 1,
        "price_start": "T_plus_1",
        "portfolio_scope": "full_t_cross_section_no_t1_filter",
        "execution_mask": "none",
    },
}

# 检查 `str(VIT_DIR) not in sys.path`；根据条件结果选择对应处理分支。
if str(VIT_DIR) not in sys.path:
    # 调用 `sys.path.insert` 把项目模块目录加入导入搜索路径。
    sys.path.insert(0, str(VIT_DIR))

from vit_stage01_foundation import CALENDAR_PATH, atomic_json, normalize_dates  # noqa: E402


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


# 定义 `validate_prediction_column` 函数，封装这一阶段可独立调用的处理步骤。
def validate_prediction_column(value: str) -> str:
    """校验待评价预测列是否属于允许的信号口径。
    
    输入：
        value: str。
    输出：
        str。
        """
    # 调用 `str(value).strip` 去除字符串首尾空白以统一配置值，并将结果保存到 `prediction_column`。
    prediction_column = str(value).strip()
    # 检查 `prediction_column not in PREDICTION_COLUMNS`；根据条件结果选择对应处理分支。
    if prediction_column not in PREDICTION_COLUMNS:
        # 抛出 `ValueError(f'未知预测列：{value!r}；可选 {PREDICTION_COLUMNS}')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"未知预测列：{value!r}；可选 {PREDICTION_COLUMNS}"
        )
    # 将 `prediction_column` 作为当前函数结果返回给调用方。
    return prediction_column


# 定义 `evaluation_directory_name` 函数，封装这一阶段可独立调用的处理步骤。
def evaluation_directory_name(prediction_column: str = FACTOR_COLUMN) -> str:
    """根据预测列生成互不覆盖的评价目录名。
    
    输入：
        prediction_column: str。
    输出：
        str。
        """
    # 将 `EVALUATION_DIRECTORY_NAMES[validate_prediction_column(prediction_column)]` 作为当前函数结果返回给调用方。
    return EVALUATION_DIRECTORY_NAMES[
        validate_prediction_column(prediction_column)
    ]


# 定义 `experiment_report_name` 函数，封装这一阶段可独立调用的处理步骤。
def experiment_report_name(prediction_column: str = FACTOR_COLUMN) -> str:
    """根据预测列生成对应实验报告文件名。
    
    输入：
        prediction_column: str。
    输出：
        str。
        """
    # 将 `EXPERIMENT_REPORT_NAMES[validate_prediction_column(prediction_column)]` 作为当前函数结果返回给调用方。
    return EXPERIMENT_REPORT_NAMES[
        validate_prediction_column(prediction_column)
    ]


# 定义 `resolve_backtest_mode` 函数，封装这一阶段可独立调用的处理步骤。
def resolve_backtest_mode(backtest_mode: str) -> dict[str, object]:
    """解析并校验统一样本外回测模式。
    
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


# 定义 `read_json` 函数，封装这一阶段可独立调用的处理步骤。
def read_json(path: Path) -> dict[str, object]:
    """读取 JSON 对象；目标不存在时按接口约定返回空结果。
    
    输入：
        path: Path。
    输出：
        dict[str, object]。
        """
    # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
    if not path.exists():
        # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(path)
    # 将 `json.loads(path.read_text(encoding='utf-8'))` 作为当前函数结果返回给调用方。
    return json.loads(path.read_text(encoding="utf-8"))


# 定义内部辅助函数 `_roots`，封装仅供本模块复用的计算步骤。
def _roots(
    vivit_data_root: str | Path | None,
    reference_data_root: str | Path | None,
) -> tuple[Path, Path]:
    """解析当前流程的数据、模型或质量目录。
    
    输入：
        vivit_data_root: str | Path | None。
        reference_data_root: str | Path | None。
    输出：
        tuple[Path, Path]。
        """
    # 调用 `Path(vivit_data_root or DEFAULT_VIVIT_DATA_ROOT).resolve` 规范化文件路径，并将结果保存到 `vivit_root`。
    vivit_root = Path(vivit_data_root or DEFAULT_VIVIT_DATA_ROOT).resolve()
    # 调用 `Path(reference_data_root or DEFAULT_REFERENCE_DATA_ROOT).resolve` 规范化文件路径，并将结果保存到 `reference_root`。
    reference_root = Path(
        reference_data_root or DEFAULT_REFERENCE_DATA_ROOT
    ).resolve()
    # 检查 `not vivit_root.is_dir()`；根据条件结果选择对应处理分支。
    if not vivit_root.is_dir():
        # 抛出 `FileNotFoundError(vivit_root)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(vivit_root)
    # 检查 `not reference_root.is_dir()`；根据条件结果选择对应处理分支。
    if not reference_root.is_dir():
        # 抛出 `FileNotFoundError(reference_root)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(reference_root)
    # 将 `(vivit_root, reference_root)` 作为当前函数结果返回给调用方。
    return vivit_root, reference_root


# 定义内部辅助函数 `_manifests`，封装仅供本模块复用的计算步骤。
def _manifests(
    run_name: str, vivit_root: Path, reference_root: Path
) -> dict[str, dict[str, object]]:
    """读取并校验上游与当前产物的 manifest 签名链。
    
    输入：
        run_name: str。
        vivit_root: Path。
        reference_root: Path。
    输出：
        dict[str, dict[str, object]]。
        """
    # 计算 `vivit_root / 'quality' / 'runs' / run_name`，并将数值结果保存到 `quality_run`。
    quality_run = vivit_root / "quality" / "runs" / run_name
    # 构造键值映射，并保存到 `result`，供后续按键查询。
    result = {
        "training": read_json(quality_run / "a2_training_manifest.json"),
        "sequence": read_json(
            vivit_root / "aligned_sequence_store" / "sequence_manifest.json"
        ),
        "dataset": read_json(
            reference_root / "feature_store" / "dataset_manifest.json"
        ),
        "rolling": read_json(
            reference_root / "feature_store" / "rolling_split_manifest.json"
        ),
        "universe": read_json(
            reference_root / "universe" / "universe_manifest.json"
        ),
        "labels": read_json(
            reference_root / "labels_t10" / "labels_manifest.json"
        ),
    }
    # 读取 `result['training']` 的当前值，并赋给 `training`。
    training = result["training"]
    # 读取 `result['sequence']` 的当前值，并赋给 `sequence`。
    sequence = result["sequence"]
    # 读取 `result['dataset']` 的当前值，并赋给 `dataset`。
    dataset = result["dataset"]
    # 读取 `result['rolling']` 的当前值，并赋给 `rolling`。
    rolling = result["rolling"]
    # 读取 `result['universe']` 的当前值，并赋给 `universe`。
    universe = result["universe"]
    # 调用 `sequence.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `sequence_payload`。
    sequence_payload = sequence.get("payload", {})
    # 调用 `dataset.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `dataset_payload`。
    dataset_payload = dataset.get("payload", {})
    # 构造键值映射，并保存到 `expected`，供后续按键查询。
    expected = {
        "training source dataset": (
            training.get("source_dataset_signature"), dataset.get("signature")
        ),
        "training source rolling": (
            training.get("source_rolling_signature"), rolling.get("signature")
        ),
        "training aligned sequence": (
            training.get("sequence_signature"), sequence.get("signature")
        ),
        "sequence source dataset": (
            sequence_payload.get("source_dataset_signature"), dataset.get("signature")
        ),
        "sequence source rolling": (
            sequence_payload.get("source_rolling_signature"), rolling.get("signature")
        ),
        "dataset Universe": (
            dataset_payload.get("universe_signature"), universe.get("signature")
        ),
    }
    # 构造当前步骤所需的列表，并保存到 `mismatches`。
    mismatches = [
        f"{name}: {actual} != {wanted}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    # 检查 `mismatches`；根据条件结果选择对应处理分支。
    if mismatches:
        # 抛出 `AssertionError('签名链不一致：' + '; '.join(mismatches))`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("签名链不一致：" + "; ".join(mismatches))
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 定义内部辅助函数 `_prediction_counts`，封装仅供本模块复用的计算步骤。
def _prediction_counts(training_manifest: dict[str, object]) -> dict[int, int]:
    """处理预测文件、预测列或对应的样本数量审计。
    
    输入：
        training_manifest: dict[str, object]。
    输出：
        dict[int, int]。
        """
    # 调用 `tuple` 整理数据结构，并将结果保存到 `years`。
    years = tuple(int(year) for year in training_manifest.get("years", ()))
    # 检查 `not years`；根据条件结果选择对应处理分支。
    if not years:
        # 抛出 `ValueError('训练 manifest 缺少 years')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练 manifest 缺少 years")
    # 调用 `training_manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `raw_counts`。
    raw_counts = training_manifest.get("factor_counts")
    # 检查 `not isinstance(raw_counts, dict)`；根据条件结果选择对应处理分支。
    if not isinstance(raw_counts, dict):
        # 抛出 `ValueError('训练 manifest 缺少 factor_counts')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练 manifest 缺少 factor_counts")
    # 构造键值映射，并保存到 `counts`，供后续按键查询。
    counts = {
        year: int(raw_counts.get(str(year), raw_counts.get(year, -1)))
        for year in years
    }
    # 检查 `any((value < 0 for value in counts.values()))`；根据条件结果选择对应处理分支。
    if any(value < 0 for value in counts.values()):
        # 抛出 `ValueError('训练 manifest 的 factor_counts 不完整')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("训练 manifest 的 factor_counts 不完整")
    # 将 `counts` 作为当前函数结果返回给调用方。
    return counts


# 定义内部辅助函数 `_calendar_last_date`，封装仅供本模块复用的计算步骤。
def _calendar_last_date(
    last_factor_date: str, backtest_mode: str = DEFAULT_BACKTEST_MODE
) -> str:
    """返回指定交易所日历中不晚于给定日期的最后交易日。
    
    输入：
        last_factor_date: str。
        backtest_mode: str。
    输出：
        str。
        """
    # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `calendar`。
    calendar = pd.read_parquet(
        CALENDAR_PATH,
        columns=["trade_days", "s_info_exchmarket"],
        filters=[("s_info_exchmarket", "==", "SSE")],
    )
    # 调用 `normalize_dates(calendar['trade_days']).drop_duplicates().sort_…` 根据分数计算截面排序或排序索引，并将结果保存到 `dates`。
    dates = (
        normalize_dates(calendar["trade_days"])
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype=str)
    )
    # 调用 `int` 统一设备或数据类型，并将结果保存到 `position`。
    position = int(np.searchsorted(dates, last_factor_date))
    # 检查 `position >= len(dates) or dates[position] != last_factor_date`；根据条件结果选择对应处理分支。
    if position >= len(dates) or dates[position] != last_factor_date:
        # 抛出 `AssertionError(f'{last_factor_date} 不在 SSE 交易日历')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError(f"{last_factor_date} 不在 SSE 交易日历")
    # 调用 `resolve_backtest_mode` 解析用户配置并转换为内部统一口径，并将结果保存到 `mode`。
    mode = resolve_backtest_mode(backtest_mode)
    # 计算 `int(mode['price_start_offset']) + max(PERIODS)`，并将数值结果保存到 `final_offset`。
    final_offset = int(mode["price_start_offset"]) + max(PERIODS)
    # 检查 `position + final_offset >= len(dates)`；根据条件结果选择对应处理分支。
    if position + final_offset >= len(dates):
        # 抛出 `ValueError(f'交易日历不足以覆盖最大 {final_offset} 个交易日收益期限')`，阻止不满足数据契约的流程继续执行。
        raise ValueError(
            f"交易日历不足以覆盖最大 {final_offset} 个交易日收益期限"
        )
    # 将 `str(dates[position + final_offset])` 作为当前函数结果返回给调用方。
    return str(dates[position + final_offset])




# 定义 `assign_factor_quantiles` 函数，封装这一阶段可独立调用的处理步骤。
def assign_factor_quantiles(
    factor: pd.Series, quantiles: int = QUANTILES
) -> pd.Series:
    """使用稳定排序在每个完整 T 日截面分配近似等量分位组。
    
    输入：
        factor: pd.Series。
        quantiles: int。
    输出：
        pd.Series。
        """

    # 检查 `quantiles < 2`；根据条件结果选择对应处理分支。
    if quantiles < 2:
        # 抛出 `ValueError('quantiles 必须至少为 2')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("quantiles 必须至少为 2")
    # 检查 `not isinstance(factor.index, pd.MultiIndex) or 'date' not in factor.index.names`；根据条件结果选择对应处理分支。
    if not isinstance(factor.index, pd.MultiIndex) or "date" not in factor.index.names:
        # 抛出 `ValueError('factor 必须使用含 date level 的 MultiIndex')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("factor 必须使用含 date level 的 MultiIndex")
    # 调用 `factor.groupby` 按指定字段建立分组对象，并将结果保存到 `groups`。
    groups = factor.groupby(level="date", sort=False)
    # 调用 `groups.rank` 根据分数计算截面排序或排序索引，并将结果保存到 `ordinal_rank`。
    ordinal_rank = groups.rank(method="first", ascending=True)
    # 调用 `groups.transform` 把分组统计结果广播回每一条原始记录，并将结果保存到 `group_size`。
    group_size = groups.transform("size")
    # 检查 `(group_size < quantiles).any()`；根据条件结果选择对应处理分支。
    if (group_size < quantiles).any():
        # 抛出 `ValueError('存在样本数少于分位组数的日期')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("存在样本数少于分位组数的日期")
    # 调用 `pd.Series` 转换数据表示，并将标准化对象保存到 `result`。
    result = pd.Series(
        np.floor(
            (ordinal_rank.to_numpy(np.float64) - 1)
            * quantiles
            / group_size.to_numpy(np.float64)
        ).astype(np.int64) + 1,
        index=factor.index,
        name="factor_quantile",
    )
    # 调用 `unstack` 把分组索引展开为列以形成日期×分组矩阵，并将结果保存到 `counts`。
    counts = result.groupby(
        [pd.Grouper(level="date"), result], observed=True
    ).size().unstack("factor_quantile")
    # 检查 `counts.isna().any(axis=None) or (counts.max(axis=1) - counts.min(axis=1)).gt(1).any()`；根据条件结果选择对应处理分支。
    if counts.isna().any(axis=None) or (
        counts.max(axis=1) - counts.min(axis=1)
    ).gt(1).any():
        # 抛出 `AssertionError('T 日完整截面分位组不均衡')`，阻止不满足数据契约的流程继续执行。
        raise AssertionError("T 日完整截面分位组不均衡")
    # 将 `result` 作为当前函数结果返回给调用方。
    return result


# 定义内部辅助函数 `_normalized_vit_manifest`，封装仅供本模块复用的计算步骤。
def _normalized_vit_manifest(
    manifests: dict[str, dict[str, object]],
) -> dict[str, object]:
    """读取并校验上游与当前产物的 manifest 签名链。
    
    输入：
        manifests: dict[str, dict[str, object]]。
    输出：
        dict[str, object]。
        """
    # 调用 `dict` 整理数据结构，并将结果保存到 `training`。
    training = dict(manifests["training"])
    # 读取 `manifests['dataset']['signature']` 的当前值，并赋给 `training['dataset_signature']`。
    training["dataset_signature"] = manifests["dataset"]["signature"]
    # 读取 `manifests['universe']['signature']` 的当前值，并赋给 `training['universe_signature']`。
    training["universe_signature"] = manifests["universe"]["signature"]
    # 将 `training` 作为当前函数结果返回给调用方。
    return training


# 定义内部辅助函数 `_create_vit_evaluation_view`，封装仅供本模块复用的计算步骤。
def _create_vit_evaluation_view(
    temporary_root: Path,
    run_name: str,
    vivit_root: Path,
    reference_root: Path,
    manifests: dict[str, dict[str, object]],
    prediction_column: str = FACTOR_COLUMN,
) -> Path:
    """在临时目录建立 ViT evaluator 所需的只读文件名视图。
    
    输入：
        temporary_root: Path。
        run_name: str。
        vivit_root: Path。
        reference_root: Path。
        manifests: dict[str, dict[str, object]]。
        prediction_column: str。
    输出：
        Path。
        """

    # 调用 `validate_prediction_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `prediction_column`。
    prediction_column = validate_prediction_column(prediction_column)
    # 遍历 `('feature_store', 'universe', 'labels_t10')`，并把当前元素绑定到 `name` 逐项处理。
    for name in ("feature_store", "universe", "labels_t10"):
        # 调用 `(temporary_root / name).symlink_to` 创建指向参考数据目录的符号链接。
        (temporary_root / name).symlink_to(
            reference_root / name, target_is_directory=True
        )
    # 计算 `vivit_root / 'predictions' / 'runs' / run_name`，并将数值结果保存到 `source_predictions`。
    source_predictions = vivit_root / "predictions" / "runs" / run_name
    # 计算 `temporary_root / 'predictions' / 'runs' / run_name`，并将数值结果保存到 `view_predictions`。
    view_predictions = temporary_root / "predictions" / "runs" / run_name
    # 调用 `view_predictions.mkdir` 创建输出目录，确保后续文件可以安全写入。
    view_predictions.mkdir(parents=True)
    # 调用 `_prediction_counts` 执行模型推理并生成样本预测分数，并将结果保存到 `counts`。
    counts = _prediction_counts(manifests["training"])
    # 遍历 `counts.items()`，并把当前元素绑定到 `(year, count)` 逐项处理。
    for year, count in counts.items():
        # 检查 `count`；根据条件结果选择对应处理分支。
        if count:
            # 计算 `source_predictions / PREDICTION_PATTERN.format(year=year)`，并将数值结果保存到 `source`。
            source = source_predictions / PREDICTION_PATTERN.format(year=year)
            # 计算 `view_predictions / f'vit_predictions.{year}.parquet'`，并将数值结果保存到 `target`。
            target = view_predictions / f"vit_predictions.{year}.parquet"
            # 检查 `prediction_column == FACTOR_COLUMN`；根据条件结果选择对应处理分支。
            if prediction_column == FACTOR_COLUMN:
                # 调用 `target.symlink_to` 创建指向参考数据目录的符号链接。
                target.symlink_to(source)
            # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
            else:
                # 调用 `pd.read_parquet(source, columns=['trade_date', 'instrument_id',…` 读取源数据，并将结果保存到 `prediction`。
                prediction = pd.read_parquet(
                    source,
                    columns=["trade_date", "instrument_id", prediction_column],
                ).rename(columns={prediction_column: FACTOR_COLUMN})
                # 调用 `prediction.to_parquet` 将当前结果持久化到目标文件。
                prediction.to_parquet(target, index=False)
                # 删除 `prediction`，释放不再使用的引用。
                del prediction

    # 计算 `temporary_root / 'quality' / 'runs' / run_name`，并将数值结果保存到 `view_quality_run`。
    view_quality_run = temporary_root / "quality" / "runs" / run_name
    # 调用 `view_quality_run.mkdir` 创建输出目录，确保后续文件可以安全写入。
    view_quality_run.mkdir(parents=True)
    # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_json(
        _normalized_vit_manifest(manifests),
        view_quality_run / "a2_training_manifest.json",
    )
    # 计算 `vivit_root / 'quality' / 'runs' / run_name / evaluation_directory_name(prediction_column)`，并将数值结果保存到 `evaluation_target`。
    evaluation_target = vivit_root / "quality" / "runs" / run_name / (
        evaluation_directory_name(prediction_column)
    )
    # 调用 `evaluation_target.mkdir` 创建输出目录，确保后续文件可以安全写入。
    evaluation_target.mkdir(parents=True, exist_ok=True)
    # 调用 `(view_quality_run / 'factor_evaluation').symlink_to` 在不更新参数的条件下计算验证或样本外评价指标。
    (view_quality_run / "factor_evaluation").symlink_to(
        evaluation_target, target_is_directory=True
    )
    # 将 `evaluation_target` 作为当前函数结果返回给调用方。
    return evaluation_target


# 定义内部辅助函数 `_markdown_table`，封装仅供本模块复用的计算步骤。
def _markdown_table(frame: pd.DataFrame) -> str:
    """把结构化结果转换为 Markdown 表格。
    
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
    # 构造当前步骤所需的列表，并保存到 `lines`。
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    # 遍历 `clean.astype(str).itertuples(index=False, name=None)`，并把当前元素绑定到 `row` 逐项处理。
    for row in clean.astype(str).itertuples(index=False, name=None):
        # 把本轮结果追加到 `lines`，累积后续需要汇总的记录。
        lines.append(
            "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        )
    # 将 `'\n'.join(lines)` 作为当前函数结果返回给调用方。
    return "\n".join(lines)


# 定义内部辅助函数 `_read_optional_parquet`，封装仅供本模块复用的计算步骤。
def _read_optional_parquet(path: Path) -> pd.DataFrame:
    """读取对应产物，并按接口约定处理缺失文件。
    
    输入：
        path: Path。
    输出：
        pd.DataFrame。
        """
    # 将 `pd.read_parquet(path) if path.exists() else pd.DataFrame()` 作为当前函数结果返回给调用方。
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


# 定义 `generate_experiment_report` 函数，封装这一阶段可独立调用的处理步骤。
def generate_experiment_report(
    run_name: str,
    vivit_data_root: str | Path | None = None,
    reference_data_root: str | Path | None = None,
    prediction_column: str = FACTOR_COLUMN,
) -> Path:
    """生成同时包含 ViViT 训练与因子评价结果的 run 专属报告。
    
    输入：
        run_name: str。
        vivit_data_root: str | Path | None。
        reference_data_root: str | Path | None。
        prediction_column: str。
    输出：
        Path。
        """

    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `run_name`。
    run_name = validate_run_name(run_name)
    # 调用 `validate_prediction_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `prediction_column`。
    prediction_column = validate_prediction_column(prediction_column)
    # 调用 `evaluation_directory_name` 在不更新参数的条件下计算验证或样本外评价指标，并将结果保存到 `evaluation_name`。
    evaluation_name = evaluation_directory_name(prediction_column)
    # 调用 `_roots` 解析本次运行对应的数据和输出目录，并将结果保存到 `(vivit_root, reference_root)`。
    vivit_root, reference_root = _roots(vivit_data_root, reference_data_root)
    # 调用 `_manifests` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `manifests`。
    manifests = _manifests(run_name, vivit_root, reference_root)
    # 计算 `vivit_root / 'quality' / 'runs' / run_name`，并将数值结果保存到 `quality_run`。
    quality_run = vivit_root / "quality" / "runs" / run_name
    # 计算 `quality_run / evaluation_name`，并将数值结果保存到 `evaluation_dir`。
    evaluation_dir = quality_run / evaluation_name
    # 调用 `_read_optional_parquet` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `training`。
    training = _read_optional_parquet(
        quality_run / "vivit_training_summary.parquet"
    )
    # 检查 `training.empty`；根据条件结果选择对应处理分支。
    if training.empty:
        # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `training_yearly`。
        training_yearly = pd.DataFrame()
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 构造键值映射，并保存到 `aggregations`，供后续按键查询。
        aggregations: dict[str, tuple[str, str]] = {
            "completed_seeds": ("seed", "nunique"),
            "mean_best_validation_ic": ("best_validation_ic", "mean"),
            "min_best_validation_ic": ("best_validation_ic", "min"),
            "mean_best_epoch": ("best_epoch", "mean"),
        }
        # 遍历 `(('mean_best_validation_rank_ic', 'best_validation_rank_ic'), ('mean_best_validation_daily_pear…`，并把当前元素绑定到 `(output, source)` 逐项处理。
        for output, source in (
            ("mean_best_validation_rank_ic", "best_validation_rank_ic"),
            ("mean_best_validation_daily_pearson_ic", "best_validation_daily_pearson_ic"),
            ("mean_best_validation_pooled_ic", "best_validation_pooled_ic"),
        ):
            # 检查 `source in training.columns`；根据条件结果选择对应处理分支。
            if source in training.columns:
                # 组织一组有序值，并保存到 `aggregations[output]`。
                aggregations[output] = (source, "mean")
        # 调用 `training.groupby('forecast_year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `training_yearly`。
        training_yearly = training.groupby(
            "forecast_year", as_index=False
        ).agg(**aggregations)

    # 调用 `_read_optional_parquet` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `sequence_summary`。
    sequence_summary = _read_optional_parquet(
        vivit_root / "aligned_sequence_store" / "rolling_split_summary.parquet"
    )
    # 调用 `_read_optional_parquet` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `final_audit`。
    final_audit = _read_optional_parquet(evaluation_dir / "final_audit.parquet")
    # 调用 `_read_optional_parquet` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `ic_summary`。
    ic_summary = _read_optional_parquet(evaluation_dir / "ic_summary.parquet")
    # 调用 `_read_optional_parquet` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `spread_summary`。
    spread_summary = _read_optional_parquet(evaluation_dir / "spread_summary.parquet")
    # 调用 `_read_optional_parquet` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `pyfolio`。
    pyfolio = _read_optional_parquet(evaluation_dir / "pyfolio_performance.parquet")
    # 调用 `_read_optional_parquet` 读取并解析后续计算所需的数据或运行状态，并将结果保存到 `relative_performance`。
    relative_performance = _read_optional_parquet(
        evaluation_dir / "relative_performance.parquet"
    )
    # 计算 `evaluation_dir / 'factor_evaluation_manifest.json'`，并将数值结果保存到 `evaluation_manifest_path`。
    evaluation_manifest_path = evaluation_dir / "factor_evaluation_manifest.json"
    # 按 `evaluation_manifest_path.exists()` 在两个候选值间选择，并保存到 `evaluation_manifest`。
    evaluation_manifest = (
        read_json(evaluation_manifest_path)
        if evaluation_manifest_path.exists()
        else None
    )
    # 检查 `evaluation_manifest`；根据条件结果选择对应处理分支。
    if evaluation_manifest:
        # 调用 `int` 统一设备或数据类型，并将结果保存到 `start_offset`。
        start_offset = int(evaluation_manifest.get("price_start_offset", 0))
        # 按 `start_offset == 0` 在两个候选值间选择，并保存到 `period_description`。
        period_description = (
            "1/5/10D 对应 T→T+1/T+5/T+10"
            if start_offset == 0
            else "1/5/10D 对应 T+1→T+2/T+6/T+11"
        )
        # 按 `evaluation_manifest.get('execution_mask') == 'can_enter_t1'` 在两个候选值间选择，并保存到 `execution_description`。
        execution_description = (
            "组合收益和换手应用 `can_enter_t1`"
            if evaluation_manifest.get("execution_mask") == "can_enter_t1"
            else "组合收益和换手不应用 `can_enter_t1` 未来状态过滤"
        )
        # 构造当前步骤所需的列表，并保存到 `evaluation_policy_lines`。
        evaluation_policy_lines = [
            f"- 回测模式：`{evaluation_manifest.get('backtest_mode')}`；{execution_description}。",
            f"- 收益公式：`{evaluation_manifest.get('forward_return_formula')}`；{period_description}。",
            "- 全市场等权基准使用完整因子覆盖池和相同收益时点，不应用 `can_enter_t1`。",
            "- Q5 与 Q5−Q1 均报告相对市场的超额年化、跟踪误差和信息比。",
        ]
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 构造当前步骤所需的列表，并保存到 `evaluation_policy_lines`。
        evaluation_policy_lines = ["- 因子评价尚未按新版回测口径生成。"]
    # 读取 `manifests['training']` 的当前值，并赋给 `training_manifest`。
    training_manifest = manifests["training"]
    # 按 `training_manifest.get('train_batch_mode') == 'trade_date'` 在两个候选值间选择，并保存到 `missing_batch_value`。
    missing_batch_value = (
        "full_trade_date"
        if training_manifest.get("train_batch_mode") == "trade_date"
        else "not_recorded"
    )
    # 调用 `training_manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `fixed_batch_size`。
    fixed_batch_size = training_manifest.get(
        "fixed_batch_size", missing_batch_value
    )
    # 调用 `training_manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `inference_batch_size`。
    inference_batch_size = training_manifest.get(
        "inference_batch_size", missing_batch_value
    )
    # 调用 `training_manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `derivation`。
    derivation = training_manifest.get("factor_derivation")
    # 检查 `isinstance(derivation, dict)`；根据条件结果选择对应处理分支。
    if isinstance(derivation, dict):
        # 构造当前步骤所需的列表，并保存到 `derivation_lines`。
        derivation_lines = [
            "## Barra 残差因子派生",
            "",
            f"- 源 run：`{training_manifest.get('source_run_name')}`",
            f"- 因子角色：`{training_manifest.get('factor_role')}`",
            f"- 派生签名：`{training_manifest.get('factor_derivation_signature')}`",
            f"- CNE5S 风格暴露：`{', '.join(derivation.get('style_columns', []))}`",
            f"- 回归连续变量：`{', '.join(derivation.get('regression_columns', derivation.get('style_columns', [])))}`",
            f"- 20 日反转：`{derivation.get('reversal', {}).get('formula')}`；缩放：`{derivation.get('reversal', {}).get('regression_scaling')}`。",
            f"- 行业字段：`{derivation.get('industry_column')}`；不使用 `cne5s_resvol`。",
            f"- 匹配规则：{derivation.get('match_policy')}",
            f"- 缺失处理：CNE5S=`{derivation.get('missing_barra_policy')}`；反转=`{derivation.get('missing_reversal_policy')}`。",
            f"- 残差尺度：{derivation.get('residual_scaling')}",
            "- 相关性、回归诊断和 lineage 位于 `barra_neutralization/`。",
            "",
        ]
    # 当前置条件不成立时进入备用分支，保证其余合法情况仍被处理。
    else:
        # 构造当前步骤所需的列表，并保存到 `derivation_lines`。
        derivation_lines = []
    # 构造当前步骤所需的列表，并保存到 `lines`。
    lines = [
        f"# ViViT 对齐训练与样本外因子评价｜{run_name}",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')} UTC",
        "",
        "## 数据与模型口径",
        "",
        "- train、validation、prediction 直接使用 ViT 年度索引，不做每 5 日降采样或完整历史过滤。",
        "- 20 个历史槽位只匹配同股票精确日期；缺失帧补零，并以 frame mask 和 0/1/2 Tubelet 覆盖状态入模。",
        "- T 日完整预测截面先固定五分位；RankIC 始终使用完整价格有效截面。",
        *evaluation_policy_lines,
        f"- 本报告评价预测列：`{prediction_column}`。",
        "",
        "## 对齐历史覆盖",
        "",
        _markdown_table(sequence_summary),
        "",
        "## 训练配置",
        "",
        f"- loss / batch mode / validation metric：`{training_manifest['loss_name']}` / `{training_manifest['train_batch_mode']}` / `{training_manifest['validation_metric']}`",
        f"- fixed / inference batch：{fixed_batch_size} / {inference_batch_size}",
        f"- compile / mode / fused Adam / CUDA prefetch：{training_manifest['torch_compile']} / `{training_manifest.get('compile_mode', 'default')}` / {training_manifest['fused_adam']} / {training_manifest['cuda_prefetch']}",
        "",
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
        "## 签名与产物",
        "",
        f"- Universe：`{manifests['universe']['signature']}`",
        f"- feature store：`{manifests['dataset']['signature']}`",
        f"- aligned sequence store：`{manifests['sequence']['signature']}`",
        f"- 累计收益图：`{evaluation_name}/portfolio_cumulative_returns.png`。",
        f"- 市场基准与相对绩效：`{evaluation_name}/portfolio_returns.parquet`、`relative_performance.parquet`。",
        f"- 最大回撤及 Alphalens returns/information/turnover tear sheets：`{evaluation_name}/pic/`。",
        "- 2025 prediction 与 ViT 一致为空，不生成或评价 2025 因子。",
        "- ViT 数据与评价产物保持只读。",
        "",
    ]
    # 检查 `derivation_lines`；根据条件结果选择对应处理分支。
    if derivation_lines:
        # 调用 `lines.index` 构造带名称的一维 Pandas 索引，并将结果保存到 `insert_at`。
        insert_at = lines.index("## 对齐历史覆盖")
        # 读取 `derivation_lines` 的当前值，并赋给 `lines[insert_at:insert_at]`。
        lines[insert_at:insert_at] = derivation_lines
    # 计算 `quality_run / experiment_report_name(prediction_column)`，并将数值结果保存到 `path`。
    path = quality_run / experiment_report_name(prediction_column)
    # 调用 `path.parent.mkdir` 创建输出目录，确保后续文件可以安全写入。
    path.parent.mkdir(parents=True, exist_ok=True)
    # 调用 `path.with_name` 规范化文件路径，并将结果保存到 `temporary`。
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    # 调用 `temporary.write_text` 将当前结果持久化到目标文件。
    temporary.write_text("\n".join(lines), encoding="utf-8")
    # 调用 `temporary.replace` 用完整临时文件原子替换正式产物。
    temporary.replace(path)
    # 调用 `print` 记录当前流程状态或诊断信息。
    print(f"实验报告：{path}")
    # 将 `path` 作为当前函数结果返回给调用方。
    return path


# 定义 `run_oos_backtest` 函数，封装这一阶段可独立调用的处理步骤。
def run_oos_backtest(
    run_name: str = DEFAULT_RUN_NAME,
    vivit_data_root: str | Path | None = None,
    reference_data_root: str | Path | None = None,
    backtest_mode: str = DEFAULT_BACKTEST_MODE,
    prediction_column: str = FACTOR_COLUMN,
) -> pd.DataFrame:
    """按指定共享口径执行 ViViT Alphalens/Pyfolio 评价。
    
    输入：
        run_name: str。
        vivit_data_root: str | Path | None。
        reference_data_root: str | Path | None。
        backtest_mode: str。
        prediction_column: str。
    输出：
        pd.DataFrame。
        """

    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `run_name`。
    run_name = validate_run_name(run_name)
    # 调用 `validate_prediction_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `prediction_column`。
    prediction_column = validate_prediction_column(prediction_column)
    # 调用 `resolve_backtest_mode` 解析用户配置并转换为内部统一口径，并将结果保存到 `mode`。
    mode = resolve_backtest_mode(backtest_mode)
    # 调用 `_roots` 解析本次运行对应的数据和输出目录，并将结果保存到 `(vivit_root, reference_root)`。
    vivit_root, reference_root = _roots(vivit_data_root, reference_data_root)
    # 调用 `audit_evaluation_inputs` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `input_audit`。
    input_audit = audit_evaluation_inputs(
        run_name, vivit_root, reference_root, str(mode["name"]),
        prediction_column,
    )
    # 调用 `_manifests` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `manifests`。
    manifests = _manifests(run_name, vivit_root, reference_root)
    # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
    try:
        import vit_stage08_factor_evaluation as vit_evaluation
    # 捕获 `except ModuleNotFoundError as error` 对应的异常，并执行明确的降级、记录或清理逻辑。
    except ModuleNotFoundError as error:
        # 抛出 `RuntimeError('请使用 Jupyter 内核 `ViT Factor Eval (Alphalens/Pyfolio)`。')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError(
            "请使用 Jupyter 内核 `ViT Factor Eval (Alphalens/Pyfolio)`。"
        ) from error

    # 在 `tempfile.TemporaryDirectory(prefix=f'vivit_factor_eva…` 管理的上下文中执行本段逻辑，并保证资源按约定释放。
    with tempfile.TemporaryDirectory(
        prefix=f"vivit_factor_eval_{run_name}_"
    ) as directory:
        # 调用 `Path` 规范化文件路径，并将结果保存到 `temporary_root`。
        temporary_root = Path(directory)
        # 调用 `_create_vit_evaluation_view` 根据当前配置和输入构造后续流程所需对象，并将结果保存到 `evaluation_dir`。
        evaluation_dir = _create_vit_evaluation_view(
            temporary_root, run_name, vivit_root, reference_root, manifests,
            prediction_column,
        )
        # 读取 `vit_evaluation.generate_experiment_report` 的当前值，并赋给 `original_report`。
        original_report = vit_evaluation.generate_experiment_report

        # 定义 `vivit_report_adapter` 函数，封装这一阶段可独立调用的处理步骤。
        def vivit_report_adapter(**_kwargs: object) -> Path:
            """把 ViT 报告回调重定向到当前 ViViT run 和预测列。
            
            输入：
                _kwargs: object，ViT 回调传入但本适配器不使用的参数。
            输出：
                Path，当前 ViViT run 的实验报告路径。
                """
            # 将 `generate_experiment_report(run_name, vivit_root, reference_root, prediction_column)` 作为当前函数结果返回给调用方。
            return generate_experiment_report(
                run_name, vivit_root, reference_root, prediction_column
            )

        # 读取 `vivit_report_adapter` 的当前值，并赋给 `vit_evaluation.generate_experiment_report`。
        vit_evaluation.generate_experiment_report = vivit_report_adapter
        # 尝试执行本段主逻辑；若发生异常，则交由后续异常分支处理。
        try:
            # 调用 `vit_evaluation.run_oos_backtest` 在不更新参数的条件下计算验证或样本外评价指标，并将结果保存到 `result`。
            result = vit_evaluation.run_oos_backtest(
                run_name=run_name,
                data_root=temporary_root,
                backtest_mode=str(mode["name"]),
            )
        # 无论前序步骤是否抛出异常，都执行本分支完成必要的收尾。
        finally:
            # 读取 `original_report` 的当前值，并赋给 `vit_evaluation.generate_experiment_report`。
            vit_evaluation.generate_experiment_report = original_report

    # 计算 `evaluation_dir / 'factor_evaluation_manifest.json'`，并将数值结果保存到 `manifest_path`。
    manifest_path = evaluation_dir / "factor_evaluation_manifest.json"
    # 调用 `read_json` 读取源数据，并将结果保存到 `evaluation_manifest`。
    evaluation_manifest = read_json(manifest_path)
    # 读取 `manifests['training']` 的当前值，并赋给 `training_manifest`。
    training_manifest = manifests["training"]
    # 用本轮键值结果更新 `evaluation_manifest`，补齐后续查询所需字段。
    evaluation_manifest.update(
        {
            "model_family": "ViViT",
            "factor_role": training_manifest.get(
                "factor_role", "model_prediction"
            ),
            "source_run_name": training_manifest.get("source_run_name"),
            "factor_derivation_signature": training_manifest.get(
                "factor_derivation_signature"
            ),
            "prediction_pattern": PREDICTION_PATTERN,
            "prediction_column": prediction_column,
            "dataset_signature": manifests["dataset"]["signature"],
            "universe_signature": manifests["universe"]["signature"],
            "sequence_signature": manifests["sequence"]["signature"],
            "input_audit_rows": int(input_audit["prediction_rows"].sum()),
            "quantiles": QUANTILES,
            "max_loss": MAX_LOSS,
        }
    )
    # 调用 `atomic_json` 先写临时文件再原子替换正式产物，避免留下半成品。
    atomic_json(evaluation_manifest, manifest_path)
    # 调用 `generate_experiment_report` 根据当前结果生成可审计的报告或展示产物。
    generate_experiment_report(
        run_name, vivit_root, reference_root, prediction_column
    )
    # 将 `result` 作为当前函数结果返回给调用方。
    return result




# 定义 `compare_factor_evaluations` 函数，封装这一阶段可独立调用的处理步骤。
def compare_factor_evaluations(
    run_name: str,
    baseline_run_name: str | None,
    vivit_data_root: str | Path | None = None,
    prediction_column: str = FACTOR_COLUMN,
    baseline_prediction_column: str = FACTOR_COLUMN,
) -> dict[str, pd.DataFrame]:
    """比较两个已完成评价的 ViViT run；基准为空时返回空字典。
    
    输入：
        run_name: str。
        baseline_run_name: str | None。
        vivit_data_root: str | Path | None。
        prediction_column: str。
        baseline_prediction_column: str。
    输出：
        dict[str, pd.DataFrame]。
        """

    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `run_name`。
    run_name = validate_run_name(run_name)
    # 调用 `validate_prediction_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `prediction_column`。
    prediction_column = validate_prediction_column(prediction_column)
    # 检查 `baseline_run_name is None`；根据条件结果选择对应处理分支。
    if baseline_run_name is None:
        # 将 `{}` 作为当前函数结果返回给调用方。
        return {}
    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `baseline_run_name`。
    baseline_run_name = validate_run_name(baseline_run_name)
    # 调用 `validate_prediction_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `baseline_prediction_column`。
    baseline_prediction_column = validate_prediction_column(
        baseline_prediction_column
    )
    # 检查 `baseline_run_name == run_name and baseline_prediction_column == prediction_column`；根据条件结果选择对应处理分支。
    if (
        baseline_run_name == run_name
        and baseline_prediction_column == prediction_column
    ):
        # 抛出 `ValueError('基准 run 与预测列不能与待评价 run 相同')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("基准 run 与预测列不能与待评价 run 相同")
    # 调用 `Path(vivit_data_root or DEFAULT_VIVIT_DATA_ROOT).resolve` 规范化文件路径，并将结果保存到 `vivit_root`。
    vivit_root = Path(
        vivit_data_root or DEFAULT_VIVIT_DATA_ROOT
    ).resolve()
    # 检查 `not vivit_root.is_dir()`；根据条件结果选择对应处理分支。
    if not vivit_root.is_dir():
        # 抛出 `FileNotFoundError(vivit_root)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(vivit_root)
    # 调用 `_validated_evaluation_manifest` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `run_manifest`。
    run_manifest = _validated_evaluation_manifest(
        run_name, vivit_root, prediction_column
    )
    # 调用 `_validated_evaluation_manifest` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `baseline_manifest`。
    baseline_manifest = _validated_evaluation_manifest(
        baseline_run_name, vivit_root,
        baseline_prediction_column,
    )
    # 遍历 `('dataset_signature', 'universe_signature', 'sequence_signature', 'grouping_policy', 'periods',…`，并把当前元素绑定到 `key` 逐项处理。
    for key in (
        "dataset_signature", "universe_signature", "sequence_signature",
        "grouping_policy", "periods", "quantiles", "backtest_mode",
        "price_start", "price_start_offset", "forward_return_formula",
        "portfolio_scope", "execution_mask", "benchmark_column",
        "benchmark_scope", "benchmark_weighting", "benchmark_execution_mask",
        "annual_excess_method", "information_ratio_method",
        "annual_tracking_error_method", "annualization_days",
        "long_short_gross_exposure",
    ):
        # 检查 `run_manifest.get(key) != baseline_manifest.get(key)`；根据条件结果选择对应处理分支。
        if run_manifest.get(key) != baseline_manifest.get(key):
            # 抛出 `RuntimeError(f'两个 run 的 {key} 不一致，禁止直接比较')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError(f"两个 run 的 {key} 不一致，禁止直接比较")

    # 定义 `read_table` 函数，封装这一阶段可独立调用的处理步骤。
    def read_table(run: str, column: str, file_name: str) -> pd.DataFrame:
        """读取对应产物，并按接口约定处理缺失文件。
        
        输入：
            run: str。
            column: str。
            file_name: str。
        输出：
            pd.DataFrame。
            """
        # 计算 `vivit_root / 'quality' / 'runs' / run / evaluation_directory_name(column) / file_name`，并将数值结果保存到 `path`。
        path = (
            vivit_root / "quality" / "runs" / run
            / evaluation_directory_name(column) / file_name
        )
        # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
        if not path.exists():
            # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
            raise FileNotFoundError(path)
        # 复制 `pd.read_parquet(path).copy`，避免后续更新污染原对象，并保存到 `frame`。
        frame = pd.read_parquet(path).copy()
        # 调用 `frame.insert` 把当前字段插入指定位置并保持输出列顺序。
        frame.insert(0, "run_name", run)
        # 调用 `frame.insert` 把当前字段插入指定位置并保持输出列顺序。
        frame.insert(1, "prediction_column", column)
        # 将 `frame` 作为当前函数结果返回给调用方。
        return frame

    # 组织一组有序值，并保存到 `runs`。
    runs = (
        (baseline_run_name, baseline_prediction_column),
        (run_name, prediction_column),
    )
    # 将 `{'ic': pd.concat([read_table(name, column, 'ic_summary.parquet') for name, column in runs], ign…` 作为当前函数结果返回给调用方。
    return {
        "ic": pd.concat(
            [read_table(name, column, "ic_summary.parquet") for name, column in runs],
            ignore_index=True,
        ),
        "spread": pd.concat(
            [read_table(name, column, "spread_summary.parquet") for name, column in runs],
            ignore_index=True,
        ),
        "pyfolio": pd.concat(
            [read_table(name, column, "pyfolio_performance.parquet") for name, column in runs],
            ignore_index=True,
        ),
        "relative": pd.concat(
            [read_table(name, column, "relative_performance.parquet") for name, column in runs],
            ignore_index=True,
        ),
    }


# 定义 `compare_prediction_signal_pearson` 函数，封装这一阶段可独立调用的处理步骤。
def compare_prediction_signal_pearson(
    run_name: str,
    baseline_run_name: str | None,
    vivit_data_root: str | Path | None = None,
    prediction_column: str = FACTOR_COLUMN,
    baseline_prediction_column: str = FACTOR_COLUMN,
) -> dict[str, pd.DataFrame]:
    """按完整业务键配对两个 run，并计算逐日横截面 Pearson。
    
    输入：
        run_name: str。
        baseline_run_name: str | None。
        vivit_data_root: str | Path | None。
        prediction_column: str。
        baseline_prediction_column: str。
    输出：
        dict[str, pd.DataFrame]。
        """

    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `run_name`。
    run_name = validate_run_name(run_name)
    # 调用 `validate_prediction_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `prediction_column`。
    prediction_column = validate_prediction_column(prediction_column)
    # 检查 `baseline_run_name is None`；根据条件结果选择对应处理分支。
    if baseline_run_name is None:
        # 将 `{}` 作为当前函数结果返回给调用方。
        return {}
    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `baseline_run_name`。
    baseline_run_name = validate_run_name(baseline_run_name)
    # 调用 `validate_prediction_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `baseline_prediction_column`。
    baseline_prediction_column = validate_prediction_column(
        baseline_prediction_column
    )
    # 检查 `baseline_run_name == run_name and baseline_prediction_column == prediction_column`；根据条件结果选择对应处理分支。
    if baseline_run_name == run_name and baseline_prediction_column == prediction_column:
        # 抛出 `ValueError('基准 run 与预测列不能与待评价 run 相同')`，阻止不满足数据契约的流程继续执行。
        raise ValueError("基准 run 与预测列不能与待评价 run 相同")
    # 调用 `Path(vivit_data_root or DEFAULT_VIVIT_DATA_ROOT).resolve` 规范化文件路径，并将结果保存到 `vivit_root`。
    vivit_root = Path(
        vivit_data_root or DEFAULT_VIVIT_DATA_ROOT
    ).resolve()
    # 检查 `not vivit_root.is_dir()`；根据条件结果选择对应处理分支。
    if not vivit_root.is_dir():
        # 抛出 `FileNotFoundError(vivit_root)`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError(vivit_root)

    # 定义 `training_manifest` 函数，封装这一阶段可独立调用的处理步骤。
    def training_manifest(name: str) -> dict[str, object]:
        """读取并校验上游与当前产物的 manifest 签名链。
        
        输入：
            name: str。
        输出：
            dict[str, object]。
            """
        # 将 `read_json(vivit_root / 'quality' / 'runs' / name / 'a2_training_manifest.json')` 作为当前函数结果返回给调用方。
        return read_json(
            vivit_root / "quality" / "runs" / name
            / "a2_training_manifest.json"
        )

    # 调用 `training_manifest` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `run_manifest`。
    run_manifest = training_manifest(run_name)
    # 调用 `training_manifest` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `baseline_manifest`。
    baseline_manifest = training_manifest(baseline_run_name)
    # 遍历 `('source_dataset_signature', 'source_rolling_signature', 'sequence_signature')`，并把当前元素绑定到 `key` 逐项处理。
    for key in (
        "source_dataset_signature",
        "source_rolling_signature",
        "sequence_signature",
    ):
        # 检查 `run_manifest.get(key) != baseline_manifest.get(key)`；根据条件结果选择对应处理分支。
        if run_manifest.get(key) != baseline_manifest.get(key):
            # 抛出 `RuntimeError(f'两个 run 的 {key} 不一致，禁止直接比较')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError(f"两个 run 的 {key} 不一致，禁止直接比较")
    # 调用 `_prediction_counts` 执行模型推理并生成样本预测分数，并将结果保存到 `run_counts`。
    run_counts = _prediction_counts(run_manifest)
    # 调用 `_prediction_counts` 执行模型推理并生成样本预测分数，并将结果保存到 `baseline_counts`。
    baseline_counts = _prediction_counts(baseline_manifest)
    # 检查 `run_counts != baseline_counts`；根据条件结果选择对应处理分支。
    if run_counts != baseline_counts:
        # 抛出 `RuntimeError('两个 run 的预测年度或行数不一致，禁止直接比较')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError("两个 run 的预测年度或行数不一致，禁止直接比较")

    # 构造当前步骤所需的列表，并保存到 `keys`。
    keys = ["trade_date", "instrument_id"]

    # 定义 `read_prediction` 函数，封装这一阶段可独立调用的处理步骤。
    def read_prediction(
        name: str, year: int, expected_count: int, column: str
    ) -> pd.DataFrame:
        """读取对应产物，并按接口约定处理缺失文件。
        
        输入：
            name: str。
            year: int。
            expected_count: int。
            column: str。
        输出：
            pd.DataFrame。
            """
        # 计算 `vivit_root / 'predictions' / 'runs' / name / PREDICTION_PATTERN.format(year=year)`，并将数值结果保存到 `path`。
        path = (
            vivit_root / "predictions" / "runs" / name
            / PREDICTION_PATTERN.format(year=year)
        )
        # 检查 `not path.exists()`；根据条件结果选择对应处理分支。
        if not path.exists():
            # 抛出 `FileNotFoundError(path)`，阻止不满足数据契约的流程继续执行。
            raise FileNotFoundError(path)
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `frame`。
        frame = pd.read_parquet(path, columns=[*keys, column])
        # 检查 `len(frame) != expected_count`；根据条件结果选择对应处理分支。
        if len(frame) != expected_count:
            # 抛出 `AssertionError(f'{name} {year} 预测行数 {len(frame):,} != {expected_count:,}')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(
                f"{name} {year} 预测行数 {len(frame):,} != {expected_count:,}"
            )
        # 调用 `frame['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `frame['trade_date']`。
        frame["trade_date"] = frame["trade_date"].astype(str)
        # 调用 `frame['instrument_id'].astype` 统一设备或数据类型，并将结果保存到 `frame['instrument_id']`。
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        # 检查 `frame.duplicated(keys).any()`；根据条件结果选择对应处理分支。
        if frame.duplicated(keys).any():
            # 抛出 `ValueError(f'{name} {year} prediction 业务键重复')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{name} {year} prediction 业务键重复")
        # 调用 `frame[column].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `values`。
        values = frame[column].to_numpy(np.float64)
        # 检查 `not np.isfinite(values).all()`；根据条件结果选择对应处理分支。
        if not np.isfinite(values).all():
            # 抛出 `ValueError(f'{name} {year} 因子含 NaN 或 Inf')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{name} {year} 因子含 NaN 或 Inf")
        # 将 `frame` 作为当前函数结果返回给调用方。
        return frame

    # 构造当前步骤所需的列表，并保存到 `daily_frames`。
    daily_frames: list[pd.DataFrame] = []
    # 遍历 `run_counts.items()`，并把当前元素绑定到 `(year, expected_count)` 逐项处理。
    for year, expected_count in run_counts.items():
        # 检查 `expected_count == 0`；根据条件结果选择对应处理分支。
        if expected_count == 0:
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 调用 `read_prediction(baseline_run_name, year, expected_count, baseli…` 读取源数据，并将结果保存到 `baseline`。
        baseline = read_prediction(
            baseline_run_name, year, expected_count, baseline_prediction_column
        ).rename(columns={baseline_prediction_column: "baseline_signal"})
        # 调用 `read_prediction(run_name, year, expected_count, prediction_colu…` 读取源数据，并将结果保存到 `target`。
        target = read_prediction(
            run_name, year, expected_count, prediction_column
        ).rename(columns={prediction_column: "target_signal"})
        # 调用 `baseline.merge` 按键合并数据，并将对齐后的结果保存到 `paired`。
        paired = baseline.merge(
            target,
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        # 检查 `not paired['_merge'].eq('both').all()`；根据条件结果选择对应处理分支。
        if not paired["_merge"].eq("both").all():
            # 调用 `paired['_merge'].value_counts().to_dict` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `mismatch`。
            mismatch = paired["_merge"].value_counts().to_dict()
            # 抛出 `AssertionError(f'{year} 两个 prediction 的业务键不一致：{mismatch}')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(
                f"{year} 两个 prediction 的业务键不一致：{mismatch}"
            )
        # 调用 `paired.drop` 执行 `drop` 对应的明确变换，并产生后续步骤使用的结果，并将结果保存到 `paired`。
        paired = paired.drop(columns="_merge")
        # 调用 `paired['baseline_signal'].pow` 逐元素计算指定次幂，并将结果保存到 `paired['baseline_square']`。
        paired["baseline_square"] = paired["baseline_signal"].pow(2)
        # 调用 `paired['target_signal'].pow` 逐元素计算指定次幂，并将结果保存到 `paired['target_square']`。
        paired["target_square"] = paired["target_signal"].pow(2)
        # 计算 `paired['baseline_signal'] * paired['target_signal']`，并将数值结果保存到 `paired['cross_product']`。
        paired["cross_product"] = (
            paired["baseline_signal"] * paired["target_signal"]
        )
        # 调用 `paired.groupby('trade_date', sort=True).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `stats`。
        stats = paired.groupby("trade_date", sort=True).agg(
            observations=("target_signal", "size"),
            baseline_sum=("baseline_signal", "sum"),
            target_sum=("target_signal", "sum"),
            baseline_square_sum=("baseline_square", "sum"),
            target_square_sum=("target_square", "sum"),
            cross_product_sum=("cross_product", "sum"),
        )
        # 调用 `stats['observations'].to_numpy` 把表格列转换为指定精度的 NumPy 数组，并将结果保存到 `count`。
        count = stats["observations"].to_numpy(np.float64)
        # 计算 `count * stats['baseline_square_sum'].to_numpy(np.float64) - stats['baseline_sum'].to_numpy(np.f…`，并将数值结果保存到 `baseline_variation`。
        baseline_variation = (
            count * stats["baseline_square_sum"].to_numpy(np.float64)
            - stats["baseline_sum"].to_numpy(np.float64) ** 2
        )
        # 计算 `count * stats['target_square_sum'].to_numpy(np.float64) - stats['target_sum'].to_numpy(np.float…`，并将数值结果保存到 `target_variation`。
        target_variation = (
            count * stats["target_square_sum"].to_numpy(np.float64)
            - stats["target_sum"].to_numpy(np.float64) ** 2
        )
        # 计算 `(count < 2) | (baseline_variation <= 0) | (target_variation <= 0)`，并将数值结果保存到 `invalid`。
        invalid = (
            (count < 2)
            | (baseline_variation <= 0)
            | (target_variation <= 0)
        )
        # 检查 `invalid.any()`；根据条件结果选择对应处理分支。
        if invalid.any():
            # 读取 `stats.index[invalid].astype(str).tolist()[:5]` 的当前值，并赋给 `dates`。
            dates = stats.index[invalid].astype(str).tolist()[:5]
            # 抛出 `ValueError(f'{year} 存在样本不足或常数信号截面：{dates}')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(
                f"{year} 存在样本不足或常数信号截面：{dates}"
            )
        # 计算 `count * stats['cross_product_sum'].to_numpy(np.float64) - stats['baseline_sum'].to_numpy(np.flo…`，并将数值结果保存到 `covariance`。
        covariance = (
            count * stats["cross_product_sum"].to_numpy(np.float64)
            - stats["baseline_sum"].to_numpy(np.float64)
            * stats["target_sum"].to_numpy(np.float64)
        )
        # 计算 `covariance / np.sqrt(baseline_variation * target_variation)`，并将数值结果保存到 `pearson`。
        pearson = covariance / np.sqrt(
            baseline_variation * target_variation
        )
        # 把本轮结果追加到 `daily_frames`，累积后续需要汇总的记录。
        daily_frames.append(
            pd.DataFrame(
                {
                    "run_name": run_name,
                    "baseline_run_name": baseline_run_name,
                    "prediction_column": prediction_column,
                    "baseline_prediction_column": baseline_prediction_column,
                    "year": year,
                    "trade_date": stats.index.astype(str),
                    "observations": stats["observations"].to_numpy(np.int64),
                    "pearson": np.clip(pearson, -1.0, 1.0),
                }
            )
        )

    # 检查 `not daily_frames`；根据条件结果选择对应处理分支。
    if not daily_frames:
        # 抛出 `FileNotFoundError('两个 run 均没有可比较的非空年度预测')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("两个 run 均没有可比较的非空年度预测")
    # 调用 `pd.concat` 拼接各批结果，并将合并值保存到 `daily`。
    daily = pd.concat(daily_frames, ignore_index=True)
    # 调用 `daily.groupby('year', as_index=False).agg` 按预先声明的聚合规则汇总分组数据，并将结果保存到 `annual`。
    annual = daily.groupby("year", as_index=False).agg(
        dates=("trade_date", "size"),
        matched_rows=("observations", "sum"),
        mean_daily_pearson=("pearson", "mean"),
        std_daily_pearson=("pearson", "std"),
        median_daily_pearson=("pearson", "median"),
        min_daily_pearson=("pearson", "min"),
        max_daily_pearson=("pearson", "max"),
    )
    # 调用 `annual.insert` 把当前字段插入指定位置并保持输出列顺序。
    annual.insert(0, "baseline_run_name", baseline_run_name)
    # 调用 `annual.insert` 把当前字段插入指定位置并保持输出列顺序。
    annual.insert(0, "baseline_prediction_column", baseline_prediction_column)
    # 调用 `annual.insert` 把当前字段插入指定位置并保持输出列顺序。
    annual.insert(0, "prediction_column", prediction_column)
    # 调用 `annual.insert` 把当前字段插入指定位置并保持输出列顺序。
    annual.insert(0, "run_name", run_name)
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `overall`。
    overall = pd.DataFrame(
        {
            "run_name": [run_name],
            "baseline_run_name": [baseline_run_name],
            "years": [daily["year"].nunique()],
            "prediction_column": [prediction_column],
            "baseline_prediction_column": [baseline_prediction_column],
            "dates": [len(daily)],
            "matched_rows": [int(daily["observations"].sum())],
            "mean_daily_pearson": [daily["pearson"].mean()],
            "std_daily_pearson": [daily["pearson"].std(ddof=1)],
            "median_daily_pearson": [daily["pearson"].median()],
            "min_daily_pearson": [daily["pearson"].min()],
            "max_daily_pearson": [daily["pearson"].max()],
        }
    )
    # 将 `{'daily': daily, 'annual': annual, 'overall': overall}` 作为当前函数结果返回给调用方。
    return {"daily": daily, "annual": annual, "overall": overall}
# 定义 `audit_evaluation_inputs` 函数，封装这一阶段可独立调用的处理步骤。
def audit_evaluation_inputs(
    run_name: str = DEFAULT_RUN_NAME,
    vivit_data_root: str | Path | None = None,
    reference_data_root: str | Path | None = None,
    backtest_mode: str = DEFAULT_BACKTEST_MODE,
    prediction_column: str = FACTOR_COLUMN,
) -> pd.DataFrame:
    """只读审计 ViViT 预测、签名链、Universe 键和 T+10 日历覆盖。
    
    输入：
        run_name: str。
        vivit_data_root: str | Path | None。
        reference_data_root: str | Path | None。
        backtest_mode: str。
        prediction_column: str。
    输出：
        pd.DataFrame。
        """

    # 调用 `validate_run_name` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `run_name`。
    run_name = validate_run_name(run_name)
    # 调用 `validate_prediction_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `prediction_column`。
    prediction_column = validate_prediction_column(prediction_column)
    # 调用 `resolve_backtest_mode` 解析用户配置并转换为内部统一口径，并将结果保存到 `mode`。
    mode = resolve_backtest_mode(backtest_mode)
    # 调用 `_roots` 解析本次运行对应的数据和输出目录，并将结果保存到 `(vivit_root, reference_root)`。
    vivit_root, reference_root = _roots(vivit_data_root, reference_data_root)
    # 调用 `_manifests` 读取或生成记录数据来源与配置签名的 manifest，并将结果保存到 `manifests`。
    manifests = _manifests(run_name, vivit_root, reference_root)
    # 调用 `_prediction_counts` 执行模型推理并生成样本预测分数，并将结果保存到 `counts`。
    counts = _prediction_counts(manifests["training"])
    # 计算 `vivit_root / 'predictions' / 'runs' / run_name`，并将数值结果保存到 `prediction_dir`。
    prediction_dir = vivit_root / "predictions" / "runs" / run_name
    # 计算 `reference_root / 'universe'`，并将数值结果保存到 `universe_dir`。
    universe_dir = reference_root / "universe"
    # 构造当前步骤所需的列表，并保存到 `records`。
    records: list[dict[str, object]] = []
    # 设置 `last_factor_date` 的固定配置值，供后续流程统一使用。
    last_factor_date: str | None = None

    # 遍历 `counts.items()`，并把当前元素绑定到 `(year, expected_count)` 逐项处理。
    for year, expected_count in counts.items():
        # 计算 `prediction_dir / PREDICTION_PATTERN.format(year=year)`，并将数值结果保存到 `prediction_path`。
        prediction_path = prediction_dir / PREDICTION_PATTERN.format(year=year)
        # 检查 `expected_count == 0`；根据条件结果选择对应处理分支。
        if expected_count == 0:
            # 检查 `prediction_path.exists()`；根据条件结果选择对应处理分支。
            if prediction_path.exists():
                # 抛出 `AssertionError(f'{year} factor_counts=0 但存在预测文件：{prediction_path}')`，阻止不满足数据契约的流程继续执行。
                raise AssertionError(
                    f"{year} factor_counts=0 但存在预测文件：{prediction_path}"
                )
            # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
            records.append(
                {
                    "year": year,
                    "prediction_rows": 0,
                    "universe_rows": 0,
                    "dates": 0,
                    "first_date": None,
                    "last_date": None,
                    "can_enter_t1_rows": 0,
                    "status": "empty_prediction_expected",
                }
            )
            # 跳过当前元素的剩余步骤，继续处理下一项。
            continue
        # 检查 `not prediction_path.exists()`；根据条件结果选择对应处理分支。
        if not prediction_path.exists():
            # 抛出 `FileNotFoundError(prediction_path)`，阻止不满足数据契约的流程继续执行。
            raise FileNotFoundError(prediction_path)
        # 调用 `pd.read_parquet` 读取源数据，并将结果保存到 `prediction`。
        prediction = pd.read_parquet(
            prediction_path,
            columns=["trade_date", "instrument_id", prediction_column],
        )
        # 调用 `prediction['trade_date'].astype` 统一设备或数据类型，并将结果保存到 `prediction['trade_date']`。
        prediction["trade_date"] = prediction["trade_date"].astype(str)
        # 调用 `prediction['instrument_id'].astype` 统一设备或数据类型，并将结果保存到 `prediction['instrument_id']`。
        prediction["instrument_id"] = prediction["instrument_id"].astype(str)
        # 检查 `len(prediction) != expected_count`；根据条件结果选择对应处理分支。
        if len(prediction) != expected_count:
            # 抛出 `AssertionError(f'{year} 预测行数 {len(prediction):,} != {expected_count:,}')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(
                f"{year} 预测行数 {len(prediction):,} != {expected_count:,}"
            )
        # 检查 `prediction.duplicated(['trade_date', 'instrument_id']).any()`；根据条件结果选择对应处理分支。
        if prediction.duplicated(["trade_date", "instrument_id"]).any():
            # 抛出 `ValueError(f'{year} prediction 业务键重复')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{year} prediction 业务键重复")
        # 检查 `not prediction['trade_date'].str.startswith(str(year)).all()`；根据条件结果选择对应处理分支。
        if not prediction["trade_date"].str.startswith(str(year)).all():
            # 抛出 `ValueError(f'{year} prediction 含跨年度日期')`，阻止不满足数据契约的流程继续执行。
            raise ValueError(f"{year} prediction 含跨年度日期")
        # 检查 `not np.isfinite(prediction[prediction_column].to_numpy(np.float64)).all()`；根据条件结果选择对应处理分支。
        if not np.isfinite(
            prediction[prediction_column].to_numpy(np.float64)
        ).all():
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
            # 调用 `joined['_merge'].value_counts().to_dict` 按股票和交易日键对齐来自不同来源的数据，并将结果保存到 `mismatch`。
            mismatch = joined["_merge"].value_counts().to_dict()
            # 抛出 `AssertionError(f'{year} prediction 与 T 日 Universe 键不一致：{mismatch}')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(
                f"{year} prediction 与 T 日 Universe 键不一致：{mismatch}"
            )
        # 检查 `joined['can_enter_t1'].isna().any()`；根据条件结果选择对应处理分支。
        if joined["can_enter_t1"].isna().any():
            # 抛出 `AssertionError(f'{year} can_enter_t1 含缺失值')`，阻止不满足数据契约的流程继续执行。
            raise AssertionError(f"{year} can_enter_t1 含缺失值")
        # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `first_date`。
        first_date = str(prediction["trade_date"].min())
        # 调用 `str` 把当前值转换为统一字符串表示，并将结果保存到 `year_last_date`。
        year_last_date = str(prediction["trade_date"].max())
        # 调用 `max` 汇总当前数据，并将统计量保存到 `last_factor_date`。
        last_factor_date = max(last_factor_date or year_last_date, year_last_date)
        # 把本轮结果追加到 `records`，累积后续需要汇总的记录。
        records.append(
            {
                "year": year,
                "prediction_rows": len(prediction),
                "universe_rows": len(expected),
                "dates": prediction["trade_date"].nunique(),
                "first_date": first_date,
                "last_date": year_last_date,
                "can_enter_t1_rows": int(joined["can_enter_t1"].astype(bool).sum()),
                "status": "passed",
            }
        )

    # 检查 `last_factor_date is None`；根据条件结果选择对应处理分支。
    if last_factor_date is None:
        # 抛出 `FileNotFoundError('没有可评价的非空年度预测')`，阻止不满足数据契约的流程继续执行。
        raise FileNotFoundError("没有可评价的非空年度预测")
    # 调用 `_calendar_last_date` 按交易日历定位收益窗口对应日期，并将结果保存到 `final_price_date`。
    final_price_date = _calendar_last_date(last_factor_date, str(mode["name"]))
    # 调用 `pd.DataFrame` 转换数据表示，并将标准化对象保存到 `audit`。
    audit = pd.DataFrame(records)
    # 用本轮键值结果更新 `audit.attrs`，补齐后续查询所需字段。
    audit.attrs.update(
        {
            "run_name": run_name,
            "model_family": "ViViT",
            "factor_role": manifests["training"].get(
                "factor_role", "model_prediction"
            ),
            "source_run_name": manifests["training"].get("source_run_name"),
            "prediction_pattern": PREDICTION_PATTERN,
            "prediction_column": prediction_column,
            "last_factor_date": last_factor_date,
            "final_price_date": final_price_date,
            "backtest_mode": str(mode["name"]),
            "price_start_offset": int(mode["price_start_offset"]),
            "dataset_signature": manifests["dataset"]["signature"],
            "universe_signature": manifests["universe"]["signature"],
            "sequence_signature": manifests["sequence"]["signature"],
        }
    )
    # 将 `audit` 作为当前函数结果返回给调用方。
    return audit

# 定义内部辅助函数 `_validated_evaluation_manifest`，封装仅供本模块复用的计算步骤。
def _validated_evaluation_manifest(
    run_name: str, vivit_root: Path,
    prediction_column: str = FACTOR_COLUMN,
) -> dict[str, object]:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。
    
    输入：
        run_name: str。
        vivit_root: Path。
        prediction_column: str。
    输出：
        dict[str, object]。
        """
    # 调用 `validate_prediction_column` 校验输入、配置或产物是否满足当前阶段的数据契约，并将结果保存到 `prediction_column`。
    prediction_column = validate_prediction_column(prediction_column)
    # 计算 `vivit_root / 'quality' / 'runs' / run_name / evaluation_directory_name(prediction_column) / 'fa…`，并将数值结果保存到 `path`。
    path = (
        vivit_root / "quality" / "runs" / run_name
        / evaluation_directory_name(prediction_column) / "factor_evaluation_manifest.json"
    )
    # 调用 `read_json` 读取源数据，并将结果保存到 `manifest`。
    manifest = read_json(path)
    # 检查 `manifest.get('schema_version') != 2 or 'backtest_mode' not in manifest`；根据条件结果选择对应处理分支。
    if manifest.get("schema_version") != 2 or "backtest_mode" not in manifest:
        # 抛出 `RuntimeError(f'{run_name} 是旧版评价产物，请用新版 Notebook 03 重新运行评价')`，阻止不满足数据契约的流程继续执行。
        raise RuntimeError(
            f"{run_name} 是旧版评价产物，请用新版 Notebook 03 重新运行评价"
        )
    # 调用 `resolve_backtest_mode` 解析用户配置并转换为内部统一口径，并将结果保存到 `mode`。
    mode = resolve_backtest_mode(str(manifest["backtest_mode"]))
    # 构造键值映射，并保存到 `expected`，供后续按键查询。
    expected = {
        "model_family": "ViViT",
        "grouping_policy": GROUPING_POLICY,
        "periods": list(PERIODS),
        "quantiles": QUANTILES,
        "price_start": mode["price_start"],
        "price_start_offset": mode["price_start_offset"],
        "portfolio_scope": mode["portfolio_scope"],
        "execution_mask": mode["execution_mask"],
        "benchmark_column": EQUAL_WEIGHT_MARKET_COLUMN,
        "benchmark_scope": BENCHMARK_SCOPE,
        "benchmark_weighting": "daily_equal_weight",
        "benchmark_execution_mask": "none",
        "prediction_column": prediction_column,
    }
    # 遍历 `expected.items()`，并把当前元素绑定到 `(key, value)` 逐项处理。
    for key, value in expected.items():
        # 调用 `manifest.get` 按键读取映射中的可选字段，并在缺失时使用约定默认值，并将结果保存到 `actual`。
        actual = manifest.get(key, FACTOR_COLUMN if key == "prediction_column" else None)
        # 检查 `actual != value`；根据条件结果选择对应处理分支。
        if actual != value:
            # 抛出 `RuntimeError(f'{run_name} 评价口径不一致：{key}={actual!r}，预期 {value!r}')`，阻止不满足数据契约的流程继续执行。
            raise RuntimeError(
                f"{run_name} 评价口径不一致：{key}={actual!r}，预期 {value!r}"
            )
    # 将 `manifest` 作为当前函数结果返回给调用方。
    return manifest

