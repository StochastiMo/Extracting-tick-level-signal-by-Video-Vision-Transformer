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

VIVIT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = VIVIT_DIR.parent
VIT_DIR = PROJECT_ROOT / "ViT"
DEFAULT_VIVIT_DATA_ROOT = VIVIT_DIR / "data"
DEFAULT_REFERENCE_DATA_ROOT = VIT_DIR / "data"
DEFAULT_RUN_NAME = "sigmoid_daybatch_rankic"
PREDICTION_PATTERN = "vivit_predictions.{year}.parquet"
FACTOR_COLUMN = "prediction_ensemble"
PREDICTION_COLUMNS = (
    "prediction_ensemble",
    "prediction_ensemble_unprojected",
    "prediction_ensemble_projected",
    "prediction_ensemble_raw",
    "prediction_ensemble_alpha",
    "prediction_ensemble_style",
    "prediction_ensemble_total_return",
)
EVALUATION_DIRECTORY_NAMES = {
    "prediction_ensemble": "factor_evaluation",
    "prediction_ensemble_unprojected": "factor_evaluation_unprojected",
    "prediction_ensemble_projected": "factor_evaluation_projected",
    "prediction_ensemble_raw": "factor_evaluation_raw",
    "prediction_ensemble_alpha": "factor_evaluation_alpha",
    "prediction_ensemble_style": "factor_evaluation_style",
    "prediction_ensemble_total_return": "factor_evaluation_total_return",
}
EXPERIMENT_REPORT_NAMES = {
    "prediction_ensemble": "EXPERIMENT_REPORT.md",
    "prediction_ensemble_unprojected": "EXPERIMENT_REPORT_unprojected.md",
    "prediction_ensemble_projected": "EXPERIMENT_REPORT_projected.md",
    "prediction_ensemble_raw": "EXPERIMENT_REPORT_raw.md",
    "prediction_ensemble_alpha": "EXPERIMENT_REPORT_alpha.md",
    "prediction_ensemble_style": "EXPERIMENT_REPORT_style.md",
    "prediction_ensemble_total_return": "EXPERIMENT_REPORT_total_return.md",
}
PERIODS = (1, 5, 10)
QUANTILES = 5
MAX_LOSS = 0.001
BACKTEST_MODES = (
    "mode1_t_with_t1_filter",
    "mode2_t_no_filter",
    "mode3_t1_no_filter",
)
DEFAULT_BACKTEST_MODE = "mode2_t_no_filter"
GROUPING_POLICY = "T_full_cross_section_before_execution_filter"
EQUAL_WEIGHT_MARKET_COLUMN = "Equal-weight market"
BENCHMARK_SCOPE = (
    "feature_valid_and_in_universe_t_full_factor_pool_after_required_price_filter"
)
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

if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from vit_stage01_foundation import CALENDAR_PATH, atomic_json, normalize_dates  # noqa: E402


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


def validate_prediction_column(value: str) -> str:
    """校验待评价预测列是否属于允许的信号口径。"""
    prediction_column = str(value).strip()
    if prediction_column not in PREDICTION_COLUMNS:
        raise ValueError(
            f"未知预测列：{value!r}；可选 {PREDICTION_COLUMNS}"
        )
    return prediction_column


def evaluation_directory_name(prediction_column: str = FACTOR_COLUMN) -> str:
    """根据预测列生成互不覆盖的评价目录名。"""
    return EVALUATION_DIRECTORY_NAMES[
        validate_prediction_column(prediction_column)
    ]


def experiment_report_name(prediction_column: str = FACTOR_COLUMN) -> str:
    """根据预测列生成对应实验报告文件名。"""
    return EXPERIMENT_REPORT_NAMES[
        validate_prediction_column(prediction_column)
    ]


def resolve_backtest_mode(backtest_mode: str) -> dict[str, object]:
    """解析并校验统一样本外回测模式。"""
    mode = str(backtest_mode).strip()
    if mode not in BACKTEST_MODE_CONFIGS:
        raise ValueError(
            f"未知回测模式：{backtest_mode!r}；可选 {BACKTEST_MODES}"
        )
    return {"name": mode, **BACKTEST_MODE_CONFIGS[mode]}


def read_json(path: Path) -> dict[str, object]:
    """读取 JSON 对象；目标不存在时按接口约定返回空结果。"""
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _roots(
    vivit_data_root: str | Path | None,
    reference_data_root: str | Path | None,
) -> tuple[Path, Path]:
    """解析当前流程的数据、模型或质量目录。"""
    vivit_root = Path(vivit_data_root or DEFAULT_VIVIT_DATA_ROOT).resolve()
    reference_root = Path(
        reference_data_root or DEFAULT_REFERENCE_DATA_ROOT
    ).resolve()
    if not vivit_root.is_dir():
        raise FileNotFoundError(vivit_root)
    if not reference_root.is_dir():
        raise FileNotFoundError(reference_root)
    return vivit_root, reference_root


def _manifests(
    run_name: str, vivit_root: Path, reference_root: Path
) -> dict[str, dict[str, object]]:
    """读取并校验上游与当前产物的 manifest 签名链。"""
    quality_run = vivit_root / "quality" / "runs" / run_name
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
    training = result["training"]
    sequence = result["sequence"]
    dataset = result["dataset"]
    rolling = result["rolling"]
    universe = result["universe"]
    sequence_payload = sequence.get("payload", {})
    dataset_payload = dataset.get("payload", {})
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
    mismatches = [
        f"{name}: {actual} != {wanted}"
        for name, (actual, wanted) in expected.items()
        if actual != wanted
    ]
    if mismatches:
        raise AssertionError("签名链不一致：" + "; ".join(mismatches))
    return result


def _prediction_counts(training_manifest: dict[str, object]) -> dict[int, int]:
    """处理预测文件、预测列或对应的样本数量审计。"""
    years = tuple(int(year) for year in training_manifest.get("years", ()))
    if not years:
        raise ValueError("训练 manifest 缺少 years")
    raw_counts = training_manifest.get("factor_counts")
    if not isinstance(raw_counts, dict):
        raise ValueError("训练 manifest 缺少 factor_counts")
    counts = {
        year: int(raw_counts.get(str(year), raw_counts.get(year, -1)))
        for year in years
    }
    if any(value < 0 for value in counts.values()):
        raise ValueError("训练 manifest 的 factor_counts 不完整")
    return counts


def _calendar_last_date(
    last_factor_date: str, backtest_mode: str = DEFAULT_BACKTEST_MODE
) -> str:
    """返回指定交易所日历中不晚于给定日期的最后交易日。"""
    calendar = pd.read_parquet(
        CALENDAR_PATH,
        columns=["trade_days", "s_info_exchmarket"],
        filters=[("s_info_exchmarket", "==", "SSE")],
    )
    dates = (
        normalize_dates(calendar["trade_days"])
        .drop_duplicates()
        .sort_values()
        .to_numpy(dtype=str)
    )
    position = int(np.searchsorted(dates, last_factor_date))
    if position >= len(dates) or dates[position] != last_factor_date:
        raise AssertionError(f"{last_factor_date} 不在 SSE 交易日历")
    mode = resolve_backtest_mode(backtest_mode)
    final_offset = int(mode["price_start_offset"]) + max(PERIODS)
    if position + final_offset >= len(dates):
        raise ValueError(
            f"交易日历不足以覆盖最大 {final_offset} 个交易日收益期限"
        )
    return str(dates[position + final_offset])


def assign_factor_quantiles(
    factor: pd.Series, quantiles: int = QUANTILES
) -> pd.Series:
    """使用稳定排序在每个完整 T 日截面分配近似等量分位组。"""

    if quantiles < 2:
        raise ValueError("quantiles 必须至少为 2")
    if not isinstance(factor.index, pd.MultiIndex) or "date" not in factor.index.names:
        raise ValueError("factor 必须使用含 date level 的 MultiIndex")
    groups = factor.groupby(level="date", sort=False)
    ordinal_rank = groups.rank(method="first", ascending=True)
    group_size = groups.transform("size")
    if (group_size < quantiles).any():
        raise ValueError("存在样本数少于分位组数的日期")
    result = pd.Series(
        np.floor(
            (ordinal_rank.to_numpy(np.float64) - 1)
            * quantiles
            / group_size.to_numpy(np.float64)
        ).astype(np.int64) + 1,
        index=factor.index,
        name="factor_quantile",
    )
    counts = result.groupby(
        [pd.Grouper(level="date"), result], observed=True
    ).size().unstack("factor_quantile")
    if counts.isna().any(axis=None) or (
        counts.max(axis=1) - counts.min(axis=1)
    ).gt(1).any():
        raise AssertionError("T 日完整截面分位组不均衡")
    return result


def _normalized_vit_manifest(
    manifests: dict[str, dict[str, object]],
) -> dict[str, object]:
    """读取并校验上游与当前产物的 manifest 签名链。"""
    training = dict(manifests["training"])
    training["dataset_signature"] = manifests["dataset"]["signature"]
    training["universe_signature"] = manifests["universe"]["signature"]
    return training


def _create_vit_evaluation_view(
    temporary_root: Path,
    run_name: str,
    vivit_root: Path,
    reference_root: Path,
    manifests: dict[str, dict[str, object]],
    prediction_column: str = FACTOR_COLUMN,
) -> Path:
    """在临时目录建立 ViT evaluator 所需的只读文件名视图。"""

    prediction_column = validate_prediction_column(prediction_column)
    for name in ("feature_store", "universe", "labels_t10"):
        (temporary_root / name).symlink_to(
            reference_root / name, target_is_directory=True
        )
    source_predictions = vivit_root / "predictions" / "runs" / run_name
    view_predictions = temporary_root / "predictions" / "runs" / run_name
    view_predictions.mkdir(parents=True)
    counts = _prediction_counts(manifests["training"])
    for year, count in counts.items():
        if count:
            source = source_predictions / PREDICTION_PATTERN.format(year=year)
            target = view_predictions / f"vit_predictions.{year}.parquet"
            if prediction_column == FACTOR_COLUMN:
                target.symlink_to(source)
            else:
                prediction = pd.read_parquet(
                    source,
                    columns=["trade_date", "instrument_id", prediction_column],
                ).rename(columns={prediction_column: FACTOR_COLUMN})
                prediction.to_parquet(target, index=False)
                del prediction

    view_quality_run = temporary_root / "quality" / "runs" / run_name
    view_quality_run.mkdir(parents=True)
    atomic_json(  # 原子落盘，避免半成品
        _normalized_vit_manifest(manifests),
        view_quality_run / "a2_training_manifest.json",
    )
    evaluation_target = vivit_root / "quality" / "runs" / run_name / (
        evaluation_directory_name(prediction_column)
    )
    evaluation_target.mkdir(parents=True, exist_ok=True)
    (view_quality_run / "factor_evaluation").symlink_to(  # 无梯度计算验证指标
        evaluation_target, target_is_directory=True
    )
    return evaluation_target


def _markdown_table(frame: pd.DataFrame) -> str:
    """把结构化结果转换为 Markdown 表格。"""
    if frame.empty:
        return "（暂无记录）"
    clean = frame.copy().fillna("")
    headers = [str(column) for column in clean.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in clean.astype(str).itertuples(index=False, name=None):
        lines.append(
            "| " + " | ".join(value.replace("|", "\\|") for value in row) + " |"
        )
    return "\n".join(lines)


def _read_optional_parquet(path: Path) -> pd.DataFrame:
    """读取对应产物，并按接口约定处理缺失文件。"""
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def generate_experiment_report(
    run_name: str,
    vivit_data_root: str | Path | None = None,
    reference_data_root: str | Path | None = None,
    prediction_column: str = FACTOR_COLUMN,
) -> Path:
    """生成同时包含 ViViT 训练与因子评价结果的 run 专属报告。"""

    run_name = validate_run_name(run_name)
    prediction_column = validate_prediction_column(prediction_column)
    evaluation_name = evaluation_directory_name(prediction_column)  # 无梯度计算验证指标
    vivit_root, reference_root = _roots(vivit_data_root, reference_data_root)
    manifests = _manifests(run_name, vivit_root, reference_root)  # 读取或生成来源清单
    quality_run = vivit_root / "quality" / "runs" / run_name
    evaluation_dir = quality_run / evaluation_name
    training = _read_optional_parquet(
        quality_run / "vivit_training_summary.parquet"
    )
    if training.empty:
        training_yearly = pd.DataFrame()
    else:
        aggregations: dict[str, tuple[str, str]] = {
            "completed_seeds": ("seed", "nunique"),
            "mean_best_validation_ic": ("best_validation_ic", "mean"),
            "min_best_validation_ic": ("best_validation_ic", "min"),
            "mean_best_epoch": ("best_epoch", "mean"),
        }
        for output, source in (
            ("mean_best_validation_rank_ic", "best_validation_rank_ic"),
            ("mean_best_validation_daily_pearson_ic", "best_validation_daily_pearson_ic"),
            ("mean_best_validation_pooled_ic", "best_validation_pooled_ic"),
        ):
            if source in training.columns:
                aggregations[output] = (source, "mean")
        training_yearly = training.groupby(  # 按既定口径聚合
            "forecast_year", as_index=False
        ).agg(**aggregations)

    sequence_summary = _read_optional_parquet(
        vivit_root / "aligned_sequence_store" / "rolling_split_summary.parquet"
    )
    final_audit = _read_optional_parquet(evaluation_dir / "final_audit.parquet")
    ic_summary = _read_optional_parquet(evaluation_dir / "ic_summary.parquet")
    spread_summary = _read_optional_parquet(evaluation_dir / "spread_summary.parquet")
    pyfolio = _read_optional_parquet(evaluation_dir / "pyfolio_performance.parquet")
    relative_performance = _read_optional_parquet(
        evaluation_dir / "relative_performance.parquet"
    )
    evaluation_manifest_path = evaluation_dir / "factor_evaluation_manifest.json"
    evaluation_manifest = (
        read_json(evaluation_manifest_path)
        if evaluation_manifest_path.exists()
        else None
    )
    if evaluation_manifest:
        start_offset = int(evaluation_manifest.get("price_start_offset", 0))
        period_description = (
            "1/5/10D 对应 T→T+1/T+5/T+10"
            if start_offset == 0
            else "1/5/10D 对应 T+1→T+2/T+6/T+11"
        )
        execution_description = (
            "组合收益和换手应用 `can_enter_t1`"
            if evaluation_manifest.get("execution_mask") == "can_enter_t1"
            else "组合收益和换手不应用 `can_enter_t1` 未来状态过滤"
        )
        evaluation_policy_lines = [
            f"- 回测模式：`{evaluation_manifest.get('backtest_mode')}`；{execution_description}。",
            f"- 收益公式：`{evaluation_manifest.get('forward_return_formula')}`；{period_description}。",
            "- 全市场等权基准使用完整因子覆盖池和相同收益时点，不应用 `can_enter_t1`。",
            "- Q5 与 Q5−Q1 均报告相对市场的超额年化、跟踪误差和信息比。",
        ]
    else:
        evaluation_policy_lines = ["- 因子评价尚未按新版回测口径生成。"]
    training_manifest = manifests["training"]
    missing_batch_value = (
        "full_trade_date"
        if training_manifest.get("train_batch_mode") == "trade_date"
        else "not_recorded"
    )
    fixed_batch_size = training_manifest.get(
        "fixed_batch_size", missing_batch_value
    )
    inference_batch_size = training_manifest.get(
        "inference_batch_size", missing_batch_value
    )
    derivation = training_manifest.get("factor_derivation")
    if isinstance(derivation, dict):
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
    else:
        derivation_lines = []
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
    if derivation_lines:
        insert_at = lines.index("## 对齐历史覆盖")
        lines[insert_at:insert_at] = derivation_lines
    path = quality_run / experiment_report_name(prediction_column)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    temporary.write_text("\n".join(lines), encoding="utf-8")
    temporary.replace(path)
    print(f"实验报告：{path}")
    return path


def run_oos_backtest(
    run_name: str = DEFAULT_RUN_NAME,
    vivit_data_root: str | Path | None = None,
    reference_data_root: str | Path | None = None,
    backtest_mode: str = DEFAULT_BACKTEST_MODE,
    prediction_column: str = FACTOR_COLUMN,
) -> pd.DataFrame:
    """按指定共享口径执行 ViViT Alphalens/Pyfolio 评价。"""

    run_name = validate_run_name(run_name)
    prediction_column = validate_prediction_column(prediction_column)
    mode = resolve_backtest_mode(backtest_mode)
    vivit_root, reference_root = _roots(vivit_data_root, reference_data_root)
    input_audit = audit_evaluation_inputs(
        run_name, vivit_root, reference_root, str(mode["name"]),
        prediction_column,
    )
    manifests = _manifests(run_name, vivit_root, reference_root)  # 读取或生成来源清单
    try:
        import vit_stage08_factor_evaluation as vit_evaluation
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "请使用 Jupyter 内核 `ViT Factor Eval (Alphalens/Pyfolio)`。"
        ) from error

    with tempfile.TemporaryDirectory(
        prefix=f"vivit_factor_eval_{run_name}_"
    ) as directory:
        temporary_root = Path(directory)
        evaluation_dir = _create_vit_evaluation_view(
            temporary_root, run_name, vivit_root, reference_root, manifests,
            prediction_column,
        )
        original_report = vit_evaluation.generate_experiment_report

        def vivit_report_adapter(**_kwargs: object) -> Path:
            """把 ViT 报告回调重定向到当前 ViViT run 和预测列。"""
            return generate_experiment_report(
                run_name, vivit_root, reference_root, prediction_column
            )

        vit_evaluation.generate_experiment_report = vivit_report_adapter
        try:
            result = vit_evaluation.run_oos_backtest(  # 无梯度计算验证指标
                run_name=run_name,
                data_root=temporary_root,
                backtest_mode=str(mode["name"]),
            )
        finally:
            vit_evaluation.generate_experiment_report = original_report

    manifest_path = evaluation_dir / "factor_evaluation_manifest.json"
    evaluation_manifest = read_json(manifest_path)
    training_manifest = manifests["training"]
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
    atomic_json(evaluation_manifest, manifest_path)  # 原子落盘，避免半成品
    generate_experiment_report(
        run_name, vivit_root, reference_root, prediction_column
    )
    return result


def compare_factor_evaluations(
    run_name: str,
    baseline_run_name: str | None,
    vivit_data_root: str | Path | None = None,
    prediction_column: str = FACTOR_COLUMN,
    baseline_prediction_column: str = FACTOR_COLUMN,
) -> dict[str, pd.DataFrame]:
    """比较两个已完成评价的 ViViT run；基准为空时返回空字典。"""

    run_name = validate_run_name(run_name)
    prediction_column = validate_prediction_column(prediction_column)
    if baseline_run_name is None:
        return {}
    baseline_run_name = validate_run_name(baseline_run_name)
    baseline_prediction_column = validate_prediction_column(
        baseline_prediction_column
    )
    if (
        baseline_run_name == run_name
        and baseline_prediction_column == prediction_column
    ):
        raise ValueError("基准 run 与预测列不能与待评价 run 相同")
    vivit_root = Path(
        vivit_data_root or DEFAULT_VIVIT_DATA_ROOT
    ).resolve()
    if not vivit_root.is_dir():
        raise FileNotFoundError(vivit_root)
    run_manifest = _validated_evaluation_manifest(
        run_name, vivit_root, prediction_column
    )
    baseline_manifest = _validated_evaluation_manifest(
        baseline_run_name, vivit_root,
        baseline_prediction_column,
    )
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
        if run_manifest.get(key) != baseline_manifest.get(key):
            raise RuntimeError(f"两个 run 的 {key} 不一致，禁止直接比较")

    def read_table(run: str, column: str, file_name: str) -> pd.DataFrame:
        """读取对应产物，并按接口约定处理缺失文件。"""
        path = (
            vivit_root / "quality" / "runs" / run
            / evaluation_directory_name(column) / file_name
        )
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path).copy()
        frame.insert(0, "run_name", run)
        frame.insert(1, "prediction_column", column)
        return frame

    runs = (
        (baseline_run_name, baseline_prediction_column),
        (run_name, prediction_column),
    )
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


def compare_prediction_signal_pearson(
    run_name: str,
    baseline_run_name: str | None,
    vivit_data_root: str | Path | None = None,
    prediction_column: str = FACTOR_COLUMN,
    baseline_prediction_column: str = FACTOR_COLUMN,
) -> dict[str, pd.DataFrame]:
    """按完整业务键配对两个 run，并计算逐日横截面 Pearson。"""

    run_name = validate_run_name(run_name)
    prediction_column = validate_prediction_column(prediction_column)
    if baseline_run_name is None:
        return {}
    baseline_run_name = validate_run_name(baseline_run_name)
    baseline_prediction_column = validate_prediction_column(
        baseline_prediction_column
    )
    if baseline_run_name == run_name and baseline_prediction_column == prediction_column:
        raise ValueError("基准 run 与预测列不能与待评价 run 相同")
    vivit_root = Path(
        vivit_data_root or DEFAULT_VIVIT_DATA_ROOT
    ).resolve()
    if not vivit_root.is_dir():
        raise FileNotFoundError(vivit_root)

    def training_manifest(name: str) -> dict[str, object]:
        """读取并校验上游与当前产物的 manifest 签名链。"""
        return read_json(
            vivit_root / "quality" / "runs" / name
            / "a2_training_manifest.json"
        )

    run_manifest = training_manifest(run_name)  # 读取或生成来源清单
    baseline_manifest = training_manifest(baseline_run_name)  # 读取或生成来源清单
    for key in (
        "source_dataset_signature",
        "source_rolling_signature",
        "sequence_signature",
    ):
        if run_manifest.get(key) != baseline_manifest.get(key):
            raise RuntimeError(f"两个 run 的 {key} 不一致，禁止直接比较")
    run_counts = _prediction_counts(run_manifest)
    baseline_counts = _prediction_counts(baseline_manifest)
    if run_counts != baseline_counts:
        raise RuntimeError("两个 run 的预测年度或行数不一致，禁止直接比较")

    keys = ["trade_date", "instrument_id"]

    def read_prediction(
        name: str, year: int, expected_count: int, column: str
    ) -> pd.DataFrame:
        """读取对应产物，并按接口约定处理缺失文件。"""
        path = (
            vivit_root / "predictions" / "runs" / name
            / PREDICTION_PATTERN.format(year=year)
        )
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_parquet(path, columns=[*keys, column])
        if len(frame) != expected_count:
            raise AssertionError(
                f"{name} {year} 预测行数 {len(frame):,} != {expected_count:,}"
            )
        frame["trade_date"] = frame["trade_date"].astype(str)
        frame["instrument_id"] = frame["instrument_id"].astype(str)
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} {year} prediction 业务键重复")
        values = frame[column].to_numpy(np.float64)
        if not np.isfinite(values).all():
            raise ValueError(f"{name} {year} 因子含 NaN 或 Inf")
        return frame

    daily_frames: list[pd.DataFrame] = []
    for year, expected_count in run_counts.items():
        if expected_count == 0:
            continue
        baseline = read_prediction(
            baseline_run_name, year, expected_count, baseline_prediction_column
        ).rename(columns={baseline_prediction_column: "baseline_signal"})
        target = read_prediction(
            run_name, year, expected_count, prediction_column
        ).rename(columns={prediction_column: "target_signal"})
        paired = baseline.merge(  # 按业务键对齐，避免依赖原始行序
            target,
            on=keys,
            how="outer",
            validate="one_to_one",
            indicator=True,
        )
        if not paired["_merge"].eq("both").all():
            mismatch = paired["_merge"].value_counts().to_dict()  # 按股票与交易日业务键对齐
            raise AssertionError(
                f"{year} 两个 prediction 的业务键不一致：{mismatch}"
            )
        paired = paired.drop(columns="_merge")
        paired["baseline_square"] = paired["baseline_signal"].pow(2)
        paired["target_square"] = paired["target_signal"].pow(2)
        paired["cross_product"] = (
            paired["baseline_signal"] * paired["target_signal"]
        )
        stats = paired.groupby("trade_date", sort=True).agg(  # 按既定口径聚合
            observations=("target_signal", "size"),
            baseline_sum=("baseline_signal", "sum"),
            target_sum=("target_signal", "sum"),
            baseline_square_sum=("baseline_square", "sum"),
            target_square_sum=("target_square", "sum"),
            cross_product_sum=("cross_product", "sum"),
        )
        count = stats["observations"].to_numpy(np.float64)
        baseline_variation = (
            count * stats["baseline_square_sum"].to_numpy(np.float64)
            - stats["baseline_sum"].to_numpy(np.float64) ** 2
        )
        target_variation = (
            count * stats["target_square_sum"].to_numpy(np.float64)
            - stats["target_sum"].to_numpy(np.float64) ** 2
        )
        invalid = (
            (count < 2)
            | (baseline_variation <= 0)
            | (target_variation <= 0)
        )
        if invalid.any():
            dates = stats.index[invalid].astype(str).tolist()[:5]
            raise ValueError(
                f"{year} 存在样本不足或常数信号截面：{dates}"
            )
        covariance = (
            count * stats["cross_product_sum"].to_numpy(np.float64)
            - stats["baseline_sum"].to_numpy(np.float64)
            * stats["target_sum"].to_numpy(np.float64)
        )
        pearson = covariance / np.sqrt(
            baseline_variation * target_variation
        )
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

    if not daily_frames:
        raise FileNotFoundError("两个 run 均没有可比较的非空年度预测")
    daily = pd.concat(daily_frames, ignore_index=True)
    annual = daily.groupby("year", as_index=False).agg(  # 按既定口径聚合
        dates=("trade_date", "size"),
        matched_rows=("observations", "sum"),
        mean_daily_pearson=("pearson", "mean"),
        std_daily_pearson=("pearson", "std"),
        median_daily_pearson=("pearson", "median"),
        min_daily_pearson=("pearson", "min"),
        max_daily_pearson=("pearson", "max"),
    )
    annual.insert(0, "baseline_run_name", baseline_run_name)
    annual.insert(0, "baseline_prediction_column", baseline_prediction_column)
    annual.insert(0, "prediction_column", prediction_column)
    annual.insert(0, "run_name", run_name)
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
    return {"daily": daily, "annual": annual, "overall": overall}
def audit_evaluation_inputs(
    run_name: str = DEFAULT_RUN_NAME,
    vivit_data_root: str | Path | None = None,
    reference_data_root: str | Path | None = None,
    backtest_mode: str = DEFAULT_BACKTEST_MODE,
    prediction_column: str = FACTOR_COLUMN,
) -> pd.DataFrame:
    """只读审计 ViViT 预测、签名链、Universe 键和 T+10 日历覆盖。"""

    run_name = validate_run_name(run_name)
    prediction_column = validate_prediction_column(prediction_column)
    mode = resolve_backtest_mode(backtest_mode)
    vivit_root, reference_root = _roots(vivit_data_root, reference_data_root)
    manifests = _manifests(run_name, vivit_root, reference_root)  # 读取或生成来源清单
    counts = _prediction_counts(manifests["training"])
    prediction_dir = vivit_root / "predictions" / "runs" / run_name
    universe_dir = reference_root / "universe"
    records: list[dict[str, object]] = []
    last_factor_date: str | None = None

    for year, expected_count in counts.items():
        prediction_path = prediction_dir / PREDICTION_PATTERN.format(year=year)
        if expected_count == 0:
            if prediction_path.exists():
                raise AssertionError(
                    f"{year} factor_counts=0 但存在预测文件：{prediction_path}"
                )
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
            continue
        if not prediction_path.exists():
            raise FileNotFoundError(prediction_path)
        prediction = pd.read_parquet(
            prediction_path,
            columns=["trade_date", "instrument_id", prediction_column],
        )
        prediction["trade_date"] = prediction["trade_date"].astype(str)
        prediction["instrument_id"] = prediction["instrument_id"].astype(str)
        if len(prediction) != expected_count:
            raise AssertionError(
                f"{year} 预测行数 {len(prediction):,} != {expected_count:,}"
            )
        if prediction.duplicated(["trade_date", "instrument_id"]).any():
            raise ValueError(f"{year} prediction 业务键重复")
        if not prediction["trade_date"].str.startswith(str(year)).all():
            raise ValueError(f"{year} prediction 含跨年度日期")
        if not np.isfinite(
            prediction[prediction_column].to_numpy(np.float64)
        ).all():
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
            mismatch = joined["_merge"].value_counts().to_dict()  # 按股票与交易日业务键对齐
            raise AssertionError(
                f"{year} prediction 与 T 日 Universe 键不一致：{mismatch}"
            )
        if joined["can_enter_t1"].isna().any():
            raise AssertionError(f"{year} can_enter_t1 含缺失值")
        first_date = str(prediction["trade_date"].min())
        year_last_date = str(prediction["trade_date"].max())
        last_factor_date = max(last_factor_date or year_last_date, year_last_date)
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

    if last_factor_date is None:
        raise FileNotFoundError("没有可评价的非空年度预测")
    final_price_date = _calendar_last_date(last_factor_date, str(mode["name"]))  # 按交易日历定位收益起止日
    audit = pd.DataFrame(records)
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
    return audit

def _validated_evaluation_manifest(
    run_name: str, vivit_root: Path,
    prediction_column: str = FACTOR_COLUMN,
) -> dict[str, object]:
    """校验输入形状、业务键、有限性和签名约束；异常时明确失败。"""
    prediction_column = validate_prediction_column(prediction_column)
    path = (
        vivit_root / "quality" / "runs" / run_name
        / evaluation_directory_name(prediction_column) / "factor_evaluation_manifest.json"
    )
    manifest = read_json(path)
    if manifest.get("schema_version") != 2 or "backtest_mode" not in manifest:
        raise RuntimeError(
            f"{run_name} 是旧版评价产物，请用新版 Notebook 03 重新运行评价"
        )
    mode = resolve_backtest_mode(str(manifest["backtest_mode"]))
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
    for key, value in expected.items():
        actual = manifest.get(key, FACTOR_COLUMN if key == "prediction_column" else None)
        if actual != value:
            raise RuntimeError(
                f"{run_name} 评价口径不一致：{key}={actual!r}，预期 {value!r}"
            )
    return manifest
