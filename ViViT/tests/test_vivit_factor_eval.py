from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pandas as pd

VIVIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIVIT_DIR))
import stage08_factor_evaluation as evaluation


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def evaluation_manifest_payload(dataset_signature: str = "d", **overrides) -> dict:
    mode = evaluation.resolve_backtest_mode(evaluation.DEFAULT_BACKTEST_MODE)
    payload = {
        "schema_version": 2,
        "model_family": "ViViT",
        "grouping_policy": evaluation.GROUPING_POLICY,
        "periods": list(evaluation.PERIODS),
        "quantiles": evaluation.QUANTILES,
        "backtest_mode": evaluation.DEFAULT_BACKTEST_MODE,
        "price_start": mode["price_start"],
        "price_start_offset": mode["price_start_offset"],
        "forward_return_formula": "adj_close(T+k) / adj_close(T) - 1",
        "portfolio_scope": mode["portfolio_scope"],
        "execution_mask": mode["execution_mask"],
        "benchmark_column": evaluation.EQUAL_WEIGHT_MARKET_COLUMN,
        "benchmark_scope": evaluation.BENCHMARK_SCOPE,
        "benchmark_weighting": "daily_equal_weight",
        "benchmark_execution_mask": "none",
        "annual_excess_method": "portfolio_geometric_annual_return_minus_benchmark_geometric_annual_return",
        "information_ratio_method": "mean_daily_active_return_div_daily_tracking_error_times_sqrt_252",
        "annual_tracking_error_method": "std_daily_active_return_ddof_1_times_sqrt_252",
        "annualization_days": 252,
        "long_short_gross_exposure": 2.0,
        "dataset_signature": dataset_signature,
        "universe_signature": "u",
        "sequence_signature": "s",
    }
    payload.update(overrides)
    return payload


class SyntheticInputAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        root = Path(self.temporary.name)
        self.vivit_root = root / "vivit"
        self.reference_root = root / "reference"
        self.calendar_path = root / "calendar.parquet"
        self.run_name = "test_run"
        dataset_signature = "dataset"
        rolling_signature = "rolling"
        universe_signature = "universe"
        sequence_signature = "sequence"
        write_json(
            self.vivit_root / "quality" / "runs" / self.run_name / "a2_training_manifest.json",
            {
                "source_dataset_signature": dataset_signature,
                "source_rolling_signature": rolling_signature,
                "sequence_signature": sequence_signature,
                "years": [2018, 2025],
                "factor_counts": {"2018": 2, "2025": 0},
            },
        )
        write_json(
            self.vivit_root / "aligned_sequence_store" / "sequence_manifest.json",
            {
                "signature": sequence_signature,
                "payload": {
                    "source_dataset_signature": dataset_signature,
                    "source_rolling_signature": rolling_signature,
                },
            },
        )
        write_json(
            self.reference_root / "feature_store" / "dataset_manifest.json",
            {
                "signature": dataset_signature,
                "payload": {"universe_signature": universe_signature},
            },
        )
        write_json(
            self.reference_root / "feature_store" / "rolling_split_manifest.json",
            {"signature": rolling_signature},
        )
        write_json(
            self.reference_root / "universe" / "universe_manifest.json",
            {"signature": universe_signature},
        )
        write_json(
            self.reference_root / "labels_t10" / "labels_manifest.json",
            {"signature": "labels"},
        )
        prediction_dir = self.vivit_root / "predictions" / "runs" / self.run_name
        prediction_dir.mkdir(parents=True)
        pd.DataFrame(
            {
                "trade_date": ["20180102", "20180102"],
                "instrument_id": ["SH.000001", "SZ.000002"],
                "prediction_ensemble": [0.1, -0.2],
                "prediction_ensemble_unprojected": [0.3, -0.4],
            }
        ).to_parquet(prediction_dir / "vivit_predictions.2018.parquet", index=False)
        universe_dir = self.reference_root / "universe"
        pd.DataFrame(
            {
                "trade_date": ["20180102", "20180102", "20180102"],
                "instrument_id": ["SH.000001", "SZ.000002", "SH.999999"],
                "feature_valid": [True, True, False],
                "in_universe_t": [True, True, True],
                "can_enter_t1": [True, False, True],
            }
        ).to_parquet(universe_dir / "universe.2018.parquet", index=False)
        business_days = pd.bdate_range("2018-01-02", periods=20)
        pd.DataFrame(
            {
                "trade_days": business_days.strftime("%Y%m%d"),
                "s_info_exchmarket": "SSE",
            }
        ).to_parquet(self.calendar_path, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def audit(self):
        with mock.patch.object(evaluation, "CALENDAR_PATH", self.calendar_path):
            return evaluation.audit_evaluation_inputs(
                self.run_name, self.vivit_root, self.reference_root
            )

    def test_prediction_discovery_empty_2025_and_signature_chain(self):
        audit = self.audit()
        self.assertEqual(audit["year"].tolist(), [2018, 2025])
        self.assertEqual(audit.loc[0, "prediction_rows"], 2)
        self.assertEqual(audit.loc[1, "prediction_rows"], 0)
        self.assertEqual(audit.loc[1, "status"], "empty_prediction_expected")
        self.assertEqual(audit.attrs["dataset_signature"], "dataset")
        self.assertEqual(audit.attrs["final_price_date"], "20180116")

        self.assertEqual(audit.attrs["backtest_mode"], "mode2_t_no_filter")

    def test_nondefault_prediction_column_is_audited(self):
        with mock.patch.object(evaluation, "CALENDAR_PATH", self.calendar_path):
            audit = evaluation.audit_evaluation_inputs(
                self.run_name, self.vivit_root, self.reference_root,
                prediction_column="prediction_ensemble_unprojected",
            )
        self.assertEqual(
            audit.attrs["prediction_column"],
            "prediction_ensemble_unprojected",
        )

    def test_unknown_prediction_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知预测列"):
            evaluation.validate_prediction_column("not_a_prediction")

    def test_mode3_extends_calendar_to_t_plus_11(self):
        with mock.patch.object(evaluation, "CALENDAR_PATH", self.calendar_path):
            audit = evaluation.audit_evaluation_inputs(
                self.run_name, self.vivit_root, self.reference_root,
                backtest_mode="mode3_t1_no_filter",
            )
        self.assertEqual(audit.attrs["final_price_date"], "20180117")
        self.assertEqual(audit.attrs["price_start_offset"], 1)

    def test_universe_key_mismatch_is_rejected(self):
        path = self.reference_root / "universe" / "universe.2018.parquet"
        universe = pd.read_parquet(path)
        universe.loc[0, "instrument_id"] = "SH.123456"
        universe.to_parquet(path, index=False)
        with self.assertRaisesRegex(AssertionError, "Universe 键不一致"):
            self.audit()

    def test_signature_mismatch_is_rejected(self):
        path = (
            self.vivit_root / "quality" / "runs" / self.run_name
            / "a2_training_manifest.json"
        )
        manifest = json.loads(path.read_text())
        manifest["source_dataset_signature"] = "stale"
        write_json(path, manifest)
        with self.assertRaisesRegex(AssertionError, "签名链不一致"):
            self.audit()

    def test_vit_adapter_writes_only_vivit_evaluation_target(self):
        fake = SimpleNamespace(generate_experiment_report=lambda **kwargs: None)

        def fake_run(run_name, data_root, backtest_mode):
            self.assertEqual(backtest_mode, evaluation.DEFAULT_BACKTEST_MODE)
            root = Path(data_root)
            prediction = (
                root / "predictions" / "runs" / run_name
                / "vit_predictions.2018.parquet"
            )
            self.assertTrue(prediction.is_symlink())
            normalized = json.loads(
                (root / "quality" / "runs" / run_name
                 / "a2_training_manifest.json").read_text()
            )
            self.assertEqual(normalized["dataset_signature"], "dataset")
            self.assertEqual(normalized["universe_signature"], "universe")
            manifest_path = (
                root / "quality" / "runs" / run_name / "factor_evaluation"
                / "factor_evaluation_manifest.json"
            )
            write_json(
                manifest_path,
                evaluation_manifest_payload(
                    dataset_signature="dataset",
                    universe_signature="universe",
                    sequence_signature="sequence",
                ),
            )
            fake.generate_experiment_report(run_name=run_name, data_root=root)
            return pd.DataFrame({"item": ["ok"], "value": [True]})

        fake.run_oos_backtest = fake_run
        with (
            mock.patch.object(evaluation, "CALENDAR_PATH", self.calendar_path),
            mock.patch.dict(sys.modules, {"vit_stage08_factor_evaluation": fake}),
            mock.patch.object(
                evaluation, "generate_experiment_report",
                return_value=self.vivit_root / "report.md",
            ) as report_mock,
        ):
            result = evaluation.run_oos_backtest(
                self.run_name, self.vivit_root, self.reference_root
            )
        self.assertTrue(result.loc[0, "value"])
        output = (
            self.vivit_root / "quality" / "runs" / self.run_name
            / "factor_evaluation" / "factor_evaluation_manifest.json"
        )
        manifest = json.loads(output.read_text())
        self.assertEqual(manifest["model_family"], "ViViT")
        self.assertEqual(manifest["prediction_pattern"], evaluation.PREDICTION_PATTERN)
        self.assertEqual(manifest["quantiles"], 5)
        self.assertEqual(report_mock.call_count, 2)


    def test_nondefault_column_uses_temporary_mapping_and_isolated_output(self):
        manifests = evaluation._manifests(
            self.run_name, self.vivit_root, self.reference_root
        )
        with tempfile.TemporaryDirectory() as directory:
            temporary_root = Path(directory)
            output = evaluation._create_vit_evaluation_view(
                temporary_root, self.run_name, self.vivit_root,
                self.reference_root, manifests,
                "prediction_ensemble_unprojected",
            )
            mapped_path = (
                temporary_root / "predictions" / "runs" / self.run_name
                / "vit_predictions.2018.parquet"
            )
            self.assertFalse(mapped_path.is_symlink())
            mapped = pd.read_parquet(mapped_path)
            self.assertEqual(
                mapped.columns.tolist(),
                ["trade_date", "instrument_id", "prediction_ensemble"],
            )
            np.testing.assert_allclose(
                mapped["prediction_ensemble"].to_numpy(), [0.3, -0.4]
            )
            self.assertEqual(output.name, "factor_evaluation_unprojected")

class ReportCompatibilityTests(unittest.TestCase):
    def test_legacy_trade_date_manifest_without_batch_fields_is_supported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            vivit_root = root / "vivit"
            reference_root = root / "reference"
            manifests = {
                "training": {
                    "loss_name": "sigmoid_rank_ic",
                    "train_batch_mode": "trade_date",
                    "validation_metric": "daily_mean_spearman_ic",
                    "torch_compile": True,
                    "compile_mode": "default",
                    "fused_adam": True,
                    "cuda_prefetch": True,
                },
                "universe": {"signature": "universe"},
                "dataset": {"signature": "dataset"},
                "sequence": {"signature": "sequence"},
            }
            with (
                mock.patch.object(
                    evaluation,
                    "_roots",
                    return_value=(vivit_root, reference_root),
                ),
                mock.patch.object(
                    evaluation, "_manifests", return_value=manifests
                ),
            ):
                report = evaluation.generate_experiment_report(
                    "legacy_style_run", vivit_root, reference_root
                )

            text = report.read_text(encoding="utf-8")
            self.assertIn(
                "fixed / inference batch：full_trade_date / full_trade_date",
                text,
            )

class QuantileTests(unittest.TestCase):
    def test_full_cross_section_quantiles_are_balanced(self):
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2020-01-02", "2020-01-03"]), list("abcdefghijk")],
            names=["date", "asset"],
        )
        factor = pd.Series(np.arange(len(index), dtype=float), index=index)
        groups = evaluation.assign_factor_quantiles(factor)
        counts = groups.groupby([pd.Grouper(level="date"), groups]).size().unstack()
        self.assertTrue(((counts.max(axis=1) - counts.min(axis=1)) <= 1).all())
        self.assertEqual(set(groups.unique()), {1, 2, 3, 4, 5})


class ComparisonTests(unittest.TestCase):
    def test_none_baseline_skips_comparison(self):
        self.assertEqual(evaluation.compare_factor_evaluations("run", None), {})

    def test_same_run_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能与待评价 run 相同"):
            evaluation.compare_factor_evaluations("run", "run")

    def test_incompatible_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            reference = root / "reference"
            reference.mkdir()
            common = evaluation_manifest_payload()
            for run, dataset in (("target", "d1"), ("baseline", "d2")):
                path = root / "quality" / "runs" / run / "factor_evaluation"
                write_json(path / "factor_evaluation_manifest.json", {**common, "dataset_signature": dataset})
            with mock.patch.object(evaluation, "DEFAULT_REFERENCE_DATA_ROOT", reference):
                with self.assertRaisesRegex(RuntimeError, "dataset_signature"):
                    evaluation.compare_factor_evaluations("target", "baseline", root)


    def test_old_evaluation_manifest_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run in ("target", "baseline"):
                output = root / "quality" / "runs" / run / "factor_evaluation"
                write_json(output / "factor_evaluation_manifest.json", {
                    "model_family": "ViViT",
                    "grouping_policy": "T_full_cross_section_before_can_enter_t1",
                })
            with self.assertRaisesRegex(RuntimeError, "旧版评价产物"):
                evaluation.compare_factor_evaluations("target", "baseline", root)


    def test_compatible_runs_return_four_comparison_tables(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            common = evaluation_manifest_payload()
            tables = {
                "ic_summary.parquet": pd.DataFrame({"period": ["1D"], "mean_ic": [0.1]}),
                "spread_summary.parquet": pd.DataFrame({"period": ["1D"], "Q5_minus_Q1_return": [0.01]}),
                "pyfolio_performance.parquet": pd.DataFrame({"metric": ["Sharpe ratio"], "Q5 long-only": [1.0]}),
                "relative_performance.parquet": pd.DataFrame({"portfolio": ["Q5 long-only"], "information_ratio": [1.0]}),
            }
            for run in ("target", "baseline"):
                output = root / "quality" / "runs" / run / "factor_evaluation"
                write_json(output / "factor_evaluation_manifest.json", common)
                for name, frame in tables.items():
                    frame.to_parquet(output / name, index=False)
            compared = evaluation.compare_factor_evaluations(
                "target", "baseline", root
            )
            self.assertEqual(set(compared), {"ic", "spread", "pyfolio", "relative"})
            for frame in compared.values():
                self.assertEqual(frame["run_name"].tolist(), ["baseline", "target"])
                self.assertEqual(
                    frame["prediction_column"].tolist(),
                    ["prediction_ensemble", "prediction_ensemble"],
                )


class PredictionSignalPearsonTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.baseline = pd.DataFrame(
            {
                "trade_date": ["20200102"] * 3 + ["20200103"] * 3,
                "instrument_id": ["a", "b", "c"] * 2,
                "prediction_ensemble": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
            }
        )
        self.target = pd.DataFrame(
            {
                "trade_date": ["20200102"] * 3 + ["20200103"] * 3,
                "instrument_id": ["a", "b", "c"] * 2,
                "prediction_ensemble": [3.0, 2.0, 1.0, 1.0, 2.0, 3.0],
                "prediction_ensemble_unprojected": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            }
        )
        self.write_run("baseline", self.baseline)
        self.write_run("target", self.target.iloc[::-1].reset_index(drop=True))

    def tearDown(self):
        self.temporary.cleanup()

    def write_run(self, name: str, prediction: pd.DataFrame) -> None:
        write_json(
            self.root / "quality" / "runs" / name
            / "a2_training_manifest.json",
            {
                "source_dataset_signature": "dataset",
                "source_rolling_signature": "rolling",
                "sequence_signature": "sequence",
                "years": [2020, 2025],
                "factor_counts": {"2020": len(prediction), "2025": 0},
            },
        )
        path = (
            self.root / "predictions" / "runs" / name
            / "vivit_predictions.2020.parquet"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        prediction.to_parquet(path, index=False)

    def test_daily_cross_sectional_pearson_and_summary(self):
        result = evaluation.compare_prediction_signal_pearson(
            "target", "baseline", self.root
        )
        self.assertEqual(set(result), {"daily", "annual", "overall"})
        self.assertEqual(
            result["daily"]["trade_date"].tolist(),
            ["20200102", "20200103"],
        )
        np.testing.assert_allclose(
            result["daily"]["pearson"].to_numpy(), [-1.0, 1.0]
        )
        self.assertEqual(result["daily"]["observations"].tolist(), [3, 3])
        self.assertAlmostEqual(
            float(result["annual"].loc[0, "mean_daily_pearson"]), 0.0
        )
        self.assertAlmostEqual(
            float(result["overall"].loc[0, "mean_daily_pearson"]), 0.0
        )
        self.assertEqual(int(result["overall"].loc[0, "matched_rows"]), 6)

    def test_same_run_different_columns_can_be_compared(self):
        result = evaluation.compare_prediction_signal_pearson(
            "target", "target", self.root,
            prediction_column="prediction_ensemble_unprojected",
            baseline_prediction_column="prediction_ensemble",
        )
        self.assertEqual(
            result["overall"].loc[0, "prediction_column"],
            "prediction_ensemble_unprojected",
        )
        self.assertEqual(
            result["overall"].loc[0, "baseline_prediction_column"],
            "prediction_ensemble",
        )
        self.assertEqual(
            result["daily"]["prediction_column"].unique().tolist(),
            ["prediction_ensemble_unprojected"],
        )

    def test_key_mismatch_is_rejected(self):
        changed = self.target.copy()
        changed.loc[0, "instrument_id"] = "different"
        self.write_run("target", changed)
        with self.assertRaisesRegex(AssertionError, "业务键不一致"):
            evaluation.compare_prediction_signal_pearson(
                "target", "baseline", self.root
            )

    def test_constant_cross_section_is_rejected(self):
        changed = self.target.copy()
        changed.loc[changed["trade_date"] == "20200102", "prediction_ensemble"] = 1.0
        self.write_run("target", changed)
        with self.assertRaisesRegex(ValueError, "常数信号截面"):
            evaluation.compare_prediction_signal_pearson(
                "target", "baseline", self.root
            )

    def test_none_baseline_skips_and_same_run_is_rejected(self):
        self.assertEqual(
            evaluation.compare_prediction_signal_pearson(
                "target", None, self.root
            ),
            {},
        )
        with self.assertRaisesRegex(ValueError, "不能与待评价 run 相同"):
            evaluation.compare_prediction_signal_pearson(
                "target", "target", self.root
            )


class RealInputPreflightTests(unittest.TestCase):
    def test_current_run_inputs(self):
        path = (
            evaluation.DEFAULT_VIVIT_DATA_ROOT / "quality" / "runs"
            / evaluation.DEFAULT_RUN_NAME / "a2_training_manifest.json"
        )
        if not path.exists():
            self.skipTest("当前 ViViT run 尚未完成")
        audit = evaluation.audit_evaluation_inputs()
        self.assertEqual(len(audit), 8)
        self.assertEqual(int(audit["prediction_rows"].gt(0).sum()), 7)
        self.assertEqual(
            int(audit.loc[audit.year == 2025, "prediction_rows"].item()), 0
        )
        self.assertTrue(audit.loc[audit.prediction_rows > 0, "status"].eq("passed").all())


if __name__ == "__main__":
    unittest.main()
