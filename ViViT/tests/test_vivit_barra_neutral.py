from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

VIVIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIVIT_DIR))
import stage09_barra_postprocess as neutral


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def synthetic_cross_section(
    rows: int = 60, seed: int = 7
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    styles = rng.normal(size=(rows, len(neutral.STYLE_COLUMNS)))
    reversal = rng.normal(scale=0.05, size=rows)
    reversal_z = (reversal - reversal.mean()) / reversal.std(ddof=0)
    regressors = np.column_stack([styles, reversal_z])
    industries = np.array(
        [f"CNE5S_IND{index % 3}" for index in range(rows)]
    )
    coefficients = np.linspace(-0.4, 0.5, len(neutral.REGRESSION_COLUMNS))
    industry_effect = np.array(
        [{"CNE5S_IND0": -0.3, "CNE5S_IND1": 0.1, "CNE5S_IND2": 0.4}[value]
         for value in industries]
    )
    noise = rng.normal(scale=0.02, size=rows)
    factor = regressors @ coefficients + industry_effect + noise
    frame = pd.DataFrame(styles, columns=neutral.STYLE_COLUMNS)
    frame[neutral.REVERSAL_COLUMN] = reversal
    frame[neutral.REVERSAL_VALID_COLUMN] = True
    frame[neutral.INDUSTRY_COLUMN] = industries
    frame[neutral.SOURCE_FACTOR_COLUMN] = factor
    frame["barra_matched"] = True
    return frame, noise


def write_price_sources(
    calendar_path: Path,
    eod_path: Path,
    instrument_ids: list[str],
    reversals: np.ndarray,
    *,
    signal_date: str = "20180131",
    missing_lag_indices: tuple[int, ...] = (),
) -> tuple[str, str]:
    dates = pd.bdate_range(end=pd.Timestamp(signal_date), periods=21)
    calendar_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "trade_days": dates.strftime("%Y%m%d"),
            "s_info_exchmarket": "SSE",
        }
    ).to_parquet(calendar_path, index=False)
    lag_date = dates[0].strftime("%Y%m%d")
    current_date = dates[-1].strftime("%Y%m%d")
    records = []
    missing = set(missing_lag_indices)
    for index, (instrument_id, reversal) in enumerate(
        zip(instrument_ids, reversals)
    ):
        exchange, number = instrument_id.split(".")
        windcode = f"{number}.{exchange}"
        if index not in missing:
            records.append(
                {
                    "s_info_windcode": windcode,
                    "trade_dt": lag_date,
                    "s_dq_adjclose": 100.0,
                }
            )
        records.append(
            {
                "s_info_windcode": windcode,
                "trade_dt": current_date,
                "s_dq_adjclose": 100.0 * (1.0 - float(reversal)),
            }
        )
    pd.DataFrame(records).to_parquet(eod_path, index=False)
    return current_date, lag_date


class IdentifierTests(unittest.TestCase):
    def test_instrument_id_mapping(self):
        actual = neutral.instrument_ids_to_assetids(
            pd.Series(["SH.600000", "SZ.000001", "BJ.430001"])
        )
        self.assertEqual(
            actual.tolist(), ["CN600000", "CN000001", "CN430001"]
        )

    def test_invalid_instrument_id_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "格式非法"):
            neutral.instrument_ids_to_assetids(
                pd.Series(["600000.SH", "HK.000001"])
            )

    def test_invalid_source_factor_column_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知源预测列"):
            neutral._validate_source_factor_column("unknown_score")


class ReversalTests(unittest.TestCase):
    def test_exact_t_minus_20_formula_and_no_forward_fill(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            calendar = root / "calendar.parquet"
            eod = root / "eod.parquet"
            signal_date, lag_date = write_price_sources(
                calendar, eod, ["SH.600000"], np.array([0.2])
            )
            prediction = pd.DataFrame(
                {
                    "trade_date": [signal_date],
                    "instrument_id": ["SH.600000"],
                    "prediction_ensemble": [0.1],
                }
            )
            attached = neutral._attach_reversal_20d(
                prediction, eod, neutral._load_sse_calendar(calendar), 2018
            )
            self.assertAlmostEqual(
                float(attached[neutral.REVERSAL_COLUMN].iloc[0]), 0.2
            )
            self.assertTrue(attached[neutral.REVERSAL_VALID_COLUMN].iloc[0])

            prices = pd.read_parquet(eod)
            prices = prices.loc[prices["trade_dt"].ne(lag_date)].copy()
            prices = pd.concat(
                [
                    prices,
                    pd.DataFrame(
                        {
                            "s_info_windcode": ["600000.SH"],
                            "trade_dt": [
                                (pd.Timestamp(lag_date) - pd.Timedelta(days=1))
                                .strftime("%Y%m%d")
                            ],
                            "s_dq_adjclose": [100.0],
                        }
                    ),
                ],
                ignore_index=True,
            )
            prices.to_parquet(eod, index=False)
            missing = neutral._attach_reversal_20d(
                prediction, eod, neutral._load_sse_calendar(calendar), 2018
            )
            self.assertFalse(
                missing[neutral.REVERSAL_VALID_COLUMN].iloc[0]
            )
            self.assertTrue(
                np.isnan(missing[neutral.REVERSAL_COLUMN].iloc[0])
            )

    def test_daily_zscore_and_missing_fill_zero(self):
        frame = pd.DataFrame(
            {
                neutral.REVERSAL_COLUMN: [1.0, 2.0, 3.0, np.nan],
                neutral.REVERSAL_VALID_COLUMN: [True, True, True, False],
            }
        )
        actual = neutral._standardized_reversal(frame)
        self.assertAlmostEqual(float(actual[:3].mean()), 0.0)
        self.assertAlmostEqual(float(actual[:3].std(ddof=0)), 1.0)
        self.assertEqual(float(actual[-1]), 0.0)


class CorrelationAndRegressionTests(unittest.TestCase):
    def test_daily_pearson_uses_exactly_ten_regressors(self):
        frame, _ = synthetic_cross_section()
        frame["trade_date"] = "20200102"
        frame["cne5s_resvol"] = np.arange(len(frame), dtype=float)
        frame.loc[frame.index[-1], neutral.REVERSAL_VALID_COLUMN] = False
        records = neutral._daily_correlations(frame, 2020)
        names = {record["factor_name"] for record in records}
        self.assertEqual(names, set(neutral.CORRELATION_COLUMNS))
        self.assertEqual({record["observations"] for record in records}, {59})
        self.assertNotIn("cne5s_resvol", names)
        self.assertNotIn(neutral.INDUSTRY_COLUMN, names)

    def test_correlation_summary_adds_newey_west_corrected_t_stat(self):
        rng = np.random.default_rng(2026)
        innovations = rng.normal(size=400)
        autocorrelated = np.empty(400, dtype=np.float64)
        autocorrelated[0] = innovations[0]
        for index in range(1, len(autocorrelated)):
            autocorrelated[index] = (
                0.9 * autocorrelated[index - 1] + innovations[index]
            )
        pearson = 0.1 + 0.05 * autocorrelated / autocorrelated.std(ddof=0)
        daily = pd.DataFrame(
            {
                "year": 2020,
                "trade_date": pd.date_range(
                    "2020-01-01", periods=len(pearson), freq="D"
                ).strftime("%Y%m%d"),
                "factor_name": neutral.STYLE_COLUMNS[0],
                "factor_label": neutral.STYLE_LABELS[
                    neutral.STYLE_COLUMNS[0]
                ],
                "pearson": pearson,
                "observations": 100,
            }
        )

        annual, overall = neutral._correlation_summaries(daily)

        corrected_column = "t_stat_newey_west_lag20"
        self.assertEqual(annual.columns[-1], corrected_column)
        self.assertEqual(overall.columns[-1], corrected_column)
        self.assertTrue(np.isfinite(overall[corrected_column].iloc[0]))
        self.assertLess(
            abs(float(overall[corrected_column].iloc[0])),
            abs(float(overall["t_stat"].iloc[0])),
        )

    def test_daily_matrix_has_121_symmetric_cells_and_unit_diagonal(self):
        frame, _ = synthetic_cross_section()
        frame["trade_date"] = "20200102"
        records = neutral._daily_correlation_matrix(frame, 2020)
        self.assertEqual(len(records), 11 * 11)
        daily = pd.DataFrame(records)
        self.assertEqual(set(daily["row_name"]), set(neutral.MATRIX_COLUMNS))
        self.assertEqual(
            set(daily["column_name"]), set(neutral.MATRIX_COLUMNS)
        )
        summary = neutral._correlation_matrix_summary(daily)
        self.assertEqual(len(summary), 121)
        matrix = (
            summary.pivot(
                index="row_name",
                columns="column_name",
                values="mean_pearson",
            )
            .loc[list(neutral.MATRIX_COLUMNS), list(neutral.MATRIX_COLUMNS)]
            .to_numpy()
        )
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12)
        np.testing.assert_allclose(np.diag(matrix), 1.0, atol=1e-12)

    def test_industry_fixed_effect_residual_is_orthogonal(self):
        frame, _ = synthetic_cross_section()
        residual, diagnostics, correlations = (
            neutral._neutralize_cross_section(frame)
        )
        self.assertEqual(
            diagnostics["matrix_rank"], len(neutral.REGRESSION_COLUMNS)
        )
        self.assertLess(
            diagnostics["max_abs_style_residual_corr"], 1e-10
        )
        self.assertLess(
            diagnostics["max_abs_industry_residual_mean"], 1e-10
        )
        self.assertEqual(
            {item["factor_name"] for item in correlations},
            set(neutral.REGRESSION_COLUMNS),
        )
        self.assertTrue(np.isfinite(residual).all())
        means = pd.Series(residual).groupby(
            frame[neutral.INDUSTRY_COLUMN]
        ).mean()
        self.assertLess(float(means.abs().max()), 1e-10)

    def test_missing_reversal_and_unknown_industry_keep_all_rows(self):
        frame, _ = synthetic_cross_section()
        missing_barra = np.array([False] * (len(frame) - 3) + [True] * 3)
        frame.loc[missing_barra, neutral.STYLE_COLUMNS] = np.nan
        frame.loc[missing_barra, neutral.INDUSTRY_COLUMN] = None
        frame.loc[missing_barra, "barra_matched"] = False
        missing_reversal = np.array([True, True] + [False] * (len(frame) - 2))
        frame.loc[missing_reversal, neutral.REVERSAL_COLUMN] = np.nan
        frame.loc[missing_reversal, neutral.REVERSAL_VALID_COLUMN] = False
        residual, diagnostics, _ = neutral._neutralize_cross_section(frame)
        self.assertEqual(len(residual), len(frame))
        self.assertEqual(diagnostics["unmatched_rows"], 3)
        self.assertEqual(diagnostics["reversal_missing_rows"], 2)
        self.assertAlmostEqual(
            float(residual[missing_barra].mean()), 0.0, places=12
        )
        self.assertTrue(np.isfinite(residual).all())

    def test_rank_deficient_styles_are_rejected(self):
        frame, _ = synthetic_cross_section()
        frame["cne5s_sizenl"] = frame["cne5s_size"]
        with self.assertRaisesRegex(np.linalg.LinAlgError, "矩阵秩"):
            neutral._neutralize_cross_section(frame)


class InputAuditTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.vivit_root = self.root / "vivit_data"
        self.run_name = "source_run"
        self.cne5s_path = self.root / "CNE5S.parquet"
        self.eod_path = self.root / "eod.parquet"
        self.calendar_path = self.root / "calendar.parquet"
        self.signal_date, _ = write_price_sources(
            self.calendar_path, self.eod_path,
            ["SH.600000"], np.array([0.1]),
        )
        write_json(
            self.vivit_root / "quality" / "runs" / self.run_name
            / "a2_training_manifest.json",
            {
                "run_name": self.run_name,
                "years": [2018, 2025],
                "factor_counts": {"2018": 1, "2025": 0},
                "source_dataset_signature": "dataset",
                "source_rolling_signature": "rolling",
                "sequence_signature": "sequence",
            },
        )
        prediction_path = (
            self.vivit_root / "predictions" / "runs" / self.run_name
            / "vivit_predictions.2018.parquet"
        )
        prediction_path.parent.mkdir(parents=True)
        pd.DataFrame(
            {
                "trade_date": [self.signal_date],
                "instrument_id": ["SH.600000"],
                "prediction_ensemble": [0.2],
            }
        ).to_parquet(prediction_path, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def write_cne5s(self, dates: list[str], duplicate: bool = False) -> None:
        rows = len(dates)
        frame = pd.DataFrame(
            {
                "datadate": pd.to_datetime(dates),
                "assetid": ["CN600000"] * rows,
                neutral.INDUSTRY_COLUMN: ["CNE5S_BANKS"] * rows,
            }
        )
        for index, column in enumerate(neutral.STYLE_COLUMNS):
            frame[column] = np.arange(rows, dtype=float) + index
        if duplicate:
            frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
        frame.to_parquet(self.cne5s_path, index=False)

    def audit(self, minimum_match_rate: float = 0.0):
        return neutral.audit_barra_inputs(
            self.run_name, self.cne5s_path, self.vivit_root,
            minimum_match_rate=minimum_match_rate,
            eod_path=self.eod_path, calendar_path=self.calendar_path,
        )

    def test_exact_cne5s_date_and_reversal_coverage(self):
        self.write_cne5s(["2018-01-30"])
        audit = self.audit()
        row = audit.loc[audit["year"].eq(2018)].iloc[0]
        self.assertEqual(int(row["matched_rows"]), 0)
        self.assertEqual(int(row["unmatched_rows"]), 1)
        self.assertEqual(int(row["reversal_valid_rows"]), 1)
        self.assertEqual(
            audit.attrs["regression_columns"],
            list(neutral.REGRESSION_COLUMNS),
        )

    def test_duplicate_cne5s_key_is_rejected(self):
        self.write_cne5s([self.signal_date], duplicate=True)
        with self.assertRaisesRegex(ValueError, "业务键重复"):
            self.audit()

    def test_invalid_cne5s_assetid_is_rejected(self):
        self.write_cne5s([self.signal_date])
        frame = pd.read_parquet(self.cne5s_path)
        frame.loc[0, "assetid"] = "600000"
        frame.to_parquet(self.cne5s_path, index=False)
        with self.assertRaisesRegex(ValueError, "assetid 格式非法"):
            self.audit()

    def test_nonfinite_cne5s_exposure_is_rejected(self):
        self.write_cne5s([self.signal_date])
        frame = pd.read_parquet(self.cne5s_path)
        frame.loc[0, neutral.STYLE_COLUMNS[0]] = np.nan
        frame.to_parquet(self.cne5s_path, index=False)
        with self.assertRaisesRegex(ValueError, "NaN 或 Inf"):
            self.audit()


class DerivedRunTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.vivit_root = self.root / "vivit_data"
        self.source_run = "source"
        self.residual_run = "source_barra_residual"
        self.cne5s_path = self.root / "CNE5S.parquet"
        self.eod_path = self.root / "eod.parquet"
        self.calendar_path = self.root / "calendar.parquet"
        rows = 60
        cross_section, _ = synthetic_cross_section(rows=rows, seed=13)
        self.cross_section = cross_section
        self.instrument_ids = [
            f"SH.{100000 + index:06d}" for index in range(rows)
        ]
        self.signal_date, _ = write_price_sources(
            self.calendar_path, self.eod_path, self.instrument_ids,
            cross_section[neutral.REVERSAL_COLUMN].to_numpy(),
            missing_lag_indices=(0,),
        )
        predictions = pd.DataFrame(
            {
                "sample_index": np.arange(rows, dtype=np.int64),
                "trade_date": [self.signal_date] * rows,
                "instrument_id": self.instrument_ids,
                "prediction_ensemble": cross_section[
                    neutral.SOURCE_FACTOR_COLUMN
                ].to_numpy(np.float32),
                "forecast_year": [2018] * rows,
                "factor_role": ["ensemble_zscore_mean"] * rows,
            }
        )
        predictions["prediction_ensemble_unprojected"] = (
            -predictions["prediction_ensemble"]
        )
        self.source_path = (
            self.vivit_root / "predictions" / "runs" / self.source_run
            / "vivit_predictions.2018.parquet"
        )
        self.source_path.parent.mkdir(parents=True)
        predictions.to_parquet(self.source_path, index=False)
        write_json(
            self.vivit_root / "quality" / "runs" / self.source_run
            / "a2_training_manifest.json",
            {
                "run_name": self.source_run,
                "years": [2018, 2025],
                "factor_counts": {"2018": rows, "2025": 0},
                "source_dataset_signature": "dataset",
                "source_rolling_signature": "rolling",
                "sequence_signature": "sequence",
                "loss_name": "sigmoid_rank_ic",
                "train_batch_mode": "trade_date",
                "validation_metric": "daily_mean_spearman_ic",
                "fixed_batch_size": 5000,
                "inference_batch_size": 5000,
                "torch_compile": False,
                "fused_adam": False,
                "cuda_prefetch": False,
            },
        )
        barra_rows = rows - 1
        barra = pd.DataFrame(
            {
                "datadate": pd.to_datetime([self.signal_date] * barra_rows),
                "assetid": [
                    f"CN{100000 + index:06d}"
                    for index in range(barra_rows)
                ],
                neutral.INDUSTRY_COLUMN: cross_section.loc[
                    : barra_rows - 1, neutral.INDUSTRY_COLUMN
                ].to_numpy(),
            }
        )
        for column in neutral.STYLE_COLUMNS:
            barra[column] = cross_section.loc[
                : barra_rows - 1, column
            ].to_numpy()
        barra.to_parquet(self.cne5s_path, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def build(self, *, overwrite_existing: bool = False):
        return neutral.build_barra_residual_run(
            self.source_run, self.residual_run, self.cne5s_path,
            self.vivit_root, minimum_match_rate=0.0,
            eod_path=self.eod_path, calendar_path=self.calendar_path,
            overwrite_existing=overwrite_existing,
        )

    def test_read_only_analysis_precedes_residual_generation(self):
        original_bytes = self.source_path.read_bytes()
        prediction_root = (
            self.vivit_root / "predictions" / "runs" / self.residual_run
        )
        quality_root = (
            self.vivit_root / "quality" / "runs" / self.residual_run
        )

        result = neutral.analyze_barra_exposures(
            self.source_run, self.cne5s_path, self.vivit_root,
            minimum_match_rate=0.0,
            eod_path=self.eod_path, calendar_path=self.calendar_path,
        )

        self.assertFalse(result["writes_performed"])
        self.assertFalse(prediction_root.exists())
        self.assertFalse(quality_root.exists())
        self.assertEqual(self.source_path.read_bytes(), original_bytes)
        self.assertEqual(
            set(result["correlation_summary"]["factor_name"]),
            set(neutral.CORRELATION_COLUMNS),
        )
        self.assertEqual(len(result["correlation_matrix_summary"]), 121)
        self.assertEqual(
            int(
                result["input_match_audit"]
                .loc[lambda frame: frame["year"].eq(2018), "prediction_rows"]
                .iloc[0]
            ),
            len(self.cross_section),
        )

    def test_derived_run_is_complete_ten_factor_and_idempotent(self):
        original_bytes = self.source_path.read_bytes()
        result = self.build()
        self.assertFalse(result["reused"])
        self.assertFalse(result["overwritten"])
        output_path = (
            self.vivit_root / "predictions" / "runs" / self.residual_run
            / "vivit_predictions.2018.parquet"
        )
        output = pd.read_parquet(output_path)
        self.assertEqual(len(output), 60)
        self.assertEqual(
            output["instrument_id"].tolist(),
            pd.read_parquet(self.source_path)["instrument_id"].tolist(),
        )
        np.testing.assert_array_equal(
            output["prediction_ensemble_raw"].to_numpy(),
            output["prediction_ensemble"].to_numpy(),
        )
        self.assertEqual(int((~output["barra_matched"]).sum()), 1)
        self.assertEqual(
            int((~output[neutral.REVERSAL_VALID_COLUMN]).sum()), 1
        )
        self.assertFalse(
            (output_path.parent / "vivit_predictions.2025.parquet").exists()
        )
        self.assertEqual(self.source_path.read_bytes(), original_bytes)
        self.assertEqual(
            set(result["correlation_summary"]["factor_name"]),
            set(neutral.CORRELATION_COLUMNS),
        )
        self.assertEqual(len(result["correlation_matrix_summary"]), 121)
        self.assertEqual(
            set(result["correlation_matrix_summary"]["row_name"]),
            set(neutral.MATRIX_COLUMNS),
        )
        self.assertTrue(result["heatmap_path"].exists())
        self.assertTrue(result["reversal_timeseries_path"].exists())
        manifest = json.loads(result["manifest_path"].read_text())
        self.assertEqual(manifest["schema_version"], 5)
        self.assertEqual(manifest["matrix_columns"], list(neutral.MATRIX_COLUMNS))
        self.assertEqual(
            manifest["correlation_inference"]["newey_west_max_lag"], 20,
        )
        self.assertEqual(
            manifest["regression_columns"], list(neutral.REGRESSION_COLUMNS)
        )
        derived_training = json.loads(
            (self.vivit_root / "quality" / "runs" / self.residual_run
             / "a2_training_manifest.json").read_text()
        )
        self.assertEqual(
            derived_training["factor_role"],
            "barra_reversal20_residual",
        )
        reused = self.build()
        self.assertTrue(reused["reused"])

    def test_selected_unprojected_column_drives_analysis_and_residual(self):
        source = pd.read_parquet(self.source_path)
        analysis = neutral.analyze_barra_exposures(
            self.source_run,
            self.cne5s_path,
            self.vivit_root,
            minimum_match_rate=0.0,
            eod_path=self.eod_path,
            calendar_path=self.calendar_path,
            source_factor_column="prediction_ensemble_unprojected",
        )
        self.assertEqual(
            analysis["source_factor_column"],
            "prediction_ensemble_unprojected",
        )

        residual_run = "source_unprojected_barra_residual"
        result = neutral.build_barra_residual_run(
            self.source_run,
            residual_run,
            self.cne5s_path,
            self.vivit_root,
            minimum_match_rate=0.0,
            eod_path=self.eod_path,
            calendar_path=self.calendar_path,
            source_factor_column="prediction_ensemble_unprojected",
        )
        output = pd.read_parquet(
            self.vivit_root / "predictions" / "runs" / residual_run
            / "vivit_predictions.2018.parquet"
        )
        np.testing.assert_array_equal(
            output["source_prediction_ensemble"].to_numpy(),
            source["prediction_ensemble_unprojected"].to_numpy(),
        )
        manifest = json.loads(result["manifest_path"].read_text())
        self.assertEqual(
            manifest["source_factor_column"],
            "prediction_ensemble_unprojected",
        )

    def test_overwrite_guard_and_stale_evaluation_removal(self):
        original_bytes = self.source_path.read_bytes()
        self.build()
        stale = (
            self.vivit_root / "quality" / "runs" / self.residual_run
            / "factor_evaluation" / "stale.txt"
        )
        stale.parent.mkdir(parents=True)
        stale.write_text("old evaluation", encoding="utf-8")
        stale_report = stale.parents[1] / "EXPERIMENT_REPORT.md"
        stale_report.write_text("old report", encoding="utf-8")
        prices = pd.read_parquet(self.eod_path)
        current = prices["trade_dt"].eq(self.signal_date)
        prices.loc[current, "s_dq_adjclose"] *= 1.0001
        prices.to_parquet(self.eod_path, index=False)
        with self.assertRaisesRegex(RuntimeError, "overwrite_existing=True"):
            self.build()
        replaced = self.build(overwrite_existing=True)
        self.assertTrue(replaced["overwritten"])
        self.assertFalse(stale.exists())
        self.assertFalse(stale_report.exists())
        self.assertEqual(self.source_path.read_bytes(), original_bytes)
        tombstones = list(
            (self.vivit_root / "quality" / "runs").glob(
                ".source_barra_residual.replacing.*"
            )
        )
        self.assertEqual(tombstones, [])


if __name__ == "__main__":
    unittest.main()
