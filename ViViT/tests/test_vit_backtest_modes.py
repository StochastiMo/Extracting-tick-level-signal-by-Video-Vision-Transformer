from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

VIT_DIR = Path(__file__).resolve().parents[2] / "ViT"
sys.path.insert(0, str(VIT_DIR))
import vit_stage08_factor_evaluation as pipeline


class BacktestModeTests(unittest.TestCase):
    def test_default_and_all_mode_definitions(self):
        self.assertEqual(pipeline.DEFAULT_BACKTEST_MODE, "mode2_t_no_filter")
        expected = {
            "mode1_t_with_t1_filter": (0, True),
            "mode2_t_no_filter": (0, False),
            "mode3_t1_no_filter": (1, False),
        }
        self.assertEqual(set(pipeline.BACKTEST_MODES), set(expected))
        for name, (offset, filtered) in expected.items():
            mode = pipeline.resolve_backtest_mode(name)
            self.assertEqual(mode["price_start_offset"], offset)
            self.assertEqual(mode["apply_can_enter_t1"], filtered)

    def test_invalid_mode_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "未知回测模式"):
            pipeline.resolve_backtest_mode("future_leaking_mode")

    def test_backtest_cli_defaults_to_mode2(self):
        with mock.patch.object(sys, "argv", ["vit_stage08_factor_evaluation.py", "backtest"]):
            args = pipeline.parse_args()
        self.assertEqual(args.backtest_mode, "mode2_t_no_filter")

    def test_price_views_match_1_5_10_day_formulas(self):
        dates = pd.bdate_range("2024-01-02", periods=12)
        prices = pd.DataFrame(100.0 + np.arange(12), index=dates, columns=["asset"])
        for mode_name, start in (
            ("mode1_t_with_t1_filter", 0),
            ("mode2_t_no_filter", 0),
            ("mode3_t1_no_filter", 1),
        ):
            view = pipeline.build_return_price_view(prices, mode_name)
            self.assertTrue(view.index.equals(prices.index))
            for period in (1, 5, 10):
                actual = view.iloc[period, 0] / view.iloc[0, 0] - 1
                expected = prices.iloc[start + period, 0] / prices.iloc[start, 0] - 1
                self.assertAlmostEqual(actual, expected)
        self.assertEqual(pipeline.price_point_label(11), "T+11")
        self.assertIn("T+1+k", pipeline.forward_return_formula(1))

    def test_only_mode1_filters_portfolio_and_benchmark_is_always_full(self):
        index = pd.MultiIndex.from_product(
            [pd.to_datetime(["2024-01-02", "2024-01-03"]), ["a", "b"]],
            names=["date", "asset"],
        )
        factor_data = pd.DataFrame(
            {"1D": [0.01, 0.03, -0.02, 0.04], "factor_quantile": [1, 5, 1, 5]},
            index=index,
        )
        mask = pd.Series([True, False, False, True], index=index)
        mode1 = pipeline.select_execution_factor_data(
            factor_data, mask, "mode1_t_with_t1_filter"
        )
        self.assertEqual(mode1.index.tolist(), [index[0], index[3]])
        for name in ("mode2_t_no_filter", "mode3_t1_no_filter"):
            selected = pipeline.select_execution_factor_data(factor_data, mask, name)
            self.assertTrue(selected.index.equals(factor_data.index))
            self.assertEqual(selected["factor_quantile"].tolist(), [1, 5, 1, 5])
        benchmark = pipeline.compute_equal_weight_market_returns(factor_data)
        np.testing.assert_allclose(benchmark.to_numpy(), [0.02, 0.01])
        self.assertEqual(benchmark.name, pipeline.EQUAL_WEIGHT_MARKET_COLUMN)


class RelativePerformanceTests(unittest.TestCase):
    def test_relative_metrics_follow_documented_formulas(self):
        frame = pd.DataFrame(
            {
                "Q5 long-only": [0.02, -0.01, 0.03, 0.00],
                "Q5-Q1 long-short": [0.03, 0.01, -0.01, 0.02],
                pipeline.EQUAL_WEIGHT_MARKET_COLUMN: [0.01, 0.00, 0.01, -0.01],
            }
        )
        result = pipeline.compute_relative_performance(frame).set_index("portfolio")
        benchmark = frame[pipeline.EQUAL_WEIGHT_MARKET_COLUMN].to_numpy()
        benchmark_annual = np.prod(1 + benchmark) ** (252 / len(frame)) - 1
        for portfolio in pipeline.RELATIVE_PORTFOLIOS:
            returns = frame[portfolio].to_numpy()
            active = returns - benchmark
            annual = np.prod(1 + returns) ** (252 / len(frame)) - 1
            active_std = active.std(ddof=1)
            row = result.loc[portfolio]
            self.assertEqual(row["observations"], len(frame))
            self.assertAlmostEqual(row["portfolio_annual_return"], annual)
            self.assertAlmostEqual(row["benchmark_annual_return"], benchmark_annual)
            self.assertAlmostEqual(row["annual_excess_return"], annual - benchmark_annual)
            self.assertAlmostEqual(row["annual_tracking_error"], active_std * np.sqrt(252))
            self.assertAlmostEqual(
                row["information_ratio"], active.mean() / active_std * np.sqrt(252)
            )

    def test_zero_active_variance_is_rejected(self):
        benchmark = np.array([0.01, -0.02, 0.03])
        frame = pd.DataFrame(
            {
                "Q5 long-only": benchmark + 0.001,
                "Q5-Q1 long-short": benchmark + np.array([0.0, 0.01, -0.01]),
                pipeline.EQUAL_WEIGHT_MARKET_COLUMN: benchmark,
            }
        )
        with self.assertRaisesRegex(ValueError, "方差为零"):
            pipeline.compute_relative_performance(frame)


if __name__ == "__main__":
    unittest.main()
