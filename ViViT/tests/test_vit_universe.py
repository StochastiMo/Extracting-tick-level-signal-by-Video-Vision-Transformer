"""全市场 Universe 数据清洗规则的独立回归测试。"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


VIT_DIR = Path(__file__).resolve().parents[2] / "ViT"
if str(VIT_DIR) not in sys.path:
    sys.path.insert(0, str(VIT_DIR))

from vit_stage02_universe import classify_market_eligibility, horizon_dates


class UniverseCleaningTests(unittest.TestCase):
    """验证 T/T+1 市场状态过滤和 fail-closed 约束。"""

    def test_missing_suspended_and_limit_status_fail_closed(self) -> None:
        candidates = pd.DataFrame(
            {
                "tradestatus_t": ["交易", "停牌", "交易", None],
                "limit_status_t": [0, 0, 1, 0],
                "tradestatus_t1": ["交易", "交易", "交易", "交易"],
                "limit_status_t1": [0, -1, 0, None],
            }
        )

        result = classify_market_eligibility(candidates)

        np.testing.assert_array_equal(
            result["in_universe_t"].to_numpy(),
            np.array([True, False, False, False]),
        )
        np.testing.assert_array_equal(
            result["can_enter_t1"].to_numpy(),
            np.array([True, False, True, False]),
        )
        self.assertEqual(len(result), len(candidates))

    def test_exchange_calendar_defines_t_plus_1_and_t_plus_11(self) -> None:
        calendars = {
            "SH": np.array(
                [f"202401{day:02d}" for day in range(1, 14)], dtype=str
            )
        }

        entry, exit_date = horizon_dates("20240101", "SH", calendars)

        self.assertEqual(entry, "20240102")
        self.assertEqual(exit_date, "20240112")


if __name__ == "__main__":
    unittest.main()
