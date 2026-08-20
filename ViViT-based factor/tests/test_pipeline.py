from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from vivit_factor.data import (
    PipelinePaths,
    SampleIndexDataset,
    TradeDateBatchSampler,
    build_history_indices,
    calculate_forward_vwap_returns,
    classify_universe,
    mad_winsorize_zscore,
)
from vivit_factor.model import OrderFlowViViT, ViViTConfig, sigmoid_rank_ic_loss
from vivit_factor import training


class PipelineTests(unittest.TestCase):
    def test_requested_universe_rule(self) -> None:
        # 六行分别覆盖正常、T 停牌、T+1 停牌、T+1 涨停、图片无效和状态缺失。
        candidates = pd.DataFrame({
            "feature_valid": [True, True, True, True, False, True],
            "tradestatus_t": ["交易", "停牌", "交易", "交易", "交易", None],
            "tradestatus_t1": ["交易", "交易", "停牌", "交易", "交易", "交易"],
            "limit_status_t1": [0, 0, 0, 1, 0, 0],
        })
        result = classify_universe(candidates)
        np.testing.assert_array_equal(
            result["is_eligible"].to_numpy(),
            np.array([True, False, False, False, False, False]),
        )

    def test_adjusted_vwap_shift_and_mad_zscore(self) -> None:
        # 线性价格序列便于直接验证 T+1 到 T+11 的位移收益率。
        prices = pd.DataFrame({
            "s_info_windcode": ["000001.SZ"] * 13,
            "trade_dt": [f"202401{day:02d}" for day in range(1, 14)],
            "s_dq_avgprice": np.arange(1, 14, dtype=float),
            "s_dq_adjfactor": np.full(13, 2.0),
        })
        returns = calculate_forward_vwap_returns(prices)
        self.assertAlmostEqual(returns.loc[0, "forward_vwap_return_10d"], 12 / 2 - 1)
        self.assertEqual(returns.loc[0, "label_end_date"], "20240112")

        normalized = mad_winsorize_zscore(pd.Series([1.0, 2.0, 3.0, 100.0]))
        # 极端值应被裁剪，随后横截面应满足均值 0、总体标准差 1。
        self.assertLess(normalized["winsorized_forward_return"].iloc[-1], 100.0)
        self.assertAlmostEqual(normalized["target_zscore"].mean(), 0.0, places=12)
        self.assertAlmostEqual(normalized["target_zscore"].std(ddof=0), 1.0, places=12)

    def test_exact_history_and_model_check(self) -> None:
        # date_id=12 缺失，因此样本 2 的前一格必须保持 -1，不能向前填充。
        sample_index = pd.DataFrame({
            "sample_index": np.arange(4, dtype=np.int64),
            "instrument_id": ["A", "A", "A", "B"],
            "date_id": [10, 11, 13, 13],
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.npy"
            build_history_indices(sample_index, path)
            history = np.load(path)
        self.assertEqual(history[1, -2], 0)
        self.assertEqual(history[2, -2], -1)
        self.assertEqual(history[2, -3], 1)

        config = ViViTConfig(
            embedding_dim=16, spatial_encoder_layers=1, temporal_encoder_layers=1,
            attention_heads=4, mlp_hidden_dim=32, dropout=0.0,
        )
        model = OrderFlowViViT(config)
        # 构造含缺失 tubelet 的最小批次，同时检查前向、损失和反向传播。
        videos = torch.randn(3, 17, 20, 8, 8)
        mask = torch.ones(3, 20, dtype=torch.bool)
        mask[1, 4:6] = False
        videos[~mask[:, None, :, None, None].expand_as(videos)] = 0
        prediction = model(videos, mask)
        loss = sigmoid_rank_ic_loss(prediction, torch.tensor([0.2, -0.1, 0.7]))
        self.assertEqual(tuple(prediction.shape), (3,))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()

    def test_trade_date_sampler_returns_dataset_positions(self) -> None:
        # 全局索引不连续，用来防止采样器误把 sample_index 当作 Dataset 位置。
        global_indices = np.array([10, 11, 30, 31, 32], dtype=np.int64)
        date_ids = np.zeros(40, dtype=np.int32)
        date_ids[global_indices] = [1, 1, 2, 2, 2]
        dataset = SampleIndexDataset(global_indices)
        sampler = TradeDateBatchSampler(global_indices, date_ids, seed=7)
        batches = [[dataset[position] for position in batch] for batch in sampler]
        self.assertCountEqual(batches, [[10, 11], [30, 31, 32]])

    def test_factor_is_built_before_next_year_starts(self) -> None:
        # 用立即结束的假进程验证调度时序，不实际占用 GPU 或训练模型。
        events = []

        class FinishedProcess:
            def __init__(self, command, **_) -> None:
                year = int(command[command.index("--forecast-year") + 1])
                events.append(("launch", year))
                self.pid = year

            def poll(self) -> int:
                return 0

            def terminate(self) -> None:
                pass

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = PipelinePaths(root, root / "level2_pic", root / "Wind", root / "artifacts")
            for year in (2018, 2019):
                completed = paths.models_dir / "baseline" / f"forecast_{year}" / "seed_42" / "completed.json"
                completed.parent.mkdir(parents=True)
                completed.write_text(
                    f'{{"summary": {{"forecast_year": {year}, "seed": 42}}}}',
                    encoding="utf-8",
                )

            def record_factor(_, year: int) -> int:
                events.append(("factor", year))
                return 1

            with (
                patch.object(training, "FORECAST_YEARS", (2018, 2019)),
                patch.object(training, "SEEDS", (42,)),
                patch.object(training.torch.cuda, "device_count", return_value=1),
                patch.object(training.subprocess, "Popen", FinishedProcess),
                patch.object(training, "build_oos_factor", side_effect=record_factor),
                patch.object(training, "validate_outputs", return_value=pd.DataFrame()),
            ):
                training.run_parallel_training(paths, device_ids=(0,), workers=0)

        self.assertLess(events.index(("factor", 2018)), events.index(("launch", 2019)))


if __name__ == "__main__":
    unittest.main()
