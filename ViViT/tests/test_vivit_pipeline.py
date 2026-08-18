from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

VIVIT_DIR = Path(__file__).resolve().parents[1]
VIT_DIR = VIVIT_DIR.parent / "ViT"
sys.path.insert(0, str(VIVIT_DIR))
sys.path.insert(0, str(VIT_DIR))

import vit_stage06_ranking as vit_ranking
import vit_stage07_training as vit_training
import stage01_sequence_data as pipeline
import stage03_model as model_module
import stage05_baseline_training as training


class HistoryIndexTests(unittest.TestCase):
    def setUp(self):
        self.sample_index = pd.DataFrame(
            {
                "sample_index": np.arange(6, dtype=np.int64),
                "trade_date": ["d10", "d10", "d11", "d12", "d13", "d14"],
                "instrument_id": [1, 2, 1, 2, 1, 2],
                "date_id": [10, 10, 11, 12, 13, 14],
            }
        )

    def test_exact_history_slots_gaps_and_no_future(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "history.npy"
            pipeline.build_history_indices(self.sample_index, path)
            history = np.load(path)

        self.assertEqual(history.shape, (6, 20))
        self.assertEqual(history.dtype, np.int32)
        np.testing.assert_array_equal(history[:, -1], np.arange(6))
        self.assertEqual(history[2, -2], 0)  # 股票1：date 11 精确回看 date 10
        self.assertEqual(history[4, -2], -1)  # 股票1：date 13 没有 date 12
        self.assertEqual(history[4, -3], 2)  # 但精确 lag=2 能匹配 date 11
        self.assertEqual(history[5, -2], -1)  # 股票2：date 14 没有 date 13
        self.assertEqual(history[5, -3], 3)  # lag=2 匹配 date 12
        self.assertTrue((history[:, :-1] == -1).any())
        dates = self.sample_index["date_id"].to_numpy()
        for target, row in enumerate(history):
            valid = row[row >= 0]
            self.assertTrue(np.all(dates[valid] <= dates[target]))

    def test_source_signature_mismatch_requires_explicit_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "sequence_manifest.json"
            manifest_path.write_text(json.dumps({"signature": "stale"}))
            with (
                mock.patch.object(
                    pipeline,
                    "aligned_store_payload",
                    return_value={
                        "source_sample_count": 1,
                        "history_shape": [1, 20],
                    },
                ),
                mock.patch.multiple(
                    pipeline,
                    HISTORY_INDICES_PATH=root / "history.npy",
                    SEQUENCE_MANIFEST_PATH=manifest_path,
                    SEQUENCE_SUMMARY_PATH=root / "summary.parquet",
                ),
            ):
                with self.assertRaisesRegex(ValueError, "--overwrite 1"):
                    pipeline.build_sequence_store(overwrite=False)

    def test_collator_zero_fill_and_boolean_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = np.stack(
                [np.full((17, 8, 8), i + 1, np.float32) for i in range(3)]
            )
            targets = np.array([0.1, 0.2, 0.3], np.float32)
            date_ids = np.array([10, 11, 12], np.int32)
            history = np.full((3, 20), -1, np.int32)
            history[:, -1] = np.arange(3)
            history[2, -3] = 0
            np.save(root / "features.npy", features)
            np.save(root / "targets.npy", targets)
            np.save(root / "date_ids.npy", date_ids)
            np.save(root / "history.npy", history)
            with mock.patch.multiple(
                pipeline,
                FEATURES_PATH=root / "features.npy",
                TARGETS_PATH=root / "targets.npy",
                DATE_IDS_PATH=root / "date_ids.npy",
                HISTORY_INDICES_PATH=root / "history.npy",
            ):
                batch = pipeline.VideoBatchCollator()([2])

        self.assertEqual(tuple(batch["videos"].shape), (1, 17, 20, 8, 8))
        self.assertEqual(batch["frame_mask"].dtype, torch.bool)
        self.assertTrue(batch["frame_mask"][0, -1])
        self.assertTrue(batch["frame_mask"][0, -3])
        self.assertFalse(batch["frame_mask"][0, -2])
        self.assertTrue(torch.count_nonzero(batch["videos"][0, :, -2]) == 0)
        self.assertTrue(torch.all(batch["videos"][0, :, -3] == 1))
        self.assertTrue(torch.all(batch["videos"][0, :, -1] == 3))


class ModelMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.config = model_module.VivitModelConfig(
            embedding_dim=16,
            spatial_encoder_layers=1,
            temporal_encoder_layers=1,
            attention_heads=4,
            mlp_hidden_dim=32,
            dropout=0.0,
        )

    def test_full_partial_and_missing_tubelets_are_finite(self):
        model = model_module.OrderFlowViViT(self.config).eval()
        videos = torch.randn(3, 17, 20, 8, 8)
        masks = torch.ones(3, 20, dtype=torch.bool)
        masks[1, 2] = False  # 部分缺失 Tubelet
        masks[2, 4:6] = False  # 整个 Tubelet 缺失
        videos[~masks[:, None, :, None, None].expand_as(videos)] = 0
        captured = {}

        def capture(_module, _args, kwargs):
            captured["mask"] = kwargs["key_padding_mask"].detach().clone()

        handle = model.temporal_blocks[0].attention.register_forward_pre_hook(
            capture, with_kwargs=True
        )
        try:
            output = model(videos, masks)
        finally:
            handle.remove()

        self.assertEqual(tuple(output.shape), (3,))
        self.assertTrue(torch.isfinite(output).all())
        self.assertEqual(tuple(captured["mask"].shape), (3, 11))
        self.assertFalse(captured["mask"][:, 0].any())  # Class token
        self.assertFalse(captured["mask"][1, 2])  # 1/2 覆盖仍保留
        self.assertTrue(captured["mask"][2, 3])  # 帧 4:6 对应第 3 个时序位置
        self.assertTrue(torch.count_nonzero(model.tubelet_coverage_embedding.weight[2]) == 0)

    def test_target_frame_must_be_valid(self):
        model = model_module.OrderFlowViViT(self.config).eval()
        mask = torch.ones(1, 20, dtype=torch.bool)
        mask[0, -1] = False
        with self.assertRaises(AssertionError):
            model(torch.zeros(1, 17, 20, 8, 8), mask)

    def test_encode_and_regression_head_exactly_match_forward(self):
        model = model_module.OrderFlowViViT(self.config).eval()
        videos = torch.randn(2, 17, 20, 8, 8)
        masks = torch.ones(2, 20, dtype=torch.bool)
        masks[1, 4:6] = False
        videos[~masks[:, None, :, None, None].expand_as(videos)] = 0
        with torch.no_grad():
            representation = model.encode(videos, masks)
            expected = model.regression_head(representation).squeeze(-1)
            actual = model(videos, masks)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
        self.assertEqual(tuple(representation.shape), (2, 16))
        keys = set(model.state_dict())
        self.assertTrue(any(key.startswith("regression_head.") for key in keys))
        self.assertFalse(any(key.startswith("encode.") for key in keys))

    def test_trade_date_compile_is_dynamic(self):
        sentinel = object()
        with (
            mock.patch.object(training.OrderFlowViViT, "to", return_value=mock.Mock()),
            mock.patch.object(training.torch, "compile", return_value=sentinel) as compile_mock,
        ):
            result = training.build_model(
                torch.device("cuda"), True, "default", "trade_date"
            )
        self.assertIs(result, sentinel)
        self.assertTrue(compile_mock.call_args.kwargs["dynamic"])
        self.assertEqual(compile_mock.call_args.kwargs["mode"], "default")


class ViTPrimitiveParityTests(unittest.TestCase):
    def setUp(self):
        self.predictions = torch.tensor([0.2, -0.4, 0.9, 0.1], requires_grad=True)
        self.targets = torch.tensor([-0.3, 0.6, 0.8, -0.2])
        self.kwargs = {
            "rank_temperature": 0.2,
            "turnover_lambda": 0.3,
            "huber_delta": 1.2,
            "smooth_l1_beta": 0.4,
            "hybrid_weight": 0.6,
        }

    def test_all_losses_use_identical_vit_primitive(self):
        self.assertIs(training.compute_training_loss, vit_ranking.compute_training_loss)
        previous = torch.tensor([-0.2, 0.7, 0.4, 0.0])
        positions = np.arange(4)
        for name in vit_ranking.LOSS_NAMES:
            extras = {}
            if name == vit_ranking.TEMPORAL_TURNOVER_LOSS:
                extras = {
                    "previous_predictions": previous,
                    "previous_positions": positions,
                    "current_positions": positions,
                }
            left = training.compute_training_loss(
                self.predictions, self.targets, loss_name=name, **self.kwargs, **extras
            )
            right = vit_ranking.compute_training_loss(
                self.predictions, self.targets, loss_name=name, **self.kwargs, **extras
            )
            torch.testing.assert_close(left, right)
            self.assertTrue(torch.isfinite(left))

    def test_trade_date_and_fixed_sample_batch_semantics(self):
        self.assertIs(pipeline.TradeDateBatchSampler, vit_training.TradeDateBatchSampler)
        indices = np.arange(7)
        dates = np.array([1, 1, 2, 2, 2, 3, 3])
        left = pipeline.TradeDateBatchSampler(indices, dates, seed=11, shuffle=True)
        right = vit_training.TradeDateBatchSampler(indices, dates, seed=11, shuffle=True)
        left.set_epoch(4)
        right.set_epoch(4)
        self.assertEqual([list(x) for x in left], [list(x) for x in right])
        fixed = DataLoader(
            pipeline.SampleIndexDataset(indices), batch_size=3, shuffle=False, drop_last=True
        )
        self.assertEqual([batch.tolist() for batch in fixed], [[0, 1, 2], [3, 4, 5]])

    def test_both_validation_metrics_are_numerically_identical(self):
        self.assertIs(
            training.summarize_validation_arrays,
            vit_training.summarize_validation_arrays,
        )
        predictions = np.array([0.1, 0.8, -0.2, 0.4, -0.5, 0.7])
        targets = np.array([-0.1, 0.9, 0.2, 0.3, -0.7, 0.8])
        dates = np.array([1, 1, 1, 2, 2, 2])
        instruments = np.array([10, 11, 12, 10, 11, 12])
        for metric in vit_training.VALIDATION_METRICS:
            left = training.summarize_validation_arrays(
                predictions, targets, dates, metric, instrument_ids=instruments
            )
            right = vit_training.summarize_validation_arrays(
                predictions, targets, dates, metric, instrument_ids=instruments
            )
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assertAlmostEqual(float(left[key]), float(right[key]), places=12)


class RealDataIntegrationTests(unittest.TestCase):
    def test_current_manifest_and_all_splits_align(self):
        if not pipeline.SEQUENCE_MANIFEST_PATH.exists():
            self.skipTest("aligned store 尚未构建")
        manifest = pipeline.require_aligned_sequence_store()
        dataset = json.loads(pipeline.SOURCE_DATASET_MANIFEST_PATH.read_text())
        rolling = json.loads(pipeline.SOURCE_ROLLING_MANIFEST_PATH.read_text())
        self.assertEqual(
            manifest["payload"]["source_dataset_signature"], dataset["signature"]
        )
        self.assertEqual(
            manifest["payload"]["source_rolling_signature"], rolling["signature"]
        )
        audit = pipeline.audit_split_alignment()
        self.assertEqual(len(audit), 8 * 3)
        self.assertTrue(audit["array_equal"].all())
        prediction_2025 = audit.query(
            "forecast_year == 2025 and split == 'prediction'"
        )["sample_count"].item()
        self.assertEqual(prediction_2025, 0)


if __name__ == "__main__":
    unittest.main()
