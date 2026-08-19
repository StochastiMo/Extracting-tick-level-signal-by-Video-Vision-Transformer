from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


VIVIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIVIT_DIR))
import stage02_barra_data as neutral_data
from stage02_barra_algebra import standardize_style_exposure
import stage04_no3_adversarial as no3_adversarial


TRAINING_PATH = VIVIT_DIR / "stage06_barra_training.py"
TRAINING_SPEC = importlib.util.spec_from_file_location(
    "stage06_barra_training_test_module", TRAINING_PATH
)
training = importlib.util.module_from_spec(TRAINING_SPEC)
assert TRAINING_SPEC.loader is not None
TRAINING_SPEC.loader.exec_module(training)


class StandardizationTests(unittest.TestCase):
    def test_valid_values_are_standardized_and_missing_is_zero(self):
        values = np.array([1.0, 2.0, 3.0, np.nan])
        valid = np.array([True, True, True, False])
        result = neutral_data._standardize_with_missing(
            values, valid, label="test"
        )
        self.assertAlmostEqual(float(result[:3].mean()), 0.0)
        self.assertAlmostEqual(float(result[:3].std(ddof=0)), 1.0)
        self.assertEqual(float(result[-1]), 0.0)

    def test_report_facing_standardization_interface_matches_backend(self):
        values = np.array([1.0, 2.0, 3.0, np.nan])
        valid = np.array([True, True, True, False])

        result = standardize_style_exposure(
            values, valid, label="test"
        )

        np.testing.assert_allclose(
            result[:3], [-1.22474487, 0.0, 1.22474487]
        )
        self.assertEqual(float(result[-1]), 0.0)

    def test_constant_exposure_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "常数"):
            neutral_data._standardize_with_missing(
                np.ones(5), np.ones(5, dtype=bool), label="constant"
            )


class ResidualTargetTests(unittest.TestCase):
    def test_target_residual_is_unit_scale_and_style_orthogonal(self):
        rng = np.random.default_rng(17)
        rows = 200
        styles = rng.normal(size=(rows, len(neutral_data.STYLE_COLUMNS)))
        coefficients = np.linspace(-0.5, 0.5, len(neutral_data.STYLE_COLUMNS))
        target = styles @ coefficients + rng.normal(scale=0.2, size=rows)
        valid = np.ones(rows, dtype=bool)

        residual, diagnostics = neutral_data.residualize_daily_targets(
            target, styles, valid
        )

        self.assertTrue(np.isfinite(residual).all())
        self.assertAlmostEqual(float(residual.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(residual.std(ddof=0)), 1.0, places=12)
        self.assertLess(
            float(diagnostics["max_abs_style_residual_pearson"]), 1e-10
        )
        correlations = np.corrcoef(
            np.column_stack([residual, styles]), rowvar=False
        )[0, 1:]
        self.assertLess(float(np.max(np.abs(correlations))), 1e-10)

    def test_invalid_labels_remain_nan_without_dropping_rows(self):
        rng = np.random.default_rng(23)
        rows = 80
        styles = rng.normal(size=(rows, len(neutral_data.STYLE_COLUMNS)))
        target = rng.normal(size=rows)
        target[-3:] = np.nan
        valid = np.isfinite(target)
        residual, diagnostics = neutral_data.residualize_daily_targets(
            target, styles, valid
        )
        self.assertEqual(len(residual), rows)
        self.assertTrue(np.isnan(residual[-3:]).all())
        self.assertTrue(np.isfinite(residual[:-3]).all())
        self.assertEqual(int(diagnostics["label_valid_rows"]), rows - 3)

    def test_rank_deficient_style_matrix_is_rejected(self):
        rng = np.random.default_rng(29)
        rows = 60
        styles = rng.normal(size=(rows, len(neutral_data.STYLE_COLUMNS)))
        styles[:, -1] = styles[:, -2]
        with self.assertRaisesRegex(np.linalg.LinAlgError, "矩阵秩"):
            neutral_data.residualize_daily_targets(
                rng.normal(size=rows), styles, np.ones(rows, dtype=bool)
            )


class FullBarraResidualTargetTests(unittest.TestCase):
    def test_full_residual_is_orthogonal_to_styles_and_industries(self):
        rng = np.random.default_rng(31)
        rows = 1200
        styles = rng.normal(
            size=(rows, len(neutral_data.STYLE_COLUMNS))
        )
        codes = rng.integers(
            0,
            neutral_data.EXPECTED_CNE5S_INDUSTRIES + 1,
            size=rows,
            dtype=np.int16,
        )
        target = (
            styles @ rng.normal(size=len(neutral_data.STYLE_COLUMNS))
            + rng.normal(size=neutral_data.EXPECTED_CNE5S_INDUSTRIES + 1)[
                codes
            ]
            + rng.normal(scale=0.2, size=rows)
        )
        residual, diagnostics = (
            neutral_data.residualize_daily_targets_with_industry(
                target, styles, codes, np.ones(rows, dtype=bool)
            )
        )
        self.assertAlmostEqual(float(residual.mean()), 0.0, places=12)
        self.assertAlmostEqual(float(residual.std(ddof=0)), 1.0, places=12)
        self.assertLess(
            float(diagnostics["max_abs_style_residual_pearson"]), 1e-10
        )
        self.assertLess(
            float(diagnostics["max_abs_industry_residual_mean"]), 1e-10
        )
        for code in np.unique(codes):
            self.assertAlmostEqual(
                float(residual[codes == code].mean()), 0.0, places=10
            )

    def test_full_residual_preserves_invalid_rows_and_unknown_group(self):
        rng = np.random.default_rng(37)
        rows = 300
        styles = rng.normal(
            size=(rows, len(neutral_data.STYLE_COLUMNS))
        )
        codes = rng.integers(0, 5, size=rows, dtype=np.int16)
        codes[:20] = 0
        target = rng.normal(size=rows)
        target[-5:] = np.nan
        valid = np.isfinite(target)
        residual, diagnostics = (
            neutral_data.residualize_daily_targets_with_industry(
                target, styles, codes, valid
            )
        )
        self.assertEqual(len(residual), rows)
        self.assertTrue(np.isnan(residual[-5:]).all())
        self.assertTrue(np.isfinite(residual[:-5]).all())
        self.assertGreater(int(diagnostics["unknown_industry_rows"]), 0)

    def test_full_residual_rejects_rank_deficiency_and_illegal_codes(self):
        rng = np.random.default_rng(41)
        rows = 120
        styles = rng.normal(
            size=(rows, len(neutral_data.STYLE_COLUMNS))
        )
        styles[:, -1] = styles[:, -2]
        target = rng.normal(size=rows)
        valid = np.ones(rows, dtype=bool)
        codes = np.arange(rows, dtype=np.int16) % 4
        with self.assertRaisesRegex(np.linalg.LinAlgError, "矩阵秩"):
            neutral_data.residualize_daily_targets_with_industry(
                target, styles, codes, valid
            )
        styles[:, -1] = rng.normal(size=rows)
        codes[0] = 33
        with self.assertRaisesRegex(ValueError, "行业代码"):
            neutral_data.residualize_daily_targets_with_industry(
                target, styles, codes, valid
            )

    def test_full_residual_rejects_constant_target(self):
        rng = np.random.default_rng(43)
        rows = 120
        styles = rng.normal(
            size=(rows, len(neutral_data.STYLE_COLUMNS))
        )
        codes = np.arange(rows, dtype=np.int16) % 4
        with self.assertRaisesRegex(ValueError, "常数"):
            neutral_data.residualize_daily_targets_with_industry(
                np.ones(rows), styles, codes, np.ones(rows, dtype=bool)
            )


class StrictFullBarraProjectionTests(unittest.TestCase):
    @staticmethod
    def sample(seed: int = 47, rows: int = 320):
        rng = np.random.default_rng(seed)
        styles = rng.normal(
            size=(rows, len(neutral_data.STYLE_COLUMNS))
        ).astype(np.float32)
        codes = np.repeat(np.arange(8), rows // 8).astype(np.int16)
        raw = (
            styles @ np.linspace(0.2, 1.1, styles.shape[1])
            + np.linspace(-1.0, 1.0, 8)[codes]
            + rng.normal(scale=0.3, size=rows)
        ).astype(np.float32)
        return rng, styles, codes, raw

    def test_projection_is_strictly_orthogonal_and_differentiable(self):
        rng, styles, codes, raw = self.sample()
        raw_tensor = torch.tensor(raw, requires_grad=True)
        style_tensor = torch.tensor(styles)
        code_tensor = torch.tensor(codes)

        projected, diagnostics = neutral_data.project_out_full_barra_torch(
            raw_tensor, style_tensor, code_tensor
        )

        self.assertAlmostEqual(float(projected.detach().mean()), 0.0, places=5)
        self.assertAlmostEqual(
            float(projected.detach().std(unbiased=False)), 1.0, places=5
        )
        self.assertLess(
            float(diagnostics["max_abs_style_pearson"]), 5e-5
        )
        self.assertLess(float(diagnostics["max_abs_industry_mean"]), 5e-5)
        for code in np.unique(codes):
            self.assertLess(
                abs(float(projected.detach().numpy()[codes == code].mean())),
                5e-5,
            )

        objective = torch.tensor(rng.normal(size=len(raw)), dtype=torch.float32)
        (projected * objective).sum().backward()
        self.assertIsNotNone(raw_tensor.grad)
        self.assertTrue(torch.isfinite(raw_tensor.grad).all())
        self.assertGreater(float(raw_tensor.grad.abs().sum()), 0.0)

    def test_strict_effective_prediction_switch_uses_full_barra(self):
        _, styles, codes, raw = self.sample(seed=53)
        raw_tensor = torch.tensor(raw)
        original = (
            training.PROJECT_MODEL_OUTPUT,
            training.BARRA_OUTPUT_NEUTRALIZATION,
        )
        try:
            training.PROJECT_MODEL_OUTPUT = False
            training.BARRA_OUTPUT_NEUTRALIZATION = True
            projected, diagnostics = training.effective_predictions(
                raw_tensor, torch.tensor(styles), torch.tensor(codes)
            )
            self.assertFalse(torch.equal(projected, raw_tensor))
            self.assertLess(
                float(diagnostics["max_abs_style_pearson"]), 5e-5
            )
            self.assertLess(
                float(diagnostics["max_abs_industry_mean"]), 5e-5
            )
            with self.assertRaisesRegex(ValueError, "行业代码"):
                training.effective_predictions(raw_tensor, torch.tensor(styles))
        finally:
            (
                training.PROJECT_MODEL_OUTPUT,
                training.BARRA_OUTPUT_NEUTRALIZATION,
            ) = original

    def test_projection_rejects_rank_deficiency_illegal_codes_and_constant(self):
        _, styles, codes, raw = self.sample(seed=59)
        styles[:, -1] = styles[:, -2]
        with self.assertRaisesRegex(np.linalg.LinAlgError, "秩不足"):
            neutral_data.project_out_full_barra_torch(
                torch.tensor(raw), torch.tensor(styles), torch.tensor(codes)
            )
        styles[:, -1] = np.random.default_rng(61).normal(size=len(styles))
        invalid_codes = codes.copy()
        invalid_codes[0] = 33
        with self.assertRaisesRegex(ValueError, "行业代码"):
            neutral_data.project_out_full_barra_torch(
                torch.tensor(raw), torch.tensor(styles), torch.tensor(invalid_codes)
            )
        with self.assertRaisesRegex(ValueError, "常数"):
            neutral_data.project_out_full_barra_torch(
                torch.ones(len(raw)), torch.tensor(styles), torch.tensor(codes)
            )


class PredictionProjectionTests(unittest.TestCase):

    def test_daily_score_dispersion_rejects_constant_predictions(self):
        prediction = torch.full((128,), 0.25, requires_grad=True)
        with self.assertRaises(training.DegenerateValidationScoreError) as caught:
            training.measure_daily_score_dispersion(
                prediction, date_id=20200102
            )
        self.assertEqual(caught.exception.date_id, 20200102)
        self.assertEqual(caught.exception.rows, 128)
        self.assertEqual(caught.exception.score_std, 0.0)
        self.assertEqual(caught.exception.total_sum_squares, 0.0)

    def test_daily_score_dispersion_is_finite_and_detached(self):
        prediction = torch.tensor(
            [-1.0, 0.0, 1.0, 2.0], requires_grad=True
        )
        diagnostics = training.measure_daily_score_dispersion(
            prediction, date_id=20200103
        )
        self.assertGreater(float(diagnostics["score_std"]), 0.0)
        self.assertGreater(float(diagnostics["total_sum_squares"]), 0.0)
        self.assertFalse(diagnostics["score_std"].requires_grad)

    def test_projection_removes_all_ten_styles_jointly(self):
        rng = np.random.default_rng(31)
        rows = 160
        styles = rng.normal(size=(rows, len(neutral_data.STYLE_COLUMNS)))
        prediction = styles @ np.linspace(
            0.1, 1.0, len(neutral_data.STYLE_COLUMNS)
        ) + rng.normal(scale=0.1, size=rows)
        residual, diagnostics = neutral_data.project_out_styles_numpy(
            prediction, styles
        )
        self.assertEqual(len(residual), rows)
        self.assertAlmostEqual(float(residual.mean()), 0.0, places=12)
        self.assertLess(float(diagnostics["max_abs_style_pearson"]), 1e-10)

    def test_nonfinite_projection_input_is_rejected(self):
        styles = np.eye(len(neutral_data.STYLE_COLUMNS) + 2)[
            :, : len(neutral_data.STYLE_COLUMNS)
        ]
        prediction = np.arange(len(styles), dtype=float)
        prediction[0] = np.nan
        with self.assertRaisesRegex(ValueError, "NaN/Inf"):
            neutral_data.project_out_styles_numpy(prediction, styles)

    def test_style_audit_is_detached_and_loss_remains_differentiable(self):
        generator = torch.Generator().manual_seed(37)
        styles = torch.randn(
            128, len(neutral_data.STYLE_COLUMNS), generator=generator
        )
        prediction = torch.randn(
            128, generator=generator, requires_grad=True
        )
        target = torch.randn(128, generator=generator)
        diagnostics = training.measure_style_correlations(prediction, styles)
        self.assertFalse(
            diagnostics["max_abs_style_pearson"].requires_grad
        )
        loss = training.rank_loss(prediction, target)
        loss.backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_torch_projection_is_standardized_orthogonal_and_differentiable(self):
        generator = torch.Generator().manual_seed(41)
        rows = 180
        styles = torch.randn(
            rows, len(neutral_data.STYLE_COLUMNS), generator=generator
        )
        prediction = (
            styles @ torch.linspace(0.1, 1.0, len(neutral_data.STYLE_COLUMNS))
            + 0.2 * torch.randn(rows, generator=generator)
        ).requires_grad_()

        effective, diagnostics = neutral_data.project_out_styles_torch(
            prediction, styles
        )

        self.assertAlmostEqual(
            float(effective.mean().detach()), 0.0, places=5
        )
        self.assertAlmostEqual(
            float(effective.std(unbiased=False).detach()), 1.0, places=5
        )
        self.assertLess(
            float(diagnostics["max_abs_style_pearson"]), 5e-5
        )
        effective.square().mean().backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_output_projection_switch_has_both_paths(self):
        generator = torch.Generator().manual_seed(43)
        styles = torch.randn(
            160, len(neutral_data.STYLE_COLUMNS), generator=generator
        )
        prediction = torch.randn(160, generator=generator)
        original = (
            training.PROJECT_MODEL_OUTPUT,
            training.BARRA_OUTPUT_NEUTRALIZATION,
        )
        try:
            training.BARRA_OUTPUT_NEUTRALIZATION = False
            training.PROJECT_MODEL_OUTPUT = False
            raw, raw_diagnostics = training.effective_predictions(
                prediction, styles
            )
            torch.testing.assert_close(raw, prediction)
            self.assertIsNone(raw_diagnostics)

            training.PROJECT_MODEL_OUTPUT = True
            projected, projected_diagnostics = training.effective_predictions(
                prediction, styles
            )
            self.assertIsNotNone(projected_diagnostics)
            self.assertFalse(torch.allclose(projected, prediction))
            self.assertLess(
                float(projected_diagnostics["max_abs_style_pearson"]), 5e-5
            )
        finally:
            (
                training.PROJECT_MODEL_OUTPUT,
                training.BARRA_OUTPUT_NEUTRALIZATION,
            ) = original

    def test_zero_lambda_rank_loss_matches_existing_objective(self):
        self.assertEqual(training.RAW_BARRA_R2_LAMBDA, 0.0)
        generator = torch.Generator().manual_seed(47)
        prediction = torch.randn(80, generator=generator)
        target = torch.randn(80, generator=generator)
        actual = training.rank_loss(prediction, target)
        expected = training.compute_training_loss(
            prediction,
            target,
            loss_name="sigmoid_rank_ic",
            rank_temperature=training.RANK_TEMPERATURE,
            turnover_lambda=training.TURNOVER_LAMBDA,
            huber_delta=training.HUBER_DELTA,
            smooth_l1_beta=training.SMOOTH_L1_BETA,
            hybrid_weight=training.HYBRID_WEIGHT,
        )
        torch.testing.assert_close(actual, expected)


    def test_four_switch_combinations_have_distinct_policies(self):
        original = (
            training.RESIDUALIZE_TARGET,
            training.PROJECT_MODEL_OUTPUT,
        )
        roles = set()
        try:
            for residualize_target in (False, True):
                for project_model_output in (False, True):
                    training.RESIDUALIZE_TARGET = residualize_target
                    training.PROJECT_MODEL_OUTPUT = project_model_output
                    roles.add(training.factor_role())
                    self.assertEqual(
                        "residual" in training.target_policy(),
                        residualize_target,
                    )
                    self.assertEqual(
                        "qr_residual" in training.output_policy(),
                        project_model_output,
                    )
        finally:
            (
                training.RESIDUALIZE_TARGET,
                training.PROJECT_MODEL_OUTPUT,
            ) = original
        self.assertEqual(len(roles), 4)



class JointBarraRSquaredTests(unittest.TestCase):
    @staticmethod
    def sample(seed: int = 53, rows: int = 320):
        rng = np.random.default_rng(seed)
        styles = rng.normal(
            size=(rows, len(neutral_data.STYLE_COLUMNS))
        )
        industries = np.repeat(np.arange(8), rows // 8).astype(np.int16)
        noise = rng.normal(scale=0.35, size=rows)
        score = (
            styles @ np.linspace(0.1, 0.8, styles.shape[1])
            + np.linspace(-0.7, 0.7, 8)[industries]
            + noise
        )
        return rng, styles, industries, score

    def test_numpy_matches_explicit_dummy_variable_ols(self):
        _, styles, industries, score = self.sample()
        dummies = np.eye(int(industries.max()) + 1)[industries][:, 1:]
        design = np.column_stack(
            [np.ones(len(score)), styles, dummies]
        )
        coefficients, _, _, _ = np.linalg.lstsq(
            design, score, rcond=None
        )
        residual = score - design @ coefficients
        centered = score - score.mean()
        expected = 1.0 - (
            np.dot(residual, residual) / np.dot(centered, centered)
        )

        actual, diagnostics = neutral_data.joint_barra_r_squared_numpy(
            score, styles, industries
        )

        self.assertAlmostEqual(actual, float(expected), places=12)
        self.assertEqual(diagnostics["industry_count"], 8)
        self.assertEqual(
            diagnostics["matrix_rank"],
            len(neutral_data.STYLE_COLUMNS),
        )
        self.assertLess(
            diagnostics["max_abs_industry_residual_mean"], 1e-12
        )

    def test_torch_matches_numpy_and_backpropagates_to_raw_score(self):
        _, styles, industries, score = self.sample(seed=59)
        expected, _ = neutral_data.joint_barra_r_squared_numpy(
            score, styles, industries
        )
        raw_score = torch.tensor(
            score, dtype=torch.float32, requires_grad=True
        )
        style_tensor = torch.tensor(styles, dtype=torch.float32)
        industry_tensor = torch.tensor(industries, dtype=torch.int16)

        actual, diagnostics = neutral_data.joint_barra_r_squared_torch(
            raw_score, style_tensor, industry_tensor
        )
        actual.backward()

        self.assertAlmostEqual(float(actual.detach()), expected, places=5)
        self.assertIsNotNone(raw_score.grad)
        self.assertTrue(torch.isfinite(raw_score.grad).all())
        self.assertGreater(float(raw_score.grad.abs().sum()), 0.0)
        self.assertFalse(style_tensor.requires_grad)
        self.assertFalse(industry_tensor.requires_grad)
        self.assertEqual(int(diagnostics["industry_count"]), 8)

    def test_barra_shift_is_invisible_to_projection_but_increases_r_squared(self):
        rng, styles, industries, _ = self.sample(seed=61)
        base_score = rng.normal(size=len(styles))
        shifted_score = (
            base_score
            + 8.0 * styles[:, 0]
            + 4.0 * industries.astype(np.float64)
        )
        base_residual, _ = neutral_data.project_out_styles_numpy(
            base_score, styles
        )
        shifted_without_industry, _ = neutral_data.project_out_styles_numpy(
            shifted_score - 4.0 * industries, styles
        )
        np.testing.assert_allclose(
            base_residual, shifted_without_industry, atol=1e-10, rtol=0.0
        )
        base_r_squared, _ = neutral_data.joint_barra_r_squared_numpy(
            base_score, styles, industries
        )
        shifted_r_squared, _ = neutral_data.joint_barra_r_squared_numpy(
            shifted_score, styles, industries
        )
        self.assertGreater(shifted_r_squared, base_r_squared)

    def test_combined_loss_is_rank_loss_plus_lambda_r_squared(self):
        _, styles, industries, score = self.sample(seed=67, rows=160)
        raw_score = torch.tensor(
            score, dtype=torch.float32, requires_grad=True
        )
        style_tensor = torch.tensor(styles, dtype=torch.float32)
        industry_tensor = torch.tensor(industries, dtype=torch.int16)
        target = torch.linspace(-1.0, 1.0, len(raw_score))
        original_lambda = training.RAW_BARRA_R2_LAMBDA
        try:
            training.RAW_BARRA_R2_LAMBDA = 0.3
            total, rank_loss, r_squared = training.combined_training_loss(
                raw_score,
                raw_score,
                target,
                style_tensor,
                industry_tensor,
            )
            torch.testing.assert_close(
                total, rank_loss.float() + 0.3 * r_squared
            )
            total.backward()
            self.assertTrue(torch.isfinite(raw_score.grad).all())
        finally:
            training.RAW_BARRA_R2_LAMBDA = original_lambda

    def test_invalid_joint_r_squared_inputs_fail_explicitly(self):
        _, styles, industries, score = self.sample(seed=71)
        with self.assertRaisesRegex(ValueError, "常数"):
            neutral_data.joint_barra_r_squared_numpy(
                np.ones(len(score)), styles, industries
            )
        invalid_codes = industries.copy()
        invalid_codes[0] = -1
        with self.assertRaisesRegex(ValueError, "不能为负数"):
            neutral_data.joint_barra_r_squared_numpy(
                score, styles, invalid_codes
            )


class AdversarialBarraTests(unittest.TestCase):
    def setUp(self):
        self.original = {
            "ADVERSARIAL_BARRA": training.ADVERSARIAL_BARRA,
            "ADVERSARIAL_GAMMA_MAX": training.ADVERSARIAL_GAMMA_MAX,
            "ADVERSARIAL_WARMUP_EPOCHS": training.ADVERSARIAL_WARMUP_EPOCHS,
            "ADVERSARIAL_RAMP_EPOCHS": training.ADVERSARIAL_RAMP_EPOCHS,
            "USE_TORCH_COMPILE": training.USE_TORCH_COMPILE,
        }
        training.ADVERSARIAL_BARRA = True
        training.USE_TORCH_COMPILE = 0

    def tearDown(self):
        for name, value in self.original.items():
            setattr(training, name, value)

    def test_gradient_reversal_is_identity_and_scales_negative_gradient(self):
        values = torch.tensor([1.0, -2.0, 3.0], requires_grad=True)
        output = no3_adversarial.gradient_reverse(values, 0.25)
        torch.testing.assert_close(output, values)
        output.sum().backward()
        torch.testing.assert_close(
            values.grad, torch.full_like(values, -0.25)
        )

    def test_warmup_and_cosine_ramp_use_absolute_epoch(self):
        training.ADVERSARIAL_GAMMA_MAX = 0.10
        training.ADVERSARIAL_WARMUP_EPOCHS = 2
        training.ADVERSARIAL_RAMP_EPOCHS = 5
        expected = [
            0.0, 0.0,
            0.05 * (1.0 - np.cos(np.pi / 5)),
            0.05 * (1.0 - np.cos(2 * np.pi / 5)),
            0.05 * (1.0 - np.cos(3 * np.pi / 5)),
            0.05 * (1.0 - np.cos(4 * np.pi / 5)),
            0.10, 0.10,
        ]
        actual = [training.adversarial_gamma(epoch) for epoch in range(1, 9)]
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.0)

    def test_masked_losses_ignore_missing_style_and_unknown_industry(self):
        styles = torch.zeros(4, training.STYLE_COUNT)
        style_prediction = torch.zeros_like(styles, requires_grad=True)
        style_prediction.data[0, 0] = 10.0
        style_valid = torch.ones_like(styles, dtype=torch.bool)
        style_valid[0, 0] = False
        logits = torch.zeros(4, training.INDUSTRY_COUNT, requires_grad=True)
        codes = torch.tensor([0, 1, 2, 3], dtype=torch.int16)
        weights = torch.ones(training.INDUSTRY_COUNT)

        style_loss, industry_loss = training.adversarial_training_loss(
            style_prediction, logits, styles, style_valid, codes, weights
        )
        self.assertEqual(float(style_loss.detach()), 0.0)
        self.assertAlmostEqual(float(industry_loss.detach()), 1.0, places=6)
        (style_loss + industry_loss).backward()
        self.assertEqual(float(style_prediction.grad[0, 0]), 0.0)
        self.assertEqual(float(logits.grad[0].abs().sum()), 0.0)
        self.assertGreater(float(logits.grad[1:].abs().sum()), 0.0)

    def test_training_industry_weights_are_inverse_sqrt_and_mean_one(self):
        counts = np.arange(1, training.INDUSTRY_COUNT + 1) ** 2
        codes = np.concatenate([
            np.full(count, industry + 1, dtype=np.int16)
            for industry, count in enumerate(counts)
        ])
        indices = np.arange(len(codes), dtype=np.int64)
        with (
            mock.patch.object(
                training.style_data, "require_industry_sidecar"
            ),
            mock.patch.object(training.np, "load", return_value=codes),
        ):
            actual = training.training_industry_class_weights(indices).numpy()
        expected = np.reciprocal(np.sqrt(counts.astype(np.float64)))
        expected /= expected.mean()
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-7)
        self.assertAlmostEqual(float(actual.mean()), 1.0, places=6)

    def test_adversarial_heads_backpropagate_to_encoder_not_revenue_head(self):
        config = training.base.VivitModelConfig(
            embedding_dim=16,
            spatial_encoder_layers=1,
            temporal_encoder_layers=1,
            attention_heads=4,
            mlp_hidden_dim=32,
            dropout=0.0,
        )
        with mock.patch.object(training.base, "MODEL_CONFIG", config):
            model = training.build_training_model(torch.device("cpu"), use_compile=False).train()
            videos = torch.randn(2, 17, 20, 8, 8)
            mask = torch.ones(2, 20, dtype=torch.bool)
            raw, style_prediction, industry_logits, representation = model(
                videos, mask, torch.tensor(0.2)
            )
            self.assertEqual(tuple(raw.shape), (2,))
            self.assertEqual(tuple(representation.shape), (2, 16))
            self.assertEqual(tuple(style_prediction.shape), (2, 10))
            self.assertEqual(tuple(industry_logits.shape), (2, 32))
            loss = style_prediction.square().mean() + industry_logits.square().mean()
            loss.backward()
            self.assertTrue(any(
                parameter.grad is not None
                and float(parameter.grad.abs().sum()) > 0
                for parameter in model.backbone.parameters()
            ))
            self.assertTrue(all(
                parameter.grad is None
                for parameter in model.backbone.regression_head.parameters()
            ))

    def test_probe_sampling_split_and_linear_recovery_are_deterministic(self):
        first = training.deterministic_probe_positions(
            900, year=2020, seed=42, date_id=20200102
        )
        second = training.deterministic_probe_positions(
            900, year=2020, seed=42, date_id=20200102
        )
        np.testing.assert_array_equal(first, second)
        self.assertEqual(len(first), 512)

        dates = np.repeat(np.arange(20), 20)
        split = training.chronological_probe_split(dates)
        date_sets = [set(dates[split[name]]) for name in ("train", "validation", "test")]
        self.assertTrue(date_sets[0].isdisjoint(date_sets[1]))
        self.assertTrue(date_sets[0].isdisjoint(date_sets[2]))
        self.assertTrue(date_sets[1].isdisjoint(date_sets[2]))

        rng = np.random.default_rng(79)
        representation = rng.normal(size=(400, 128)).astype(np.float32)
        coefficient = rng.normal(size=(128, training.STYLE_COUNT))
        styles = (representation @ coefficient).astype(np.float32)
        valid = np.ones_like(styles, dtype=bool)
        predictions = training.fit_linear_style_probe(
            representation, styles, valid, split, torch.device("cpu")
        )
        r_squared, _ = training._style_probe_metrics(
            predictions, styles[split["test"]], valid[split["test"]]
        )
        self.assertGreater(float(np.mean(r_squared)), 0.99)

        nonlinear_rows = 2000
        nonlinear_representation = rng.normal(
            size=(nonlinear_rows, 128)
        ).astype(np.float32)
        nonlinear_styles = np.tanh(
            nonlinear_representation[:, :training.STYLE_COUNT]
        ).astype(np.float32)
        nonlinear_valid = np.ones_like(nonlinear_styles, dtype=bool)
        nonlinear_dates = np.repeat(np.arange(20), nonlinear_rows // 20)
        nonlinear_split = training.chronological_probe_split(nonlinear_dates)
        nonlinear_predictions = training.fit_torch_probe(
            nonlinear_representation,
            nonlinear_styles,
            nonlinear_valid,
            nonlinear_split,
            torch.device("cpu"),
            output_dim=training.STYLE_COUNT,
            nonlinear=True,
            classification=False,
            seed=1083,
        )
        nonlinear_r_squared, _ = training._style_probe_metrics(
            nonlinear_predictions,
            nonlinear_styles[nonlinear_split["test"]],
            nonlinear_valid[nonlinear_split["test"]],
        )
        self.assertGreater(float(np.mean(nonlinear_r_squared)), 0.40)

        perfect_codes = np.tile(
            np.arange(1, training.INDUSTRY_COUNT + 1), 2
        )
        perfect_logits = np.full(
            (len(perfect_codes), training.INDUSTRY_COUNT), -10.0
        )
        perfect_logits[
            np.arange(len(perfect_codes)), perfect_codes - 1
        ] = 10.0
        industry_metrics = training._industry_probe_metrics(
            perfect_logits, perfect_codes
        )
        self.assertEqual(industry_metrics["accuracy"], 1.0)
        self.assertEqual(industry_metrics["balanced_accuracy"], 1.0)
        self.assertEqual(industry_metrics["macro_f1"], 1.0)

    def test_full_checkpoint_extracts_identical_raw_predictor(self):
        config = training.base.VivitModelConfig(
            embedding_dim=16,
            spatial_encoder_layers=1,
            temporal_encoder_layers=1,
            attention_heads=4,
            mlp_hidden_dim=32,
            dropout=0.0,
        )
        with mock.patch.object(training.base, "MODEL_CONFIG", config):
            wrapped = training.build_training_model(torch.device("cpu"), use_compile=False).eval()
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "best.pt"
                torch.save({"model_state": wrapped.state_dict()}, path)
                predictor = training.load_best_model(
                    path, torch.device("cpu")
                )
            videos = torch.randn(2, 17, 20, 8, 8)
            mask = torch.ones(2, 20, dtype=torch.bool)
            with torch.no_grad():
                wrapped_raw = wrapped(videos, mask, torch.tensor(0.0))[0]
                predictor_raw = predictor(videos, mask)
            torch.testing.assert_close(wrapped_raw, predictor_raw)


class RealIndustrySidecarTests(unittest.TestCase):
    def test_real_sidecar_has_unknown_plus_32_known_industries(self):
        path = (
            neutral_data.DEFAULT_INDUSTRY_STORE_DIR
            / neutral_data.INDUSTRY_MANIFEST_FILE
        )
        if not path.exists():
            self.skipTest("行业 sidecar 尚未构建")
        import json

        manifest = json.loads(path.read_text(encoding="utf-8"))
        mapping = manifest["industry_mapping"]
        self.assertEqual(mapping[neutral_data.barra.UNKNOWN_INDUSTRY], 0)
        self.assertEqual(manifest["known_industry_count"], 32)
        self.assertEqual(set(mapping.values()), set(range(33)))
        codes = np.load(
            neutral_data.DEFAULT_INDUSTRY_STORE_DIR
            / neutral_data.INDUSTRY_CODES_FILE,
            mmap_mode="r",
        )
        self.assertEqual(set(np.unique(codes)), set(range(33)))


class BarraConditioningTests(unittest.TestCase):
    def setUp(self):
        self.original = {
            "BARRA_CONDITIONING_MODE": training.BARRA_CONDITIONING_MODE,
            "BARRA_OUTPUT_NEUTRALIZATION": (
                training.BARRA_OUTPUT_NEUTRALIZATION
            ),
            "BARRA_GATE_HIDDEN_DIM": training.BARRA_GATE_HIDDEN_DIM,
            "BARRA_INDUSTRY_EMBEDDING_DIM": (
                training.BARRA_INDUSTRY_EMBEDDING_DIM
            ),
            "BARRA_GATE_DROPOUT": training.BARRA_GATE_DROPOUT,
            "BARRA_GATE_SCALE": training.BARRA_GATE_SCALE,
            "ADVERSARIAL_BARRA": training.ADVERSARIAL_BARRA,
            "USE_TORCH_COMPILE": training.USE_TORCH_COMPILE,
        }
        training.BARRA_CONDITIONING_MODE = "real"
        training.BARRA_OUTPUT_NEUTRALIZATION = False
        training.BARRA_GATE_HIDDEN_DIM = 16
        training.BARRA_INDUSTRY_EMBEDDING_DIM = 8
        training.BARRA_GATE_DROPOUT = 0.0
        training.BARRA_GATE_SCALE = 0.5
        training.ADVERSARIAL_BARRA = False
        training.USE_TORCH_COMPILE = 0
        self.config = training.base.VivitModelConfig(
            embedding_dim=16,
            spatial_encoder_layers=1,
            temporal_encoder_layers=1,
            attention_heads=4,
            mlp_hidden_dim=32,
            dropout=0.0,
        )

    def tearDown(self):
        for name, value in self.original.items():
            setattr(training, name, value)

    def _inputs(self, rows: int = 4):
        videos = torch.randn(rows, 17, 20, 8, 8)
        masks = torch.ones(rows, 20, dtype=torch.bool)
        styles = torch.randn(rows, training.STYLE_COUNT)
        valid = torch.ones_like(styles, dtype=torch.bool)
        codes = torch.arange(rows, dtype=torch.int16)
        return videos, masks, styles, valid, codes

    def test_zero_initialized_gate_matches_unconditioned_backbone(self):
        with mock.patch.object(training.base, "MODEL_CONFIG", self.config):
            model = training.build_training_model(torch.device("cpu"), use_compile=False).eval()
            videos, masks, styles, valid, codes = self._inputs()
            with torch.no_grad():
                raw, gate, representation = model(
                    videos, masks, styles, valid, codes
                )
                baseline = model.backbone(videos, masks)
                changed, changed_gate, _ = model(
                    videos, masks, styles + 100.0, valid, codes.flip(0)
                )
            torch.testing.assert_close(raw, baseline)
            torch.testing.assert_close(raw, changed)
            self.assertEqual(tuple(gate.shape), (4, self.config.embedding_dim))
            self.assertEqual(float(gate.abs().max()), 0.0)
            self.assertEqual(float(changed_gate.abs().max()), 0.0)
            self.assertEqual(
                float(model.industry_embedding.weight[0].detach().abs().max()), 0.0
            )
            self.assertEqual(
                tuple(representation.shape), (4, self.config.embedding_dim)
            )

    def test_gate_is_bounded_and_gradients_reach_barra_branch(self):
        with mock.patch.object(training.base, "MODEL_CONFIG", self.config):
            model = training.build_training_model(torch.device("cpu"), use_compile=False).train()
            torch.nn.init.constant_(model.gate_projection.weight, 0.2)
            videos, masks, styles, valid, codes = self._inputs()
            raw, gate, _ = model(videos, masks, styles, valid, codes)
            self.assertLessEqual(
                float(gate.detach().abs().max()), training.BARRA_GATE_SCALE
            )
            raw.square().mean().backward()
            self.assertTrue(any(
                parameter.grad is not None
                and float(parameter.grad.abs().sum()) > 0
                for parameter in model.barra_encoder.parameters()
            ))
            self.assertGreater(
                float(model.backbone.regression_head.weight.grad.abs().sum()),
                0.0,
            )
            self.assertEqual(
                float(model.industry_embedding.weight.grad[0].abs().sum()),
                0.0,
            )

    def test_conditioning_modes_and_shuffle_are_deterministic(self):
        styles = torch.arange(40, dtype=torch.float32).reshape(4, 10)
        valid = torch.tensor(
            [[True] * 10, [False] * 10, [True] * 10, [False] * 10]
        )
        codes = torch.tensor([1, 2, 3, 4], dtype=torch.int16)
        sample_indices = torch.tensor([13, 10, 12, 11], dtype=torch.int64)
        real = training.prepare_barra_conditioning_inputs(
            styles, valid, codes, sample_indices, mode="real"
        )
        self.assertIs(real[0], styles)
        zero = training.prepare_barra_conditioning_inputs(
            styles, valid, codes, sample_indices, mode="zero"
        )
        self.assertEqual(float(zero[0].abs().sum()), 0.0)
        self.assertFalse(bool(zero[1].any()))
        self.assertEqual(int(zero[2].abs().sum()), 0)
        shuffled_1 = training.prepare_barra_conditioning_inputs(
            styles, valid, codes, sample_indices, mode="shuffled"
        )
        shuffled_2 = training.prepare_barra_conditioning_inputs(
            styles, valid, codes, sample_indices, mode="shuffled"
        )
        for first, second in zip(shuffled_1, shuffled_2):
            torch.testing.assert_close(first, second)
        self.assertFalse(torch.equal(shuffled_1[0], styles))
        self.assertEqual(
            sorted(shuffled_1[2].tolist()), sorted(codes.tolist())
        )
        for row, mask, code in zip(*shuffled_1):
            source = int(row[0].item() // 10)
            self.assertEqual(bool(mask[0]), bool(valid[source, 0]))
            self.assertEqual(int(code), int(codes[source]))

    def test_full_checkpoint_reloads_identical_conditional_score(self):
        with mock.patch.object(training.base, "MODEL_CONFIG", self.config):
            model = training.build_training_model(torch.device("cpu"), use_compile=False).eval()
            torch.nn.init.normal_(model.gate_projection.weight, std=0.05)
            videos, masks, styles, valid, codes = self._inputs(2)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "best.pt"
                torch.save({"model_state": model.state_dict()}, path)
                restored = training.load_best_model(
                    path, torch.device("cpu")
                )
            with torch.no_grad():
                expected = model(videos, masks, styles, valid, codes)[0]
                actual = restored(videos, masks, styles, valid, codes)[0]
            torch.testing.assert_close(expected, actual)


class DualTowerTargetAndModelTests(unittest.TestCase):
    def setUp(self):
        names = (
            "BARRA_DUAL_TOWER", "BARRA_DUAL_TOWER_MODE",
            "BARRA_CONDITIONING_MODE", "BARRA_OUTPUT_NEUTRALIZATION",
            "ADVERSARIAL_BARRA", "RESIDUALIZE_TARGET",
            "PROJECT_MODEL_OUTPUT", "RAW_BARRA_R2_LAMBDA",
            "DUAL_TOWER_ALPHA_LOSS_WEIGHT",
            "DUAL_TOWER_STYLE_LOSS_WEIGHT",
            "DUAL_TOWER_STYLE_HIDDEN_DIM",
            "DUAL_TOWER_STYLE_DROPOUT",
            "BARRA_GATE_HIDDEN_DIM", "BARRA_INDUSTRY_EMBEDDING_DIM",
            "BARRA_GATE_DROPOUT", "BARRA_GATE_SCALE",
            "LOSS_NAME", "VALIDATION_METRIC",
        )
        self.original = {name: getattr(training, name) for name in names}
        training.BARRA_DUAL_TOWER = True
        training.BARRA_DUAL_TOWER_MODE = "real"
        training.BARRA_CONDITIONING_MODE = "off"
        training.BARRA_OUTPUT_NEUTRALIZATION = False
        training.ADVERSARIAL_BARRA = False
        training.RESIDUALIZE_TARGET = False
        training.PROJECT_MODEL_OUTPUT = False
        training.RAW_BARRA_R2_LAMBDA = 0.30
        training.DUAL_TOWER_ALPHA_LOSS_WEIGHT = 1.0
        training.DUAL_TOWER_STYLE_LOSS_WEIGHT = 1.0
        training.DUAL_TOWER_STYLE_HIDDEN_DIM = 16
        training.DUAL_TOWER_STYLE_DROPOUT = 0.0
        training.BARRA_GATE_HIDDEN_DIM = 16
        training.BARRA_INDUSTRY_EMBEDDING_DIM = 8
        training.BARRA_GATE_DROPOUT = 0.0
        training.BARRA_GATE_SCALE = 0.5
        training.LOSS_NAME = "sigmoid_rank_ic"
        training.VALIDATION_METRIC = "daily_mean_spearman_ic"
        self.config = training.base.VivitModelConfig(
            embedding_dim=16,
            spatial_encoder_layers=1,
            temporal_encoder_layers=1,
            attention_heads=4,
            mlp_hidden_dim=32,
            dropout=0.0,
        )

    def tearDown(self):
        for name, value in self.original.items():
            setattr(training, name, value)

    def _model_inputs(self, rows=8):
        videos = torch.randn(rows, 17, 20, 8, 8)
        masks = torch.ones(rows, 20, dtype=torch.bool)
        styles = torch.randn(rows, training.STYLE_COUNT)
        valid = torch.ones_like(styles, dtype=torch.bool)
        codes = torch.arange(rows, dtype=torch.int16) % 5
        indices = torch.arange(100, 100 + rows, dtype=torch.int64)
        return videos, masks, styles, valid, codes, indices

    def test_target_decomposition_reconstructs_and_spans_risk_space(self):
        rng = np.random.default_rng(101)
        rows = 1000
        styles = rng.normal(
            size=(rows, len(neutral_data.STYLE_COLUMNS))
        )
        codes = rng.integers(0, 9, size=rows, dtype=np.int16)
        target = (
            styles @ rng.normal(size=styles.shape[1])
            + rng.normal(size=9)[codes]
            + rng.normal(scale=0.35, size=rows)
        )
        residual, fitted, diagnostics = (
            neutral_data.decompose_daily_targets_with_industry(
                target, styles, codes, np.ones(rows, dtype=bool)
            )
        )
        for values in (residual, fitted):
            self.assertAlmostEqual(float(values.mean()), 0.0, places=10)
            self.assertAlmostEqual(
                float(values.std(ddof=0)), 1.0, places=10
            )
        self.assertLess(
            float(diagnostics["reconstruction_max_abs_error"]), 1e-12
        )
        self.assertLess(
            float(diagnostics["max_abs_style_residual_pearson"]), 1e-10
        )
        fitted_r_squared, _ = neutral_data.joint_barra_r_squared_numpy(
            fitted, styles, codes
        )
        self.assertAlmostEqual(float(fitted_r_squared), 1.0, places=10)

    def test_zero_initialization_and_four_input_modes(self):
        with mock.patch.object(training.base, "MODEL_CONFIG", self.config):
            model = training.build_training_model(torch.device("cpu"), use_compile=False).eval()
            videos, masks, styles, valid, codes, indices = self._model_inputs()
            with torch.no_grad():
                alpha, style, total, gate, _ = model(
                    videos, masks, styles, valid, codes,
                    styles, valid, codes,
                )
                expected_alpha = model.alpha_tower(
                    videos, masks, styles, valid, codes
                )[0]
            torch.testing.assert_close(alpha, expected_alpha)
            torch.testing.assert_close(style, torch.zeros_like(style))
            torch.testing.assert_close(total, alpha)
            self.assertEqual(float(gate.abs().max()), 0.0)
            self.assertEqual(
                float(
                    model.style_industry_embedding.weight[0]
                    .detach().abs().max()
                ),
                0.0,
            )
        for mode in training.BARRA_DUAL_TOWER_MODES:
            alpha_1, style_1 = training.prepare_dual_tower_inputs(
                styles, valid, codes, indices, mode=mode
            )
            alpha_2, style_2 = training.prepare_dual_tower_inputs(
                styles, valid, codes, indices, mode=mode
            )
            for first, second in zip(alpha_1 + style_1, alpha_2 + style_2):
                torch.testing.assert_close(first, second)
            for alpha_value, style_value in zip(alpha_1, style_1):
                torch.testing.assert_close(alpha_value, style_value)
        zero_alpha, zero_style = training.prepare_dual_tower_inputs(
            styles, valid, codes, indices, mode="zero"
        )
        self.assertEqual(float(zero_alpha[0].abs().sum()), 0.0)
        self.assertEqual(int(zero_style[2].abs().sum()), 0)

    def test_four_term_loss_is_exact_and_routes_score_gradients(self):
        rows = 180
        generator = torch.Generator().manual_seed(109)
        styles = torch.randn(
            rows, training.STYLE_COUNT, generator=generator
        )
        codes = torch.arange(rows, dtype=torch.int16) % 8
        alpha = torch.randn(rows, generator=generator, requires_grad=True)
        style = torch.randn(rows, generator=generator, requires_grad=True)
        pieces = training.dual_tower_training_loss(
            alpha, style, alpha + style,
            torch.randn(rows, generator=generator),
            torch.randn(rows, generator=generator),
            torch.randn(rows, generator=generator),
            styles, codes,
        )
        expected = (
            pieces["return_rank_loss"]
            + training.DUAL_TOWER_ALPHA_LOSS_WEIGHT
            * pieces["alpha_rank_loss"]
            + training.DUAL_TOWER_STYLE_LOSS_WEIGHT
            * pieces["style_rank_loss"]
            + training.RAW_BARRA_R2_LAMBDA
            * pieces["alpha_barra_r_squared"]
        )
        torch.testing.assert_close(pieces["loss"], expected)
        pieces["alpha_rank_loss"].backward(retain_graph=True)
        self.assertIsNotNone(alpha.grad)
        self.assertIsNone(style.grad)
        alpha.grad = None
        pieces["style_rank_loss"].backward(retain_graph=True)
        self.assertIsNone(alpha.grad)
        self.assertIsNotNone(style.grad)
        style.grad = None
        pieces["return_rank_loss"].backward()
        self.assertIsNotNone(alpha.grad)
        self.assertIsNotNone(style.grad)

    def test_bfloat16_total_is_audited_then_rebuilt_from_components(self):
        alpha = np.array([-0.9296875, 2.03125], dtype=np.float32)
        style = np.array([-0.10546875, 0.1484375], dtype=np.float32)
        stored_bfloat16 = (
            torch.from_numpy(alpha.copy()).to(torch.bfloat16)
            + torch.from_numpy(style.copy()).to(torch.bfloat16)
        ).float().numpy()
        self.assertFalse(np.array_equal(stored_bfloat16, alpha + style))

        canonical, audit = training.canonical_dual_tower_total_scores(
            alpha, style, stored_bfloat16
        )
        np.testing.assert_array_equal(canonical, alpha + style)
        self.assertEqual(audit["stored_total_precision"], "bfloat16")
        self.assertGreater(
            audit["stored_vs_canonical_max_abs_error"], 0.0
        )

        corrupted = stored_bfloat16.copy()
        corrupted[0] += np.float32(0.25)
        with self.assertRaisesRegex(AssertionError, "既不满足"):
            training.canonical_dual_tower_total_scores(
                alpha, style, corrupted
            )

    def test_dual_checkpoint_reload_preserves_three_scores(self):
        with mock.patch.object(training.base, "MODEL_CONFIG", self.config):
            model = training.build_training_model(torch.device("cpu"), use_compile=False).eval()
            torch.nn.init.normal_(model.style_output.weight, std=0.03)
            torch.nn.init.normal_(
                model.alpha_tower.gate_projection.weight, std=0.02
            )
            videos, masks, styles, valid, codes, _ = self._model_inputs(3)
            with tempfile.TemporaryDirectory() as directory:
                checkpoint = Path(directory) / "best.pt"
                torch.save({"model_state": model.state_dict()}, checkpoint)
                restored = training.load_best_model(
                    checkpoint, torch.device("cpu")
                )
            with torch.no_grad():
                expected = model(
                    videos, masks, styles, valid, codes,
                    styles, valid, codes,
                )[:3]
                actual = restored(
                    videos, masks, styles, valid, codes,
                    styles, valid, codes,
                )[:3]
            for expected_score, actual_score in zip(expected, actual):
                torch.testing.assert_close(expected_score, actual_score)

    def test_zero_initialized_style_rank_loss_has_finite_gradient(self):
        prediction = torch.zeros(128, requires_grad=True)
        target = torch.linspace(-1.0, 1.0, 128)
        loss = training.stable_sigmoid_rank_loss(prediction, target)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(float(prediction.grad.abs().sum()), 0.0)

    def test_style_loss_and_alpha_loss_update_only_their_own_tower(self):
        with mock.patch.object(training.base, "MODEL_CONFIG", self.config):
            model = training.build_training_model(torch.device("cpu"), use_compile=False).train()
            torch.nn.init.normal_(model.style_output.weight, std=0.03)
            videos, masks, styles, valid, codes, _ = self._model_inputs(12)
            alpha, style, _, _, _ = model(
                videos, masks, styles, valid, codes,
                styles, valid, codes,
            )
            target = torch.randn_like(style)
            training.training_loss(style, target).backward()
            self.assertTrue(any(
                parameter.grad is not None
                for parameter in model.style_encoder.parameters()
            ))
            self.assertTrue(all(
                parameter.grad is None
                for parameter in model.alpha_tower.parameters()
            ))
            model.zero_grad(set_to_none=True)
            alpha, _, _, _, _ = model(
                videos, masks, styles, valid, codes,
                styles, valid, codes,
            )
            training.training_loss(alpha, target).backward()
            self.assertTrue(any(
                parameter.grad is not None
                for parameter in model.alpha_tower.parameters()
            ))
            self.assertIsNone(model.style_output.weight.grad)


class NotebookInterfaceTests(unittest.TestCase):
    def test_notebook02_is_thin_and_notebook06_was_removed(self):
        import ast
        import json

        notebook = json.loads(
            (VIVIT_DIR / "02_train_rolling_vivit.ipynb").read_text(
                encoding="utf-8"
            )
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        tree = ast.parse(code)
        definitions = [
            node for node in ast.walk(tree)
            if isinstance(
                node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
        ]
        self.assertEqual(definitions, [])
        assigned_names = {
            target.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            for target in node.targets
            if isinstance(target, ast.Name)
        }
        self.assertTrue(
            {
                "RESIDUALIZE_TARGET",
                "PROJECT_MODEL_OUTPUT",
                "RAW_BARRA_R2_LAMBDA",
                "ADVERSARIAL_BARRA",
                "ADVERSARIAL_GAMMA_MAX",
                "BARRA_CONDITIONING_MODE",
                "BARRA_OUTPUT_NEUTRALIZATION",
                "BARRA_DUAL_TOWER",
                "BARRA_DUAL_TOWER_MODE",
                "DUAL_TOWER_ALPHA_LOSS_WEIGHT",
                "DUAL_TOWER_STYLE_LOSS_WEIGHT",
                "DUAL_TOWER_STYLE_HIDDEN_DIM",
                "DUAL_TOWER_STYLE_DROPOUT",
                "BARRA_GATE_HIDDEN_DIM",
                "BARRA_INDUSTRY_EMBEDDING_DIM",
                "BARRA_GATE_DROPOUT",
                "BARRA_GATE_SCALE",
                "RUN_FULL_TRAINING",
            }.issubset(assigned_names)
        )
        self.assertFalse(
            (VIVIT_DIR / "06_train_style_neutral_vivit.ipynb").exists()
        )

    def test_all_modes_have_distinct_default_run_names(self):
        import json

        notebook = json.loads(
            (VIVIT_DIR / "02_train_rolling_vivit.ipynb").read_text(
                encoding="utf-8"
            )
        )
        code = "\n".join(
            "".join(cell.get("source", []))
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        )
        for run_name in (
            "sigmoid_daybatch_rankic",
            "sigmoid_daybatch_rankic_style_residual_label",
            "sigmoid_daybatch_rankic_style_projected_output",
            "sigmoid_daybatch_rankic_style_residual_label_projected_output",
            "sigmoid_daybatch_rankic_raw_barra_r2_lambda_",
            "adversarial_barra_grl_0p10",
            "barra_gate_",
            "_strict_neutral",
            "barra_dual_tower_",
            "_eta_",
            "_mu_",
        ):
            self.assertIn(run_name, code)
        self.assertIn("daily_mean_spearman_ic", code)
        self.assertIn("联合 R² 早停只允许", code)
        self.assertIn("No.3 早停只允许", code)
        self.assertIn("No.4 早停只允许", code)
        self.assertIn("No.6 早停只允许", code)
        self.assertIn('BARRA_CONDITIONING_MODE = "off"', code)
        self.assertIn('BARRA_DUAL_TOWER_MODE = "real"', code)
        self.assertIn("RUN_FULL_TRAINING", code)
        self.assertIn('BACKTEST_MODE = "mode2_t_no_filter"', code)

if __name__ == "__main__":
    unittest.main()
