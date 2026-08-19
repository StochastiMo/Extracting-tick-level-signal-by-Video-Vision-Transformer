from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.multiprocessing as mp


VIVIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIVIT_DIR))

import stage05_distributed_training as distributed
import stage06_barra_training as barra_training


def _gather_worker(rank: int, world_size: int, output_path: str) -> None:
    context = distributed.initialize_process_group(rank, world_size, backend="gloo")
    try:
        values = (
            torch.tensor([1.0, 2.0])
            if rank == 0
            else torch.tensor([3.0, 4.0, 5.0])
        ).requires_grad_()
        gathered = distributed.differentiable_all_gather(values)
        target = torch.arange(5, dtype=torch.float32)
        loss = (gathered - target).square().mean()
        loss.backward()
        gradients = distributed.all_gather_objects(values.grad.tolist())
        if context.is_primary:
            Path(output_path).write_text(
                json.dumps(
                    {
                        "gathered": gathered.detach().tolist(),
                        "loss": float(loss.detach()),
                        "gradients": gradients,
                    }
                ),
                encoding="utf-8",
            )
    finally:
        distributed.destroy_process_group()


def _conditioning_worker(rank: int, world_size: int, output_path: str) -> None:
    context = distributed.initialize_process_group(rank, world_size, backend="gloo")
    try:
        barra_training.BARRA_DUAL_TOWER = True
        barra_training.BARRA_DUAL_TOWER_MODE = "shuffled"
        barra_training.BARRA_CONDITIONING_MODE = "off"
        indices = torch.tensor([101, 103, 107, 109, 113], dtype=torch.int64)
        styles = torch.arange(50, dtype=torch.float32).reshape(5, 10)
        valid = torch.ones((5, 10), dtype=torch.bool)
        industries = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int16)
        local = torch.arange(rank, 5, world_size)
        transformed = barra_training.prepare_distributed_conditioning(
            styles[local], valid[local], industries[local], indices[local], context
        )
        gathered_indices = distributed.all_gather_tensor(indices[local])
        gathered_styles = distributed.all_gather_tensor(transformed[0])
        gathered_industries = distributed.all_gather_tensor(transformed[2])
        order = torch.argsort(gathered_indices)
        if context.is_primary:
            Path(output_path).write_text(
                json.dumps(
                    {
                        "styles": gathered_styles[order].tolist(),
                        "industries": gathered_industries[order].tolist(),
                        "prepared": transformed[3],
                    }
                ),
                encoding="utf-8",
            )
    finally:
        distributed.destroy_process_group()


class DistributedTrainingTests(unittest.TestCase):
    def test_near_constant_style_score_is_classified_as_collapsed(self):
        scores = torch.tensor(
            [0.0, 1.0e-13, 2.0e-13, 3.0e-13], dtype=torch.float64
        ).numpy()
        targets = torch.tensor(
            [-1.5, -0.5, 0.5, 1.5], dtype=torch.float64
        ).numpy()
        dates = torch.zeros(4, dtype=torch.int64).numpy()
        self.assertGreater(float(scores.std()), torch.finfo(torch.float64).eps)
        self.assertTrue(
            barra_training.has_degenerate_daily_pearson(scores, targets, dates)
        )
        self.assertFalse(
            barra_training.has_degenerate_daily_pearson(targets, targets, dates)
        )
        invalid = scores.copy()
        invalid[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            barra_training.has_degenerate_daily_pearson(invalid, targets, dates)

    def test_fixed_batch_sampler_has_no_padding_duplicates_or_loss(self):
        samplers = [
            distributed.DistributedBatchSampler(
                12,
                5,
                rank=rank,
                world_size=2,
                seed=7,
                shuffle=False,
                drop_last=False,
            )
            for rank in range(2)
        ]
        rank_batches = [list(sampler) for sampler in samplers]
        rebuilt = []
        for batch_number in range(len(rank_batches[0])):
            logical = rank_batches[0][batch_number] + rank_batches[1][batch_number]
            rebuilt.extend(logical)
            self.assertEqual(len(logical), len(set(logical)))
        self.assertEqual(sorted(rebuilt), list(range(12)))

    @unittest.skipUnless(torch.distributed.is_available(), "torch.distributed unavailable")
    def test_variable_length_gather_matches_single_process_loss_and_gradient(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "result.json")
            distributed.configure_master(distributed.reserve_local_port())
            mp.spawn(
                _gather_worker,
                args=(2, output_path),
                nprocs=2,
                join=True,
            )
            result = json.loads(Path(output_path).read_text(encoding="utf-8"))

        reference = torch.arange(1.0, 6.0, requires_grad=True)
        target = torch.arange(5, dtype=torch.float32)
        reference_loss = (reference - target).square().mean()
        reference_loss.backward()
        gathered_gradients = torch.tensor(
            result["gradients"][0] + result["gradients"][1]
        ) / 2
        self.assertEqual(result["gathered"], reference.detach().tolist())
        self.assertAlmostEqual(result["loss"], float(reference_loss.detach()), places=7)
        torch.testing.assert_close(gathered_gradients, reference.grad)


    def test_shuffled_barra_is_built_once_from_complete_cross_section(self):
        with tempfile.TemporaryDirectory() as directory:
            output_path = str(Path(directory) / "conditioning.json")
            distributed.configure_master(distributed.reserve_local_port())
            mp.spawn(
                _conditioning_worker,
                args=(2, output_path),
                nprocs=2,
                join=True,
            )
            result = json.loads(Path(output_path).read_text(encoding="utf-8"))

        indices = torch.tensor([101, 103, 107, 109, 113], dtype=torch.int64)
        styles = torch.arange(50, dtype=torch.float32).reshape(5, 10)
        valid = torch.ones((5, 10), dtype=torch.bool)
        industries = torch.tensor([1, 2, 3, 4, 5], dtype=torch.int16)
        with (
            mock.patch.object(barra_training, "BARRA_DUAL_TOWER", True),
            mock.patch.object(
                barra_training, "BARRA_DUAL_TOWER_MODE", "shuffled"
            ),
        ):
            expected = barra_training.prepare_barra_conditioning_inputs(
                styles, valid, industries, indices, mode="shuffled"
            )
        self.assertTrue(result["prepared"])
        torch.testing.assert_close(torch.tensor(result["styles"]), expected[0])
        self.assertEqual(result["industries"], expected[2].tolist())

    def test_ddp_cli_does_not_expose_tasks_per_gpu(self):
        with mock.patch.object(
            sys,
            "argv",
            ["stage06_barra_training.py", "ddp", "--device-ids", "0,1"],
        ):
            ddp_args = barra_training.parse_args()
        with mock.patch.object(
            sys,
            "argv",
            ["stage06_barra_training.py", "a2", "--device-ids", "0,1"],
        ):
            a2_args = barra_training.parse_args()
        self.assertFalse(hasattr(ddp_args, "tasks_per_gpu"))
        self.assertEqual(a2_args.tasks_per_gpu, 1)

    def test_notebook_builds_distinct_a2_and_ddp_commands(self):
        notebook = json.loads(
            (VIVIT_DIR / "02_train_rolling_vivit.ipynb").read_text(
                encoding="utf-8"
            )
        )
        training_source = "".join(notebook["cells"][6]["source"])
        common = {
            "PYTHON": Path("python"),
            "BACKEND": Path("backend.py"),
            "TRAINING_ARGS": ["--run-name", "demo"],
            "DEVICE_IDS": "0,1",
            "TASKS_PER_GPU": 2,
            "RUN_FULL_TRAINING": False,
        }
        a2_scope = {**common, "TRAINING_ROUTE": "a2"}
        ddp_scope = {**common, "TRAINING_ROUTE": "ddp"}
        exec(training_source, a2_scope)
        exec(training_source, ddp_scope)
        self.assertEqual(a2_scope["training_command"][2], "a2")
        self.assertIn("--tasks-per-gpu", a2_scope["training_command"])
        self.assertEqual(ddp_scope["training_command"][2], "ddp")
        self.assertNotIn("--tasks-per-gpu", ddp_scope["training_command"])

        configuration_source = "".join(notebook["cells"][2]["source"])
        self.assertTrue(
            any(
                line.strip()
                in {'TRAINING_ROUTE = "a2"', 'TRAINING_ROUTE = "ddp"'}
                for line in configuration_source.splitlines()
            )
        )
        self.assertIn('f"{BASE_RUN_NAME}_ddp"', configuration_source)
        smoke_source = "".join(notebook["cells"][4]["source"])
        self.assertIn('"ddp-smoke"', smoke_source)
        self.assertIn('["--device-ids", DEVICE_IDS]', smoke_source)
        smoke_call = smoke_source[smoke_source.index("smoke = subprocess.run(") :]
        self.assertLess(
            smoke_call.index("*smoke_device_args"),
            smoke_call.index("],\n        cwd=VIVIT_DIR"),
        )


if __name__ == "__main__":
    unittest.main()
