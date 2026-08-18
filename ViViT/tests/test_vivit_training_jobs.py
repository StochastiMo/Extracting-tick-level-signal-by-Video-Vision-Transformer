from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock


VIVIT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(VIVIT_DIR))

import stage07_training_jobs as jobs


class TrainingJobTests(unittest.TestCase):
    def test_screen_name_is_safe_and_deterministic(self):
        self.assertEqual(
            jobs.default_screen_session_name("penalty r2/0.30"),
            "vivit_penalty_r2_0.30",
        )
        long_name = jobs.default_screen_session_name("x" * 200)
        self.assertLessEqual(len(long_name), 80)
        self.assertEqual(long_name, jobs.default_screen_session_name("x" * 200))

    def test_status_distinguishes_completed_resumable_and_queued_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            completed_root = root / "models/runs/demo/forecast_2018/seed_42"
            completed_root.mkdir(parents=True)
            (completed_root / "completed.json").write_text(
                json.dumps({"summary": {"last_epoch": 11}}), encoding="utf-8"
            )
            resumable_root = root / "models/runs/demo/forecast_2018/seed_43"
            resumable_root.mkdir(parents=True)
            (resumable_root / "last.pt").touch()
            with (
                mock.patch.object(jobs, "_screen_sessions", return_value={}),
                mock.patch.object(jobs, "_training_processes", return_value=[]),
            ):
                status = jobs.training_job_status("demo", vivit_dir=root)

        states = {
            (item["forecast_year"], item["seed"]): item["state"]
            for item in status["tasks"]
        }
        self.assertEqual(status["status"], "resumable")
        self.assertEqual(status["completed_tasks"], 1)
        self.assertEqual(status["resumable_tasks"], 1)
        self.assertEqual(states[(2018, 42)], "completed")
        self.assertEqual(states[(2018, 43)], "resumable")
        self.assertEqual(states[(2018, 44)], "queued")

    def test_completed_outputs_are_reused_without_starting_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for year in jobs.FORECAST_YEARS:
                for seed in jobs.SEEDS:
                    task_root = root / f"models/runs/demo/forecast_{year}/seed_{seed}"
                    task_root.mkdir(parents=True)
                    (task_root / "completed.json").write_text(
                        json.dumps({"summary": {"last_epoch": 3}}), encoding="utf-8"
                    )
            prediction_root = root / "data/predictions/runs/demo"
            prediction_root.mkdir(parents=True)
            for year in jobs.FORECAST_YEARS[:-1]:
                (prediction_root / f"vivit_predictions.{year}.parquet").touch()
            with (
                mock.patch.object(jobs.shutil, "which", return_value="/usr/bin/screen"),
                mock.patch.object(jobs, "_screen_sessions", return_value={}),
                mock.patch.object(jobs, "_training_processes", return_value=[]),
                mock.patch.object(jobs.subprocess, "run") as run_mock,
            ):
                status = jobs.launch_training_job(
                    ["python", "backend.py", "a2", "--run-name", "demo"],
                    "demo",
                    vivit_dir=root,
                )

        self.assertEqual(status["status"], "completed_existing")
        self.assertEqual(status["launch_action"], "reuse_completed")
        run_mock.assert_not_called()

    def test_launch_writes_manifest_and_starts_detached_screen(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = jobs.default_screen_session_name("demo")
            with (
                mock.patch.object(jobs.shutil, "which", return_value="/usr/bin/screen"),
                mock.patch.object(
                    jobs,
                    "_screen_sessions",
                    side_effect=[{}, {}, {session: f"123.{session}"}],
                ),
                mock.patch.object(jobs, "_training_processes", return_value=[]),
                mock.patch.object(jobs.subprocess, "run") as run_mock,
                mock.patch.object(jobs.time, "sleep"),
            ):
                status = jobs.launch_training_job(
                    ["python", "backend.py", "a2", "--run-name", "demo"],
                    "demo",
                    vivit_dir=root,
                )

            job_path = root / "data/quality/runs/demo/background_job/job.json"
            payload = json.loads(job_path.read_text(encoding="utf-8"))

        self.assertEqual(status["status"], "running")
        self.assertEqual(status["launch_action"], "launched")
        self.assertEqual(payload["status"], "launching")
        screen_command = run_mock.call_args.args[0]
        self.assertEqual(screen_command[:3], ["/usr/bin/screen", "-dmS", session])
        self.assertIn("run-job", screen_command)

    def test_ddp_worker_process_is_reported_as_active_task_parent(self):
        completed = mock.Mock(
            stdout=(
                "4321 python stage06_barra_training.py ddp-worker "
                "--run-name demo --worker-year 2020 --worker-seed 43 "
                "--device-ids 0,1\n"
            )
        )
        with mock.patch.object(jobs.subprocess, "run", return_value=completed):
            processes = jobs._training_processes("demo")
        self.assertEqual(
            processes,
            [{
                "pid": 4321,
                "role": "worker",
                "forecast_year": 2020,
                "seed": 43,
            }],
        )

    def test_ddp_task_log_supplies_running_epoch_and_parent_pid(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            log_root = root / "data/quality/runs/demo/ddp_task_logs"
            log_root.mkdir(parents=True)
            (log_root / "forecast_2018.seed_42.log").write_text(
                "[2018/seed=42] epoch=007 loss=0.1\n", encoding="utf-8"
            )
            processes = [{
                "pid": 4321,
                "role": "worker",
                "forecast_year": 2018,
                "seed": 42,
            }]
            with mock.patch.object(
                jobs, "_training_processes", return_value=processes
            ):
                tasks = jobs._task_progress("demo", root)
        current = next(
            item for item in tasks
            if item["forecast_year"] == 2018 and item["seed"] == 42
        )
        self.assertEqual(current["state"], "running")
        self.assertEqual(current["last_epoch"], 7)
        self.assertEqual(current["pid"], 4321)

    def test_runner_records_successful_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job_path = root / "job.json"
            job_path.write_text(
                json.dumps(
                    {
                        "command": [sys.executable, "-c", "print('job-ok')"],
                        "cwd": str(root),
                        "status": "launching",
                    }
                ),
                encoding="utf-8",
            )
            code = jobs.run_job(job_path)
            payload = json.loads(job_path.read_text(encoding="utf-8"))

        self.assertEqual(code, 0)
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["exit_code"], 0)


if __name__ == "__main__":
    unittest.main()
